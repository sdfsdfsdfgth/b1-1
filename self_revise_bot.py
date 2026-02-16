import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse
from urllib.robotparser import RobotFileParser

import requests
from jsonschema import ValidationError, validate
from rapidfuzz.distance import Levenshtein
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
USER_AGENT = "self-revise-bot/1.0 (+https://example.local)"

GEN_MODEL = "llama3.1:8b"
CRITIC_MODEL = "llama3.1:8b"
REWRITE_MODEL = "llama3.1:8b"

MAX_ITERS = 5
CRITIC_TEMP = 0.1
REWRITE_TEMP = 0.2
GEN_TEMP = 0.6

# Optional: prevent rewrites from drifting too far
MAX_REWRITE_EDIT_RATIO = 0.40

# Example hard constraints
MAX_OUTPUT_CHARS = 2000
FORBIDDEN_PATTERNS = [
    r"\bas an ai\b",
    r"\bi think\b",
]

MAX_SEARCH_RESULTS = 8
MAX_RESEARCH_SOURCES = 6
MAX_SNIPPET_CHARS = 900
RESEARCH_TIMEOUT_SECS = 20
MAX_RETRIES = 2
MAX_FACTS = 12
MIN_MULTI_SOURCE_CONFIRMATIONS = 2
MAX_RESEARCH_QUERIES = 5
RESEARCH_CACHE_PATH = Path(".research_cache.json")
KNOWLEDGE_STORE_PATH = Path(".knowledge_store.json")
MIN_CLAIM_TOKEN_OVERLAP = 0.45
SEARCH_PROVIDER = "duckduckgo"  # duckduckgo | bing | google
MAX_TOTAL_SEARCH_CALLS = 6
MAX_TOTAL_PAGE_FETCHES = 18
MAX_PAGES_PER_QUERY = 5
REQUEST_DELAY_SECS = 0.2
FRESHNESS_MAX_AGE_YEARS = 3
OLLAMA_TIMEOUT_SECS = 300
OLLAMA_MAX_RETRIES = 3
OLLAMA_NUM_PREDICT = 600
MAX_CANDIDATE_FOR_CRITIC = 6000

AUTHORITATIVE_DOMAINS = {
    ".gov",
    ".edu",
    "wikipedia.org",
    "who.int",
    "un.org",
    "worldbank.org",
    "oecd.org",
    "europa.eu",
}

RESEARCH_KEYWORDS = [
    "latest",
    "current",
    "today",
    "price",
    "law",
    "regulation",
    "event",
    "date",
    "news",
    "source",
    "evidence",
    "research",
    "statistics",
    "up-to-date",
]

# Output contract: JSON with exact keys (adjust to your needs)
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["givens", "unknown", "equations", "solve", "solution"],
    "properties": {
        "givens": {"type": "array", "items": {"type": "string"}},
        "unknown": {"type": "array", "items": {"type": "string"}},
        "equations": {"type": "array", "items": {"type": "string"}},
        "solve": {"type": "array", "items": {"type": "string"}},
        "solution": {"type": "string"},
    },
    "additionalProperties": False,
}


# =========================
# DATA TYPES
# =========================


@dataclass
class Violation:
    rule_id: str
    message: str
    location: str


@dataclass
class ResearchSource:
    url: str
    title: str
    snippet: str
    domain: str
    score: float


@dataclass
class ExtractedFact:
    text: str
    source_url: str
    source_title: str


@dataclass
class VerifiedFact:
    text: str
    source_urls: List[str]
    conflict: bool = False


@dataclass
class SourceAwareViolation:
    rule_id: str
    location: str
    problem: str
    fix: str


@dataclass
class SearchAuditEntry:
    kind: str
    timestamp: float
    query: str = ""
    url: str = ""
    status: str = ""
    detail: str = ""


# =========================
# NETWORK / CACHE
# =========================


def build_http_session() -> requests.Session:
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_fact_cache() -> Dict[str, Dict[str, str]]:
    if not RESEARCH_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(RESEARCH_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_fact_cache(cache: Dict[str, Dict[str, str]]) -> None:
    RESEARCH_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def load_knowledge_store() -> Dict[str, Any]:
    if not KNOWLEDGE_STORE_PATH.exists():
        return {"facts": []}
    try:
        return json.loads(KNOWLEDGE_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"facts": []}


def save_knowledge_store(payload: Dict[str, Any]) -> None:
    KNOWLEDGE_STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# =========================
# LLM CLIENT (OLLAMA)
# =========================


def call_ollama(model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }

    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECS,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.ReadTimeout:
            if attempt == OLLAMA_MAX_RETRIES:
                raise
            time.sleep(2 * attempt)

    return ""


# =========================
# RESEARCH PIPELINE
# =========================


def needs_research(task: str, guideline: str) -> bool:
    text = f"{task}\n{guideline}".lower()
    if any(k in text for k in RESEARCH_KEYWORDS):
        return True
    return bool(re.search(r"\b(cite|citation|verify|verified|evidence|source)\b", text))


def domain_authority_score(url: str) -> float:
    domain = urlparse(url).netloc.lower()
    score = 0.0
    if any(domain.endswith(suffix) for suffix in AUTHORITATIVE_DOMAINS):
        score += 2.0
    if any(token in domain for token in ["gov", "edu", "org"]):
        score += 1.0
    return score


def recency_score(title: str, snippet: str) -> float:
    text = f"{title} {snippet}"
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return 0.0
    most_recent = max(years)
    delta = max(0, datetime.now().year - most_recent)
    return max(0.0, 1.5 - 0.25 * delta)


def is_fresh_enough(source: ResearchSource, max_age_years: int = FRESHNESS_MAX_AGE_YEARS) -> bool:
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", f"{source.title} {source.snippet}" or "")]
    if not years:
        return True
    return (datetime.now().year - max(years)) <= max_age_years


def can_fetch_url(robots_cache: Dict[str, RobotFileParser], url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception:
            return True
        robots_cache[base] = rp
    try:
        return robots_cache[base].can_fetch(USER_AGENT, url)
    except Exception:
        return True


def provider_urls(query: str) -> List[str]:
    q = quote_plus(query)
    if SEARCH_PROVIDER == "bing":
        return [f"https://www.bing.com/search?q={q}"]
    if SEARCH_PROVIDER == "google":
        return [f"https://www.google.com/search?q={q}"]
    return [
        f"https://duckduckgo.com/html/?q={q}",
        f"https://lite.duckduckgo.com/lite/?q={q}",
    ]


def web_search(session: requests.Session, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[ResearchSource]:
    """
    Uses configurable HTML search endpoints and ranks results by authority + recency + position.
    """

    def _parse_results(html: str) -> List[Tuple[str, str]]:
        patterns = [
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        ]
        for pat in patterns:
            matches = re.findall(pat, html, flags=re.IGNORECASE | re.DOTALL)
            if matches:
                return matches
        return []

    raw_matches: List[Tuple[str, str]] = []
    for url in provider_urls(query):
        try:
            resp = session.get(url, timeout=RESEARCH_TIMEOUT_SECS, allow_redirects=True)
            resp.raise_for_status()
            raw_matches = _parse_results(resp.text)
        except Exception:
            continue
        if raw_matches:
            break

    ranked: List[ResearchSource] = []
    for idx, (raw_href, raw_title) in enumerate(raw_matches, start=1):
        title = unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
        href = unescape(raw_href)
        if not href.startswith("http"):
            continue
        domain = urlparse(href).netloc.lower()
        if SEARCH_PROVIDER in domain:
            continue
        score = domain_authority_score(href) + max(0.0, 1.2 - 0.12 * idx) + recency_score(title, "")
        ranked.append(
            ResearchSource(url=href, title=title or href, snippet="", domain=domain, score=score)
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:max_results]


def fetch_page_html(session: requests.Session, url: str, cache: Dict[str, Dict[str, str]]) -> str:
    key = cache_key(url)
    if key in cache and "html" in cache[key]:
        return cache[key]["html"]

    resp = session.get(url, timeout=RESEARCH_TIMEOUT_SECS, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text

    cache[key] = {"url": url, "html": html}
    return html


def strip_navigation_and_ads(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<nav[\s\S]*?</nav>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<footer[\s\S]*?</footer>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<aside[\s\S]*?</aside>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_targeted_sections(html: str) -> Dict[str, List[str]]:
    headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html, flags=re.IGNORECASE | re.DOTALL)
    list_items = re.findall(r"<li[^>]*>(.*?)</li>", html, flags=re.IGNORECASE | re.DOTALL)
    table_cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, flags=re.IGNORECASE | re.DOTALL)

    def clean(items: List[str], max_items: int) -> List[str]:
        out: List[str] = []
        for raw in items[:max_items]:
            val = unescape(re.sub(r"<[^>]+>", " ", raw))
            val = re.sub(r"\s+", " ", val).strip()
            if val:
                out.append(val)
        return out

    return {
        "headings": clean(headings, 10),
        "list_items": clean(list_items, 20),
        "table_cells": clean(table_cells, 30),
    }


def regex_capture_data(text: str) -> List[str]:
    patterns = [
        r"\b\d+(?:\.\d+)?%\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:USD|EUR|GBP|\$)\s?\d+(?:,\d{3})*(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s?(?:million|billion|trillion)\b",
    ]
    captures: List[str] = []
    for pat in patterns:
        captures.extend(re.findall(pat, text, flags=re.IGNORECASE))
    return captures[:20]


def extract_facts_from_page(source: ResearchSource, html: str) -> Tuple[str, List[ExtractedFact], Dict[str, List[str]]]:
    clean_text = strip_navigation_and_ads(html)
    targeted = extract_targeted_sections(html)

    candidates = []
    candidates.extend(targeted["headings"][:4])
    candidates.extend(targeted["list_items"][:4])
    if clean_text:
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        candidates.extend(sentences[:6])

    facts: List[ExtractedFact] = []
    seen: Set[str] = set()
    for c in candidates:
        c2 = re.sub(r"\s+", " ", c).strip()
        if len(c2) < 20:
            continue
        key = c2.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(ExtractedFact(text=c2, source_url=source.url, source_title=source.title))
        if len(facts) >= 10:
            break

    snippet = clean_text[:MAX_SNIPPET_CHARS]
    return snippet, facts, targeted


def generate_search_queries(task: str) -> List[str]:
    base = task.strip()
    queries = [base]
    queries.append(f"{base} official source")
    queries.append(f"{base} latest data")
    queries.append(f"{base} statistics")
    queries.append(f"{base} site:.gov")

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            deduped.append(q)
    return deduped[:MAX_RESEARCH_QUERIES]


def refine_queries(task: str, collected_sources: List[ResearchSource], collected_facts: List[ExtractedFact]) -> List[str]:
    refined: List[str] = []

    # If authority is weak, bias toward official sources.
    if not any(domain_authority_score(s.url) >= 2.0 for s in collected_sources):
        refined.append(f"{task} official report")

    # Use top recurring keywords from extracted facts as refinements.
    tokens = re.findall(r"\b[a-zA-Z]{5,}\b", " ".join(f.text for f in collected_facts))
    freq: Dict[str, int] = {}
    for t in tokens:
        t_norm = t.lower()
        if t_norm in {"about", "which", "their", "there", "these", "those"}:
            continue
        freq[t_norm] = freq.get(t_norm, 0) + 1
    top_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
    for term, _ in top_terms:
        refined.append(f"{task} {term} source")

    return refined


def normalize_fact_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:14])


def detect_conflict(cluster: List[ExtractedFact]) -> bool:
    # heuristic: same anchor but different primary numbers across sources => conflict
    nums: Set[str] = set()
    for fact in cluster:
        number_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", fact.text)
        if number_tokens:
            nums.add(number_tokens[0])
    return len(nums) > 1


def verify_facts(facts: List[ExtractedFact]) -> List[VerifiedFact]:
    grouped: Dict[str, List[ExtractedFact]] = {}
    for fact in facts:
        key = normalize_fact_for_match(fact.text)
        grouped.setdefault(key, []).append(fact)

    verified: List[VerifiedFact] = []
    for _, group in grouped.items():
        unique_sources = sorted({g.source_url for g in group})
        if len(unique_sources) < MIN_MULTI_SOURCE_CONFIRMATIONS:
            continue
        verified.append(
            VerifiedFact(
                text=group[0].text,
                source_urls=unique_sources,
                conflict=detect_conflict(group),
            )
        )

    verified.sort(key=lambda f: (f.conflict, -len(f.source_urls)))
    return verified[:MAX_FACTS]


def build_research_context(
    sources: List[ResearchSource],
    verified_facts: List[VerifiedFact],
    regex_captures: List[str],
    targeted_sections: Dict[str, List[str]],
) -> str:
    lines: List[str] = []

    if sources:
        lines.append("SOURCES (ranked):")
        for i, s in enumerate(sources, 1):
            lines.append(f"[{i}] {s.title} | {s.url} | score={s.score:.2f}")

    lines.append("\nVERIFIED FACTS (must stay cited):")
    if verified_facts:
        for i, fact in enumerate(verified_facts, 1):
            conflict_note = " [CONFLICT]" if fact.conflict else ""
            lines.append(f"F{i}:{conflict_note} {fact.text}")
            lines.append("  Sources: " + ", ".join(fact.source_urls))
    else:
        lines.append("No facts verified across >=2 sources.")

    if regex_captures:
        lines.append("\nREGEX CAPTURES:")
        lines.append(", ".join(regex_captures[:20]))

    if targeted_sections.get("headings"):
        lines.append("\nTARGETED HEADINGS:")
        lines.extend(f"- {h}" for h in targeted_sections["headings"][:5])

    if targeted_sections.get("list_items"):
        lines.append("\nTARGETED LIST ITEMS:")
        lines.extend(f"- {li}" for li in targeted_sections["list_items"][:5])

    return "\n".join(lines).strip()


def research_web(task: str, guideline: str) -> Dict[str, Any]:
    """
    Research flow with search gating, budget limits, rate-limit friendliness,
    parallel page fetches, partial-failure reporting, and replayable audit logs.
    """
    session = build_http_session()
    cache = load_fact_cache()
    robots_cache: Dict[str, RobotFileParser] = {}

    planned_queries = generate_search_queries(task)
    all_sources: List[ResearchSource] = []
    extracted_facts: List[ExtractedFact] = []
    aggregated_regex: List[str] = []
    aggregated_targeted = {"headings": [], "list_items": [], "table_cells": []}
    seen_urls: Set[str] = set()
    issued_queries: List[str] = []
    audit_log: List[Dict[str, Any]] = []
    failures: List[str] = []

    query_queue = list(planned_queries)
    search_calls = 0
    page_fetches = 0

    while query_queue and len(issued_queries) < MAX_RESEARCH_QUERIES and search_calls < MAX_TOTAL_SEARCH_CALLS:
        query = query_queue.pop(0)
        issued_queries.append(query)
        search_calls += 1
        audit_log.append({"kind": "query", "ts": time.time(), "query": query, "status": "issued"})

        try:
            ranked_results = web_search(session, query, max_results=min(MAX_SEARCH_RESULTS, MAX_PAGES_PER_QUERY))
        except Exception as e:
            failures.append(f"query failed: {query} ({e})")
            audit_log.append({"kind": "query", "ts": time.time(), "query": query, "status": "failed", "detail": str(e)})
            continue

        ranked_results = [r for r in ranked_results if is_fresh_enough(r)]
        if not ranked_results:
            failures.append(f"no fresh sources for query: {query}")
            continue

        fetch_batch: List[ResearchSource] = []
        for source in ranked_results:
            if source.url in seen_urls or page_fetches >= MAX_TOTAL_PAGE_FETCHES:
                continue
            if not can_fetch_url(robots_cache, source.url):
                audit_log.append({"kind": "url", "ts": time.time(), "query": query, "url": source.url, "status": "blocked_robots"})
                continue
            seen_urls.add(source.url)
            fetch_batch.append(source)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch_page_html, session, src.url, cache): src for src in fetch_batch}
            for fut in as_completed(futures):
                if page_fetches >= MAX_TOTAL_PAGE_FETCHES:
                    break
                src = futures[fut]
                page_fetches += 1
                time.sleep(REQUEST_DELAY_SECS)
                try:
                    html = fut.result()
                except Exception as e:
                    failures.append(f"fetch failed: {src.url} ({e})")
                    audit_log.append({"kind": "url", "ts": time.time(), "query": query, "url": src.url, "status": "failed", "detail": str(e)})
                    continue

                snippet, facts, targeted = extract_facts_from_page(src, html)
                src.snippet = snippet
                all_sources.append(src)
                extracted_facts.extend(facts)
                cleaned_text = strip_navigation_and_ads(html)
                aggregated_regex.extend(regex_capture_data(cleaned_text))
                for k in aggregated_targeted:
                    aggregated_targeted[k].extend(targeted.get(k, []))

                audit_log.append({"kind": "url", "ts": time.time(), "query": query, "url": src.url, "status": "ok", "facts": len(facts)})

        verified = verify_facts(extracted_facts)
        if len(verified) >= 3 and len(all_sources) >= 3:
            break

        for rq in refine_queries(task, all_sources, extracted_facts):
            if rq.lower() not in {q.lower() for q in issued_queries + query_queue}:
                query_queue.append(rq)

    save_fact_cache(cache)
    verified = verify_facts(extracted_facts)
    store = load_knowledge_store()
    store.setdefault("facts", []).extend(
        {"text": v.text, "source_urls": v.source_urls, "conflict": v.conflict, "ts": time.time()}
        for v in verified
    )
    save_knowledge_store(store)

    return {
        "required": True,
        "queries": issued_queries,
        "sources": [s.__dict__ for s in all_sources[:MAX_RESEARCH_SOURCES]],
        "verified_facts": [v.__dict__ for v in verified],
        "context": build_research_context(
            all_sources[:MAX_RESEARCH_SOURCES],
            verified,
            aggregated_regex,
            aggregated_targeted,
        ),
        "audit": audit_log,
        "failures": failures,
        "partial": bool(failures),
        "budget": {
            "max_search_calls": MAX_TOTAL_SEARCH_CALLS,
            "used_search_calls": search_calls,
            "max_page_fetches": MAX_TOTAL_PAGE_FETCHES,
            "used_page_fetches": page_fetches,
        },
        "provider": SEARCH_PROVIDER,
    }



# =========================
# CORE HELPERS
# =========================


# =========================
# RESEARCH PIPELINE
# =========================


def needs_research(task: str, guideline: str) -> bool:
    text = f"{task}\n{guideline}".lower()
    if any(k in text for k in RESEARCH_KEYWORDS):
        return True
    return bool(re.search(r"\b(cite|citation|verify|verified|evidence|source)\b", text))


def domain_authority_score(url: str) -> float:
    domain = urlparse(url).netloc.lower()
    score = 0.0
    if any(domain.endswith(suffix) for suffix in AUTHORITATIVE_DOMAINS):
        score += 2.0
    if any(token in domain for token in ["gov", "edu", "org"]):
        score += 1.0
    return score


def recency_score(title: str, snippet: str) -> float:
    text = f"{title} {snippet}"
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return 0.0
    most_recent = max(years)
    delta = max(0, datetime.now().year - most_recent)
    return max(0.0, 1.5 - 0.25 * delta)


def is_fresh_enough(source: ResearchSource, max_age_years: int = FRESHNESS_MAX_AGE_YEARS) -> bool:
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", f"{source.title} {source.snippet}" or "")]
    if not years:
        return True
    return (datetime.now().year - max(years)) <= max_age_years


def can_fetch_url(robots_cache: Dict[str, RobotFileParser], url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception:
            return True
        robots_cache[base] = rp
    try:
        return robots_cache[base].can_fetch(USER_AGENT, url)
    except Exception:
        return True


def provider_urls(query: str) -> List[str]:
    q = quote_plus(query)
    if SEARCH_PROVIDER == "bing":
        return [f"https://www.bing.com/search?q={q}"]
    if SEARCH_PROVIDER == "google":
        return [f"https://www.google.com/search?q={q}"]
    return [
        f"https://duckduckgo.com/html/?q={q}",
        f"https://lite.duckduckgo.com/lite/?q={q}",
    ]


def web_search(session: requests.Session, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[ResearchSource]:
    """
    Uses configurable HTML search endpoints and ranks results by authority + recency + position.
    """

    def _parse_results(html: str) -> List[Tuple[str, str]]:
        patterns = [
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        ]
        for pat in patterns:
            matches = re.findall(pat, html, flags=re.IGNORECASE | re.DOTALL)
            if matches:
                return matches
        return []

    raw_matches: List[Tuple[str, str]] = []
    for url in provider_urls(query):
        try:
            resp = session.get(url, timeout=RESEARCH_TIMEOUT_SECS, allow_redirects=True)
            resp.raise_for_status()
            raw_matches = _parse_results(resp.text)
        except Exception:
            continue
        if raw_matches:
            break

    ranked: List[ResearchSource] = []
    for idx, (raw_href, raw_title) in enumerate(raw_matches, start=1):
        title = unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
        href = unescape(raw_href)
        if not href.startswith("http"):
            continue
        domain = urlparse(href).netloc.lower()
        if SEARCH_PROVIDER in domain:
            continue
        score = domain_authority_score(href) + max(0.0, 1.2 - 0.12 * idx) + recency_score(title, "")
        ranked.append(
            ResearchSource(url=href, title=title or href, snippet="", domain=domain, score=score)
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:max_results]


def fetch_page_html(session: requests.Session, url: str, cache: Dict[str, Dict[str, str]]) -> str:
    key = cache_key(url)
    if key in cache and "html" in cache[key]:
        return cache[key]["html"]

    resp = session.get(url, timeout=RESEARCH_TIMEOUT_SECS, allow_redirects=True)
    resp.raise_for_status()
    html = resp.text

    cache[key] = {"url": url, "html": html}
    return html


def strip_navigation_and_ads(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<nav[\s\S]*?</nav>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<footer[\s\S]*?</footer>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<aside[\s\S]*?</aside>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_targeted_sections(html: str) -> Dict[str, List[str]]:
    headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html, flags=re.IGNORECASE | re.DOTALL)
    list_items = re.findall(r"<li[^>]*>(.*?)</li>", html, flags=re.IGNORECASE | re.DOTALL)
    table_cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", html, flags=re.IGNORECASE | re.DOTALL)

    def clean(items: List[str], max_items: int) -> List[str]:
        out: List[str] = []
        for raw in items[:max_items]:
            val = unescape(re.sub(r"<[^>]+>", " ", raw))
            val = re.sub(r"\s+", " ", val).strip()
            if val:
                out.append(val)
        return out

    return {
        "headings": clean(headings, 10),
        "list_items": clean(list_items, 20),
        "table_cells": clean(table_cells, 30),
    }


def regex_capture_data(text: str) -> List[str]:
    patterns = [
        r"\b\d+(?:\.\d+)?%\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:USD|EUR|GBP|\$)\s?\d+(?:,\d{3})*(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s?(?:million|billion|trillion)\b",
    ]
    captures: List[str] = []
    for pat in patterns:
        captures.extend(re.findall(pat, text, flags=re.IGNORECASE))
    return captures[:20]


def extract_facts_from_page(source: ResearchSource, html: str) -> Tuple[str, List[ExtractedFact], Dict[str, List[str]]]:
    clean_text = strip_navigation_and_ads(html)
    targeted = extract_targeted_sections(html)

    candidates = []
    candidates.extend(targeted["headings"][:4])
    candidates.extend(targeted["list_items"][:4])
    if clean_text:
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        candidates.extend(sentences[:6])

    facts: List[ExtractedFact] = []
    seen: Set[str] = set()
    for c in candidates:
        c2 = re.sub(r"\s+", " ", c).strip()
        if len(c2) < 20:
            continue
        key = c2.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(ExtractedFact(text=c2, source_url=source.url, source_title=source.title))
        if len(facts) >= 10:
            break

    snippet = clean_text[:MAX_SNIPPET_CHARS]
    return snippet, facts, targeted


def generate_search_queries(task: str) -> List[str]:
    base = task.strip()
    queries = [base]
    queries.append(f"{base} official source")
    queries.append(f"{base} latest data")
    queries.append(f"{base} statistics")
    queries.append(f"{base} site:.gov")

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            deduped.append(q)
    return deduped[:MAX_RESEARCH_QUERIES]


def refine_queries(task: str, collected_sources: List[ResearchSource], collected_facts: List[ExtractedFact]) -> List[str]:
    refined: List[str] = []

    # If authority is weak, bias toward official sources.
    if not any(domain_authority_score(s.url) >= 2.0 for s in collected_sources):
        refined.append(f"{task} official report")

    # Use top recurring keywords from extracted facts as refinements.
    tokens = re.findall(r"\b[a-zA-Z]{5,}\b", " ".join(f.text for f in collected_facts))
    freq: Dict[str, int] = {}
    for t in tokens:
        t_norm = t.lower()
        if t_norm in {"about", "which", "their", "there", "these", "those"}:
            continue
        freq[t_norm] = freq.get(t_norm, 0) + 1
    top_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
    for term, _ in top_terms:
        refined.append(f"{task} {term} source")

    return refined


def normalize_fact_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:14])


def detect_conflict(cluster: List[ExtractedFact]) -> bool:
    # heuristic: same anchor but different primary numbers across sources => conflict
    nums: Set[str] = set()
    for fact in cluster:
        number_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", fact.text)
        if number_tokens:
            nums.add(number_tokens[0])
    return len(nums) > 1


def verify_facts(facts: List[ExtractedFact]) -> List[VerifiedFact]:
    grouped: Dict[str, List[ExtractedFact]] = {}
    for fact in facts:
        key = normalize_fact_for_match(fact.text)
        grouped.setdefault(key, []).append(fact)

    verified: List[VerifiedFact] = []
    for _, group in grouped.items():
        unique_sources = sorted({g.source_url for g in group})
        if len(unique_sources) < MIN_MULTI_SOURCE_CONFIRMATIONS:
            continue
        verified.append(
            VerifiedFact(
                text=group[0].text,
                source_urls=unique_sources,
                conflict=detect_conflict(group),
            )
        )

    verified.sort(key=lambda f: (f.conflict, -len(f.source_urls)))
    return verified[:MAX_FACTS]


def build_research_context(
    sources: List[ResearchSource],
    verified_facts: List[VerifiedFact],
    regex_captures: List[str],
    targeted_sections: Dict[str, List[str]],
) -> str:
    lines: List[str] = []

    if sources:
        lines.append("SOURCES (ranked):")
        for i, s in enumerate(sources, 1):
            lines.append(f"[{i}] {s.title} | {s.url} | score={s.score:.2f}")

    lines.append("\nVERIFIED FACTS (must stay cited):")
    if verified_facts:
        for i, fact in enumerate(verified_facts, 1):
            conflict_note = " [CONFLICT]" if fact.conflict else ""
            lines.append(f"F{i}:{conflict_note} {fact.text}")
            lines.append("  Sources: " + ", ".join(fact.source_urls))
    else:
        lines.append("No facts verified across >=2 sources.")

    if regex_captures:
        lines.append("\nREGEX CAPTURES:")
        lines.append(", ".join(regex_captures[:20]))

    if targeted_sections.get("headings"):
        lines.append("\nTARGETED HEADINGS:")
        lines.extend(f"- {h}" for h in targeted_sections["headings"][:5])

    if targeted_sections.get("list_items"):
        lines.append("\nTARGETED LIST ITEMS:")
        lines.extend(f"- {li}" for li in targeted_sections["list_items"][:5])

    return "\n".join(lines).strip()


def research_web(task: str, guideline: str) -> Dict[str, Any]:
    """
    Research flow with search gating, budget limits, rate-limit friendliness,
    parallel page fetches, partial-failure reporting, and replayable audit logs.
    """
    session = build_http_session()
    cache = load_fact_cache()
    robots_cache: Dict[str, RobotFileParser] = {}

    planned_queries = generate_search_queries(task)
    all_sources: List[ResearchSource] = []
    extracted_facts: List[ExtractedFact] = []
    aggregated_regex: List[str] = []
    aggregated_targeted = {"headings": [], "list_items": [], "table_cells": []}
    seen_urls: Set[str] = set()
    issued_queries: List[str] = []
    audit_log: List[Dict[str, Any]] = []
    failures: List[str] = []

    query_queue = list(planned_queries)
    search_calls = 0
    page_fetches = 0

    while query_queue and len(issued_queries) < MAX_RESEARCH_QUERIES and search_calls < MAX_TOTAL_SEARCH_CALLS:
        query = query_queue.pop(0)
        issued_queries.append(query)
        search_calls += 1
        audit_log.append({"kind": "query", "ts": time.time(), "query": query, "status": "issued"})

        try:
            ranked_results = web_search(session, query, max_results=min(MAX_SEARCH_RESULTS, MAX_PAGES_PER_QUERY))
        except Exception as e:
            failures.append(f"query failed: {query} ({e})")
            audit_log.append({"kind": "query", "ts": time.time(), "query": query, "status": "failed", "detail": str(e)})
            continue

        ranked_results = [r for r in ranked_results if is_fresh_enough(r)]
        if not ranked_results:
            failures.append(f"no fresh sources for query: {query}")
            continue

        fetch_batch: List[ResearchSource] = []
        for source in ranked_results:
            if source.url in seen_urls or page_fetches >= MAX_TOTAL_PAGE_FETCHES:
                continue
            if not can_fetch_url(robots_cache, source.url):
                audit_log.append({"kind": "url", "ts": time.time(), "query": query, "url": source.url, "status": "blocked_robots"})
                continue
            seen_urls.add(source.url)
            fetch_batch.append(source)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch_page_html, session, src.url, cache): src for src in fetch_batch}
            for fut in as_completed(futures):
                if page_fetches >= MAX_TOTAL_PAGE_FETCHES:
                    break
                src = futures[fut]
                page_fetches += 1
                time.sleep(REQUEST_DELAY_SECS)
                try:
                    html = fut.result()
                except Exception as e:
                    failures.append(f"fetch failed: {src.url} ({e})")
                    audit_log.append({"kind": "url", "ts": time.time(), "query": query, "url": src.url, "status": "failed", "detail": str(e)})
                    continue

                snippet, facts, targeted = extract_facts_from_page(src, html)
                src.snippet = snippet
                all_sources.append(src)
                extracted_facts.extend(facts)
                cleaned_text = strip_navigation_and_ads(html)
                aggregated_regex.extend(regex_capture_data(cleaned_text))
                for k in aggregated_targeted:
                    aggregated_targeted[k].extend(targeted.get(k, []))

                audit_log.append({"kind": "url", "ts": time.time(), "query": query, "url": src.url, "status": "ok", "facts": len(facts)})

        verified = verify_facts(extracted_facts)
        if len(verified) >= 3 and len(all_sources) >= 3:
            break

        for rq in refine_queries(task, all_sources, extracted_facts):
            if rq.lower() not in {q.lower() for q in issued_queries + query_queue}:
                query_queue.append(rq)

    save_fact_cache(cache)
    verified = verify_facts(extracted_facts)
    store = load_knowledge_store()
    store.setdefault("facts", []).extend(
        {"text": v.text, "source_urls": v.source_urls, "conflict": v.conflict, "ts": time.time()}
        for v in verified
    )
    save_knowledge_store(store)

    return {
        "required": True,
        "queries": issued_queries,
        "sources": [s.__dict__ for s in all_sources[:MAX_RESEARCH_SOURCES]],
        "verified_facts": [v.__dict__ for v in verified],
        "context": build_research_context(
            all_sources[:MAX_RESEARCH_SOURCES],
            verified,
            aggregated_regex,
            aggregated_targeted,
        ),
        "audit": audit_log,
        "failures": failures,
        "partial": bool(failures),
        "budget": {
            "max_search_calls": MAX_TOTAL_SEARCH_CALLS,
            "used_search_calls": search_calls,
            "max_page_fetches": MAX_TOTAL_PAGE_FETCHES,
            "used_page_fetches": page_fetches,
        },
        "provider": SEARCH_PROVIDER,
    }



# =========================
# CORE HELPERS
# =========================


def extract_json_object(text: str) -> Dict[str, Any]:
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        raise ValueError("Response must be ONLY a JSON object with no extra text.")
    return json.loads(s)


def normalized_edit_ratio(a: str, b: str) -> float:
    dist = Levenshtein.distance(a, b)
    denom = max(1, max(len(a), len(b)))
    return dist / denom


# =========================
# VALIDATORS (HARD RULES)
# =========================


def hard_validate(raw_text: str) -> Tuple[bool, List[Violation], Optional[Dict[str, Any]]]:
    violations: List[Violation] = []
    s = raw_text.strip()

    if not (s.startswith("{") and s.endswith("}")):
        violations.append(Violation("HR-05", "Extra text outside JSON object", "$"))

    if len(raw_text) > MAX_OUTPUT_CHARS:
        violations.append(
            Violation("HR-01", f"Output too long (> {MAX_OUTPUT_CHARS} chars)", "$")
        )

    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, raw_text, flags=re.IGNORECASE):
            violations.append(Violation("HR-02", f"Forbidden pattern matched: {pat}", "$"))

    try:
        obj = extract_json_object(raw_text)
    except Exception as e:
        violations.append(Violation("HR-03", f"Invalid JSON / cannot parse: {e}", "$"))
        return False, violations, None

    try:
        validate(instance=obj, schema=OUTPUT_SCHEMA)
    except ValidationError as e:
        violations.append(Violation("HR-04", f"Schema violation: {e.message}", "$"))
        return False, violations, obj

    return len(violations) == 0, violations, obj



def build_structured_research_notes(research: Dict[str, Any]) -> Dict[str, Any]:
    facts = research.get("verified_facts", [])
    sources = research.get("sources", [])
    return {
        "required": bool(research.get("required", False)),
        "queries": research.get("queries", []),
        "fact_count": len(facts),
        "source_count": len(sources),
        "audit": research.get("audit", []),
        "failures": research.get("failures", []),
        "budget": research.get("budget", {}),
        "provider": research.get("provider", SEARCH_PROVIDER),
        "partial": bool(research.get("partial", False)),
        "facts": [
            {
                "text": f.get("text", ""),
                "source_urls": f.get("source_urls", []),
                "conflict": bool(f.get("conflict", False)),
            }
            for f in facts
        ],
        "sources": [
            {
                "url": s.get("url", ""),
                "title": s.get("title", ""),
                "domain": s.get("domain", ""),
                "score": s.get("score", 0),
            }
            for s in sources
        ],
    }


def citations_required(task: str, guideline: str) -> bool:
    return bool(re.search(r"\b(cite|citation|citations|required source|sources required)\b", f"{task}\n{guideline}", flags=re.IGNORECASE))


def allowed_to_search(task: str, guideline: str) -> Tuple[bool, str]:
    if needs_research(task, guideline):
        return True, "research-trigger keywords present"
    return False, "search gating blocked web lookup"


def tokenize_claim(text: str) -> Set[str]:
    tokens = re.findall(r"\b[a-z0-9]{3,}\b", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "are", "was"}
    return {t for t in tokens if t not in stop}


def claim_support_score(claim: str, fact: str) -> float:
    c = tokenize_claim(claim)
    f = tokenize_claim(fact)
    if not c or not f:
        return 0.0
    overlap = len(c & f)
    return overlap / max(1, len(c))


def extract_candidate_claims(parsed_obj: Dict[str, Any]) -> List[Tuple[str, str]]:
    claims: List[Tuple[str, str]] = []
    for key in ["givens", "unknown", "equations", "solve"]:
        values = parsed_obj.get(key, [])
        if isinstance(values, list):
            for idx, value in enumerate(values):
                if isinstance(value, str) and value.strip():
                    claims.append((f"$.{key}[{idx}]", value.strip()))
    solution = parsed_obj.get("solution", "")
    if isinstance(solution, str):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", solution) if s.strip()]
        if not sentences and solution.strip():
            sentences = [solution.strip()]
        for idx, sentence in enumerate(sentences):
            claims.append((f"$.solution[{idx}]", sentence))
    return claims


def map_claims_to_verified_facts(parsed_obj: Optional[Dict[str, Any]], research_notes: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not parsed_obj:
        return []
    mapped: List[Dict[str, Any]] = []
    facts = research_notes.get("facts", [])
    for location, claim in extract_candidate_claims(parsed_obj):
        best_score = 0.0
        best_fact: Dict[str, Any] = {}
        for fact in facts:
            score = claim_support_score(claim, fact.get("text", ""))
            if score > best_score:
                best_score = score
                best_fact = fact
        mapped.append({
            "location": location,
            "claim": claim,
            "matched_fact": best_fact.get("text", ""),
            "score": best_score,
            "source_urls": best_fact.get("source_urls", []),
        })
    return mapped


def citation_violations(parsed_obj: Optional[Dict[str, Any]], research_notes: Dict[str, Any], required: bool) -> List[Violation]:
    if not required or not parsed_obj:
        return []
    out: List[Violation] = []
    for row in map_claims_to_verified_facts(parsed_obj, research_notes):
        if len(tokenize_claim(row["claim"])) < 5:
            continue
        if row["score"] < MIN_CLAIM_TOKEN_OVERLAP or not row["source_urls"]:
            out.append(Violation("RV-04", "Claim lacks citation-backed support.", row["location"]))
    return out


def source_aware_critic(parsed_obj: Optional[Dict[str, Any]], research_notes: Dict[str, Any], citation_is_required: bool) -> List[SourceAwareViolation]:
    if not parsed_obj:
        return []

    facts = research_notes.get("facts", [])
    fact_texts = [f.get("text", "") for f in facts if f.get("text")]

    violations: List[SourceAwareViolation] = []
    for location, claim in extract_candidate_claims(parsed_obj):
        has_numeric_signal = bool(re.search(r"\b\d", claim))
        likely_factual = has_numeric_signal or len(tokenize_claim(claim)) >= 6
        if not likely_factual:
            continue

        best_score = max((claim_support_score(claim, ft) for ft in fact_texts), default=0.0)
        if best_score < MIN_CLAIM_TOKEN_OVERLAP:
            violations.append(
                SourceAwareViolation(
                    rule_id="RV-03",
                    location=location,
                    problem="Claim is not grounded in research notes; potential hallucinated data.",
                    fix="Rewrite the claim using only verified facts from research notes, or remove it.",
                )
            )
        elif citation_is_required and not facts:
            violations.append(
                SourceAwareViolation(
                    rule_id="RV-02",
                    location=location,
                    problem="Citation/source support required but no verified sources were available.",
                    fix="Only include claims that can be backed by sourced research notes.",
                )
            )

    return violations


def validate_with_research(
    raw_text: str,
    task: str,
    guideline: str,
    research_notes: Optional[Dict[str, Any]],
) -> Tuple[bool, List[Violation], Optional[Dict[str, Any]]]:
    ok, violations, parsed = hard_validate(raw_text)
    if not parsed or not research_notes or not research_notes.get("required"):
        return ok, violations, parsed

    citation_is_required = citations_required(task, guideline)
    source_violations = source_aware_critic(parsed, research_notes, citation_is_required)
    violations.extend(
        Violation(
            rule_id=v.rule_id,
            message=v.problem,
            location=v.location,
        )
        for v in source_violations
    )
    violations.extend(citation_violations(parsed, research_notes, citation_is_required))

    for mapping in map_claims_to_verified_facts(parsed, research_notes):
        if len(tokenize_claim(mapping["claim"])) < 5:
            continue
        if mapping["score"] < MIN_CLAIM_TOKEN_OVERLAP:
            violations.append(Violation("RV-05", "Claim could not be aligned to verified facts.", mapping["location"]))

    return len(violations) == 0, violations, parsed

# =========================
# PROMPT BUILDERS
# =========================


def generator_prompt(
    task: str,
    guideline: str,
    research_context: str = "",
    research_notes: Optional[Dict[str, Any]] = None,
) -> str:
    return f"""
SYSTEM:
You are the GENERATOR. Produce the final answer for the task.

RESEARCH AND FACTUALITY REQUIREMENTS:
- If web research context is provided below, use it for factual grounding.
- Use only facts that can be tied to explicit sources in the provided context.
- Do not invent facts that are not supported by task context or research context.
- If a fact is marked [CONFLICT], state uncertainty in the JSON content.
- If no verified facts exist, avoid specific claims requiring citations.

NON-NEGOTIABLE OUTPUT CONTRACT:
- Output MUST be a single JSON object and NOTHING ELSE.
- Do NOT include markdown, backticks, code fences, comments, or explanations.
- Do NOT include any text before '{{' or after '}}'.
- Keys MUST be exactly: "givens", "unknown", "equations", "solve", "solution".
- No additional keys. No nulls. No empty strings.

HARD RULES (MUST PASS):
{guideline}

TASK:
{task}

WEB RESEARCH CONTEXT (if any):
{research_context}

STRUCTURED RESEARCH NOTES (JSON):
{json.dumps(research_notes or {}, ensure_ascii=False, indent=2)}

INTERNAL SELF-CHECK BEFORE YOU OUTPUT:
1) Is the output valid JSON? (parseable)
2) Does it contain ONLY the 5 keys? (no extras)
3) Do types match? (arrays of strings; solution is string)
4) No forbidden phrases; no extra text outside JSON.

Now output the JSON object only.
""".strip()


def critic_prompt(
    task: str,
    guideline: str,
    candidate: str,
    validator_violations: List[Violation],
    research_notes: Optional[Dict[str, Any]] = None,
) -> str:
    v = [vi.__dict__ for vi in validator_violations]
    return f"""
SYSTEM:
You are the CRITIC. You do NOT rewrite. You ONLY diagnose violations.

NON-NEGOTIABLE OUTPUT CONTRACT:
Return ONE JSON object and NOTHING ELSE (no markdown, no backticks, no extra text).
The JSON MUST follow this exact schema:
{{
  "violations_hard": [
    {{"rule_id":"HR-xx or guideline id","location":"$.path or 'raw_text'","problem":"...","fix":"..."}}
  ],
  "violations_soft": [
    {{"rule_id":"SR-xx","location":"$.path or 'raw_text'","problem":"...","fix":"..."}}
  ]
}}

RULES FOR YOUR DIAGNOSIS:
- Every item MUST cite a real rule_id from the guideline or HR-01..HR-05.
- Location must be specific (JSONPath like $.solution or $.givens[0]) when possible.
- Only include violations that are actually present.
- The "fix" must be concrete and minimal (tell the rewriter exactly what to change).
- If deterministic validator violations exist, you MUST include them in violations_hard.

GUIDELINE (WITH RULE IDS):
{guideline}

TASK:
{task}

CANDIDATE (may be invalid / may have extra text):
{candidate}

DETERMINISTIC VALIDATOR VIOLATIONS (AUTHORITATIVE):
{json.dumps(v, indent=2)}

STRUCTURED RESEARCH NOTES (for source-aware checks):
{json.dumps(research_notes or {}, ensure_ascii=False, indent=2)}

SOURCE-AWARE CRITIC REQUIREMENTS:
- Flag a hard violation when a factual claim cannot be tied to verified facts in research notes.
- Flag a hard violation when citations are required but a claim lacks source support in research notes.
- Flag hard violations for hallucinated numbers/dates not grounded in verified facts.

Now output the JSON defect list only.
""".strip()


def rewriter_prompt(guideline: str, candidate: str, defects: Dict[str, Any]) -> str:
    return f"""
SYSTEM:
You are the REWRITER. Apply FIXES ONLY. Do not improve style unless required by a listed violation.

NON-NEGOTIABLE OUTPUT CONTRACT:
- Output MUST be a single JSON object and NOTHING ELSE.
- No markdown, no backticks, no code fences, no commentary.
- Do NOT include any text before '{{' or after '}}'.
- Keys MUST be exactly: "givens", "unknown", "equations", "solve", "solution".
- No additional keys. No nulls.

STRICT REWRITE RULES:
1) Only change fields necessary to fix the listed violations.
2) Preserve all other content verbatim.
3) If the candidate is invalid JSON, reconstruct it into valid JSON while preserving content as much as possible.
4) Do not introduce new information not required for compliance.

Guideline:
{guideline}

Candidate:
{candidate}

Defect list (apply these fixes exactly):
{json.dumps(defects, indent=2)}

INTERNAL SELF-CHECK BEFORE OUTPUT:
- Valid JSON only
- Exactly 5 keys
- Types correct
- No extra text outside JSON
- All listed hard violations fixed

Now output the corrected JSON only.
""".strip()


# =========================
# ORCHESTRATOR
# =========================


def interrupted_result(reason: str, log_path: str) -> Dict[str, Any]:
    return {
        "status": "INTERRUPTED",
        "iterations": 0,
        "final_text": "",
        "final_json": None,
        "last_violations": [],
        "reason": reason,
        "log_path": log_path,
    }


def self_revise(task: str, guideline: str, log_path: str = "run_log.jsonl") -> Dict[str, Any]:
    """
    Returns dict with final_text, status, iterations, last_violations, and final_json (if parseable).
    Writes JSONL logs for debugging/replay.
    """

    def log(event: Dict[str, Any]) -> None:
        event["ts"] = time.time()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    research_context = "No external research required."
    research_notes: Dict[str, Any] = {"required": False, "facts": [], "sources": []}

    # 1) Research phase (pre-generation)
    search_allowed, search_reason = allowed_to_search(task, guideline)
    log({"phase": "search_gate", "allowed": search_allowed, "reason": search_reason})
    if search_allowed:
        print("\n🔍 Starting web research...")
        try:
            research = research_web(task, guideline)
            print(f"✅ Research complete: {len(research.get('sources', []))} sources, {len(research.get('verified_facts', []))} verified facts")
        except KeyboardInterrupt:
            log({"phase": "interrupted", "stage": "research"})
            return interrupted_result("Interrupted during research phase.", log_path)

        research_context = research.get("context", "No reliable web sources retrieved.")
        research_notes = build_structured_research_notes(research)
        log({"phase": "research", **research})
        log({"phase": "research_notes", "notes": research_notes})

    # 2) Generate
    print("\n📝 Generating initial response...")
    try:
        candidate = call_ollama(
            GEN_MODEL,
            generator_prompt(
                task,
                guideline,
                research_context=research_context,
                research_notes=research_notes,
            ),
            temperature=GEN_TEMP,
        )
        print("✅ Generation complete")
    except KeyboardInterrupt:
        log({"phase": "interrupted", "stage": "generate"})
        return interrupted_result("Interrupted during generation phase.", log_path)
    log({"phase": "generate", "model": GEN_MODEL, "text": candidate})

    last_violations: List[Violation] = []

    for i in range(1, MAX_ITERS + 1):
        print(f"\n🔄 Iteration {i}/{MAX_ITERS}: Validating...")
        ok, violations, parsed = validate_with_research(candidate, task, guideline, research_notes)
        last_violations = violations
        log(
            {
                "phase": "validate",
                "iter": i,
                "ok": ok,
                "violations": [v.__dict__ for v in violations],
            }
        )

        if ok:
            print(f"✅ PASS at iteration {i}!")
            return {
                "status": "PASS",
                "iterations": i,
                "final_text": candidate,
                "final_json": parsed,
                "last_violations": [],
                "log_path": log_path,
            }

        # 2) Critique
        print(f"   → Found {len(violations)} violations. Critiquing...")
        if len(candidate) > MAX_CANDIDATE_FOR_CRITIC:
            candidate_for_critic = candidate[:MAX_CANDIDATE_FOR_CRITIC] + "\n...[TRUNCATED]..."
        else:
            candidate_for_critic = candidate

        try:
            critic_out = call_ollama(
                CRITIC_MODEL,
                critic_prompt(task, guideline, candidate_for_critic, violations, research_notes=research_notes),
                temperature=CRITIC_TEMP,
            )
        except KeyboardInterrupt:
            log({"phase": "interrupted", "stage": "critic", "iter": i})
            return interrupted_result("Interrupted during critic phase.", log_path)
        log({"phase": "critic_raw", "iter": i, "model": CRITIC_MODEL, "text": critic_out})

        try:
            defects = extract_json_object(critic_out)
        except Exception:
            # fallback: use validator violations only
            defects = {
                "violations_hard": [
                    {
                        "rule_id": v.rule_id,
                        "location": v.location,
                        "problem": v.message,
                        "fix": "Fix this issue.",
                    }
                    for v in violations
                ],
                "violations_soft": [],
            }

        log({"phase": "critic_json", "iter": i, "defects": defects})

        # 3) Rewrite
        print(f"   → Rewriting based on feedback...")
        try:
            rewritten = call_ollama(
                REWRITE_MODEL,
                rewriter_prompt(guideline, candidate, defects),
                temperature=REWRITE_TEMP,
            )
        except KeyboardInterrupt:
            log({"phase": "interrupted", "stage": "rewrite", "iter": i})
            return interrupted_result("Interrupted during rewrite phase.", log_path)
        log({"phase": "rewrite", "iter": i, "model": REWRITE_MODEL, "text": rewritten})

        # Optional drift guard: if rewrite changed too much, clamp scope to hard violations only
        if normalized_edit_ratio(candidate, rewritten) > MAX_REWRITE_EDIT_RATIO:
            defects_hard_only = {
                "violations_hard": defects.get("violations_hard", []),
                "violations_soft": [],
            }
            try:
                rewritten2 = call_ollama(
                    REWRITE_MODEL,
                    rewriter_prompt(guideline, candidate, defects_hard_only),
                    temperature=0.1,
                )
            except KeyboardInterrupt:
                log({"phase": "interrupted", "stage": "rewrite_drift_guard", "iter": i})
                return interrupted_result("Interrupted during rewrite drift guard phase.", log_path)
            log({"phase": "rewrite_drift_guard", "iter": i, "text": rewritten2})
            rewritten = rewritten2

        candidate = rewritten

    # Final check after max iters
    print(f"\n⚠️  Reached max iterations ({MAX_ITERS}). Final validation...")
    ok, violations, parsed = validate_with_research(candidate, task, guideline, research_notes)
    status = "PASS" if ok else "FAIL"
    print(f"\n📊 FINAL STATUS: {status}")
    return {
        "status": status,
        "iterations": MAX_ITERS,
        "final_text": candidate,
        "final_json": parsed,
        "last_violations": [v.__dict__ for v in violations],
        "log_path": log_path,
    }


# =========================
# EXAMPLE USAGE
# =========================

if __name__ == "__main__":
    GUIDELINE = """
HR-01 Output must be <= 2000 characters.
HR-02 Must not include 'I think' or 'as an AI'.
HR-03 Must be valid JSON (single object).
HR-04 Must match schema exactly (no extra keys).
HR-05 Must not include extra text outside the JSON object.
SR-01 Prefer short bullet items.
""".strip()

    TASK = "Explain Newton's 2nd law and give one numeric example."

    try:
        result = self_revise(TASK, GUIDELINE)
    except KeyboardInterrupt:
        print("STATUS: INTERRUPTED")
        print("Run interrupted by user.")
    else:
        print("STATUS:", result["status"])
        print("ITERATIONS:", result["iterations"])
        if result["status"] not in {"PASS", "INTERRUPTED"}:
            print("LAST VIOLATIONS:", json.dumps(result["last_violations"], indent=2))
        if result.get("reason"):
            print("REASON:", result["reason"])
        if result.get("final_text"):
            print(result["final_text"])
        print(f"\nLog written to: {result['log_path']}")
