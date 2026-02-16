import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

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
CRITIC_MODEL = "qwen2.5:7b"
REWRITE_MODEL = "qwen2.5:7b"

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


# =========================
# LLM CLIENT (OLLAMA)
# =========================


def call_ollama(model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    resp = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


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


def web_search(session: requests.Session, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[ResearchSource]:
    """
    Uses DuckDuckGo HTML endpoints and ranks results by authority + recency + position.
    """

    def _parse_results(html: str) -> List[Tuple[str, str]]:
        patterns = [
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        ]
        for pat in patterns:
            matches = re.findall(pat, html, flags=re.IGNORECASE | re.DOTALL)
            if matches:
                return matches
        return []

    candidate_urls = [
        f"https://duckduckgo.com/html/?q={quote_plus(query)}",
        f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}",
    ]

    raw_matches: List[Tuple[str, str]] = []
    for url in candidate_urls:
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
        if "duckduckgo.com" in domain:
            continue
        score = domain_authority_score(href) + max(0.0, 1.2 - 0.12 * idx) + recency_score(title, "")
        ranked.append(
            ResearchSource(
                url=href,
                title=title or href,
                snippet="",
                domain=domain,
                score=score,
            )
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
    Full research flow:
    - planning (query generation)
    - search + ranking
    - page fetching with redirect support, timeout, retries, caching
    - scraping/extraction (text + targeted sections + regex captures)
    - source tracking per fact
    - multi-source verification + conflict flagging
    - stopping policy when enough verified facts are collected
    """
    session = build_http_session()
    cache = load_fact_cache()

    planned_queries = generate_search_queries(task)
    all_sources: List[ResearchSource] = []
    extracted_facts: List[ExtractedFact] = []
    aggregated_regex: List[str] = []
    aggregated_targeted = {"headings": [], "list_items": [], "table_cells": []}
    seen_urls: Set[str] = set()
    issued_queries: List[str] = []

    query_queue = list(planned_queries)

    while query_queue and len(issued_queries) < MAX_RESEARCH_QUERIES:
        query = query_queue.pop(0)
        issued_queries.append(query)

        try:
            ranked_results = web_search(session, query)
        except Exception:
            continue

        for source in ranked_results:
            if source.url in seen_urls:
                continue
            seen_urls.add(source.url)

            try:
                html = fetch_page_html(session, source.url, cache)
            except Exception:
                continue

            snippet, facts, targeted = extract_facts_from_page(source, html)
            source.snippet = snippet
            all_sources.append(source)
            extracted_facts.extend(facts)

            cleaned_text = strip_navigation_and_ads(html)
            aggregated_regex.extend(regex_capture_data(cleaned_text))
            for k in aggregated_targeted:
                aggregated_targeted[k].extend(targeted.get(k, []))

            verified = verify_facts(extracted_facts)
            if len(verified) >= 3 and len(all_sources) >= 3:
                # enough breadth and verification to stop early
                save_fact_cache(cache)
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
                }

        refined = refine_queries(task, all_sources, extracted_facts)
        for rq in refined:
            if rq.lower() not in {q.lower() for q in issued_queries + query_queue}:
                query_queue.append(rq)

    save_fact_cache(cache)
    verified = verify_facts(extracted_facts)

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


# =========================
# PROMPT BUILDERS
# =========================


def generator_prompt(task: str, guideline: str, research_context: str = "") -> str:
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
    if needs_research(task, guideline):
        research = research_web(task, guideline)
        research_context = research.get("context", "No reliable web sources retrieved.")
        log({"phase": "research", **research})

    # 1) Generate
    candidate = call_ollama(
        GEN_MODEL,
        generator_prompt(task, guideline, research_context=research_context),
        temperature=GEN_TEMP,
    )
    log({"phase": "generate", "model": GEN_MODEL, "text": candidate})

    last_violations: List[Violation] = []

    for i in range(1, MAX_ITERS + 1):
        ok, violations, parsed = hard_validate(candidate)
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
            return {
                "status": "PASS",
                "iterations": i,
                "final_text": candidate,
                "final_json": parsed,
                "last_violations": [],
                "log_path": log_path,
            }

        # 2) Critique
        critic_out = call_ollama(
            CRITIC_MODEL,
            critic_prompt(task, guideline, candidate, violations),
            temperature=CRITIC_TEMP,
        )
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
        rewritten = call_ollama(
            REWRITE_MODEL,
            rewriter_prompt(guideline, candidate, defects),
            temperature=REWRITE_TEMP,
        )
        log({"phase": "rewrite", "iter": i, "model": REWRITE_MODEL, "text": rewritten})

        # Optional drift guard: if rewrite changed too much, clamp scope to hard violations only
        if normalized_edit_ratio(candidate, rewritten) > MAX_REWRITE_EDIT_RATIO:
            defects_hard_only = {
                "violations_hard": defects.get("violations_hard", []),
                "violations_soft": [],
            }
            rewritten2 = call_ollama(
                REWRITE_MODEL,
                rewriter_prompt(guideline, candidate, defects_hard_only),
                temperature=0.1,
            )
            log({"phase": "rewrite_drift_guard", "iter": i, "text": rewritten2})
            rewritten = rewritten2

        candidate = rewritten

    # Final check after max iters
    ok, violations, parsed = hard_validate(candidate)
    return {
        "status": "PASS" if ok else "FAIL",
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

    result = self_revise(TASK, GUIDELINE)
    print("STATUS:", result["status"])
    print("ITERATIONS:", result["iterations"])
    if result["status"] != "PASS":
        print("LAST VIOLATIONS:", json.dumps(result["last_violations"], indent=2))
    print(result["final_text"])
    print(f"\nLog written to: {result['log_path']}")
