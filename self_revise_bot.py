import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from jsonschema import ValidationError, validate
from rapidfuzz.distance import Levenshtein


# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"

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


# =========================
# LLM CLIENT (OLLAMA)
# =========================


def call_ollama(model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


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


def generator_prompt(task: str, guideline: str) -> str:
    return f"""
SYSTEM:
You are the GENERATOR. Produce the final answer for the task.

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

    # 1) Generate
    candidate = call_ollama(GEN_MODEL, generator_prompt(task, guideline), temperature=GEN_TEMP)
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
