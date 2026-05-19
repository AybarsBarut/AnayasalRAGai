from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = ROOT / "data" / "eval_queries.json"
DEFAULT_API_URL = "http://127.0.0.1:8000/api/v1/chat"


def main() -> int:
    api_url = os.getenv("ANAYASA_EVAL_API_URL", DEFAULT_API_URL)
    cases_path = Path(os.getenv("ANAYASA_EVAL_CASES", str(DEFAULT_CASES_PATH)))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    failures = 0
    print(f"Evaluating {len(cases)} cases against {api_url}")

    for case in cases:
        result = run_case(api_url, case)
        failures += 0 if result["passed"] else 1
        marker = "PASS" if result["passed"] else "FAIL"
        print(f"{marker} {case['id']}: {result['summary']}")

    if failures:
        print(f"{failures} evaluation case(s) failed.")
        return 1

    print("All evaluation cases passed.")
    return 0


def run_case(api_url: str, case: dict) -> dict:
    try:
        payload = post_json(api_url, {"query": case["query"]})
    except (HTTPError, URLError, TimeoutError) as exc:
        return {"passed": False, "summary": f"request failed: {exc}"}

    answer = str(payload.get("answer", ""))
    citations = payload.get("citations") or []
    missing_terms = [term for term in case.get("required_terms", []) if term.casefold() not in answer.casefold()]
    min_citations = int(case.get("min_citations", 0))

    if missing_terms:
        return {"passed": False, "summary": f"missing terms: {', '.join(missing_terms)}"}
    if len(citations) < min_citations:
        return {"passed": False, "summary": f"expected at least {min_citations} citation(s), got {len(citations)}"}
    if payload.get("confidence") == "needs_review":
        return {"passed": False, "summary": "confidence is needs_review"}

    return {
        "passed": True,
        "summary": f"confidence={payload.get('confidence')} citations={len(citations)}",
    }


def post_json(url: str, body: dict) -> dict:
    raw = json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    sys.exit(main())
