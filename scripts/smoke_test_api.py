"""End-to-end smoke test against a running API (uvicorn semantic_rag.api:app --port 8000).

Exits non-zero if any expectation fails.  Usage:  python scripts/smoke_test_api.py [base_url]
"""
import sys

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

CASES = [
    # (question, expected status set, expected document or None)
    ("What were Apple's total net sales in fiscal year 2025?", {"answered", "cache_hit"}, "04_Apple_10K_FY2025.htm"),
    ("What were Apple's total net sales in fiscal year 2025?", {"cache_hit"}, "04_Apple_10K_FY2025.htm"),
    ("What was Microsoft's total revenue in fiscal year 2025?", {"answered", "cache_hit"}, "06_Microsoft_10K_FY2025.htm"),
    ("Wie viele Tage im Jahr dürfen internationale Studierende in Deutschland arbeiten?", {"answered", "cache_hit"}, None),
    ("Which earlier RFC does RFC 9111 obsolete?", {"answered", "cache_hit"}, "08_RFC_9111_HTTP_Caching.html"),
    ("What is the capital of Australia?", {"not_found"}, None),
    ("What were Apple's total net sales in fiscal year 2030?", {"not_found"}, None),
]


def main() -> int:
    print("Root:", requests.get(BASE_URL, timeout=10).json())
    print("Health:", requests.get(f"{BASE_URL}/health", timeout=10).json())
    failures = 0
    for question, statuses, document in CASES:
        r = requests.post(f"{BASE_URL}/query", json={"query": question}, timeout=180)
        data = r.json() if r.status_code == 200 else {}
        ok = r.status_code == 200 and data["status"] in statuses and (document is None or data["primary_document"] == document)
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question}\n       status={data.get('status')} "
              f"doc={data.get('primary_document')} model={data.get('model_used')} "
              f"{data.get('latency_ms')} ms ${data.get('cost')}\n       {str(data.get('answer', r.text))[:160]}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
