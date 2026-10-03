import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_query(query: str, desc: str):
    print(f"\n--- {desc} ---")
    print(f"Sending Query: '{query}'")
    start = time.time()
    resp = requests.post(f"{BASE_URL}/query", json={"query": query})
    elapsed = (time.time() - start) * 1000.0
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Status Code:   {resp.status_code}")
        print(f"  Model Used:    {data['model_used']}")
        print(f"  Cache Hit:     {data['cache_hit']}")
        print(f"  Escalated:     {data['escalated']}")
        print(f"  Latency (API): {data['latency_ms']} ms (client total: {elapsed:.1f} ms)")
        print(f"  Cost:          ${data['cost']:.6f}")
        print(f"  Answer Snippet:{data['answer'][:120]}...")
    else:
        print(f"  Error: {resp.status_code} - {resp.text}")

if __name__ == "__main__":
    print("Testing Root Endpoint...")
    r = requests.get(BASE_URL)
    print("Root response:", r.json())

    # 1. Simple query (Cache miss -> Small Model)
    test_query("What is Nemotron 3 Super?", "1. Simple Factual Query (Cache Miss)")

    # 2. Same query again (Cache Hit!)
    test_query("What is Nemotron 3 Super?", "2. Identical Query (Cache Hit Test)")

    # 3. Semantically similar query (Semantic Cache Hit!)
    test_query("Tell me about Nemotron 3 Super model", "3. Semantically Similar Query (Semantic Cache Hit Test)")

    # 4. Complex comparative query (Cost-Aware Router -> Large Model)
    test_query("Compare and contrast Mamba vs Transformer attention tradeoffs in high-throughput reasoning workloads.", "4. Complex Query (Cost-Aware Router -> Large Model)")
