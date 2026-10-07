import os
import sys
import time

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_rag import retrieve_chunks
from cache import check_cache, put_cache, init_cache_collection
from router import route_and_generate

def run_tests():
    print("\n" + "="*60)
    print("  TEST 1: Cross-Lingual Chunk Retrieval")
    print("="*60)
    en_query = "Can international students work in Germany and for how many days?"
    print(f"Query (English): '{en_query}'")
    chunks = retrieve_chunks(en_query, top_k=3)
    print(f"Retrieved {len(chunks)} chunks:")
    for idx, c in enumerate(chunks, 1):
        print(f"\n[{idx}] Score: {c['score']:.4f} | Source: {c['source']} (Page {c['page']})")
        print(f"    Snippet: {c['text'][:160]}...")

    print("\n" + "="*60)
    print("  TEST 2: Cross-Lingual Semantic Cache Calibration")
    print("="*60)
    q1_en = "How many days can foreign students work in Germany per year?"
    sample_answer = "International students from third countries can work 140 full days or 280 half days per calendar year according to § 16b AufenthG."
    sample_sources = [{"source": "202406_Studierende_Fachkraefteeinwanderung.pdf", "page": 3}]

    # Populate cache
    put_cache(q1_en, sample_answer, sources=sample_sources, model_used="gpt-6-luna")
    print(f"Cached Q1 (English): '{q1_en}'")

    # Check German semantically equivalent query
    q1_de = "Wie viele Tage dürfen internationale Studierende in Deutschland pro Jahr arbeiten?"
    hit_de = check_cache(q1_de, threshold=0.86)
    if hit_de:
        print(f"\n[CACHE HIT SUCCESS] German query hit English cache entry!")
        print(f"  Similarity Score: {hit_de['similarity']:.4f}")
        print(f"  Cached Query:     {hit_de['cached_query']}")
        print(f"  Sources:          {hit_de['sources']}")
    else:
        print(f"\n[CACHE MISS] German query similarity below threshold.")

    # Near-miss calibration test: § 18b skilled worker visa should NOT falsely hit § 16b student visa cache
    q_near_miss = "Welche Voraussetzungen gelten für das Fachkräftevisum nach § 18b AufenthG?"
    hit_near_miss = check_cache(q_near_miss, threshold=0.88)
    if not hit_near_miss:
        print(f"\n[NEAR-MISS REJECTION SUCCESS] § 18b skilled worker query correctly avoided false positive cache hit.")
    else:
        print(f"\n[WARNING] Near-miss query hit cache with similarity {hit_near_miss['similarity']:.4f}")

    print("\n" + "="*60)
    print("  TEST 3: Bilingual Generation with Cited Provenance")
    print("="*60)
    test_q = "How many days can a non-EU student work in Germany?"
    print(f"Generating answer for: '{test_q}'")
    retrieved = retrieve_chunks(test_q, top_k=3)
    answer, model_used, stats, escalated = route_and_generate(test_q, retrieved)
    print(f"Model: {model_used} | Cost: ${stats['cost']:.5f} | Escalated: {escalated}")
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    run_tests()
