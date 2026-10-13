"""Calibrate SIMILARITY_THRESHOLD for the semantic cache on labeled question pairs.

Positives are paraphrases that SHOULD share a cached answer; negatives are near-misses that
must NOT (different company, year, paragraph, format, directive...). For each threshold we
report precision/recall of "cosine >= threshold AND key tokens equal" (the cache's own rule)
and recommend the lowest threshold with zero false positives.

With --verify-llm it also runs the cache's full rule (cosine candidate + key tokens + language
+ exact-content or LLM equivalence check) on every pair and reports its precision/recall.
That costs a few hundred tokens per pair on the small model.

Usage:  python scripts/calibrate_cache_threshold.py [--verify-llm]
Writes: logs/cache_threshold_calibration.csv
"""
import argparse
import csv

from semantic_rag.cache import questions_equivalent
from semantic_rag.embeddings import get_embedder
from semantic_rag.settings import LOGS_DIR, get_settings
from semantic_rag.textutils import content_words, detect_language, key_tokens

POSITIVE = [
    ("What were Apple's total net sales in fiscal 2025?", "How much did Apple sell in total in fiscal year 2025?"),
    ("What was Microsoft's revenue in fiscal year 2025?", "How much revenue did Microsoft make in FY 2025?"),
    ("How many employees did Apple have in 2025?", "What was Apple's headcount in 2025?"),
    ("Which RFC does RFC 9111 obsolete?", "RFC 9111 replaces which older RFC?"),
    ("What does the no-store directive mean?", "What is the meaning of the no-store cache directive?"),
    ("Where are Delta Lake checkpoints stored?", "In which directory does Delta Lake keep its checkpoint files?"),
    ("What is a row group in Parquet?", "Can you explain what a Parquet row group is?"),
    ("What does a Kubernetes cluster consist of?", "What are the parts of a Kubernetes cluster?"),
    ("How do I declare task dependencies in Airflow?", "What is the way to set dependencies between Airflow tasks?"),
    ("What are the two main data structures in pandas?", "Which two primary data structures does pandas provide?"),
    ("What file extension do Python stub files use?", "Which extension is used for stub files in PEP 484?"),
    ("How does NIST define confabulation?", "What is confabulation according to NIST?"),
    ("How many days per year can international students work in Germany?",
     "How many days a year are foreign students allowed to work in Germany?"),
    ("Wie viele Tage dürfen internationale Studierende im Jahr arbeiten?",
     "Wie viele Arbeitstage pro Jahr sind für ausländische Studierende erlaubt?"),
    ("Wie hoch ist die Gebühr für die Aufenthaltserlaubnis?", "Was kostet die Aufenthaltserlaubnis?"),
]
CROSS_LINGUAL = [
    ("How many days per year can international students work in Germany?",
     "Wie viele Tage im Jahr dürfen internationale Studierende in Deutschland arbeiten?"),
    ("How much does the residence permit cost?", "Wie viel kostet die Aufenthaltserlaubnis?"),
]
NEGATIVE = [
    ("What were Apple's total net sales in fiscal 2025?", "What were Microsoft's total net sales in fiscal 2025?"),
    ("What was Microsoft's revenue in fiscal year 2025?", "What was Microsoft's revenue in fiscal year 2024?"),
    ("How many employees did Apple have in 2025?", "How many employees did Microsoft have in 2025?"),
    ("What was Apple's net income in 2025?", "What was Apple's net sales in 2025?"),
    ("Which RFC does RFC 9111 obsolete?", "Which RFC does RFC 9110 obsolete?"),
    ("What does the no-store directive mean?", "What does the no-cache directive mean?"),
    ("Where are Delta Lake checkpoints stored?", "Where are Iceberg manifest lists stored?"),
    ("What is a row group in Parquet?", "What is a record batch in Arrow?"),
    ("What does the kube-proxy component do?", "What does the kube-scheduler component do?"),
    ("Was regelt § 16b AufenthG?", "Was regelt § 18b AufenthG?"),
    ("How many days per year can international students work in Germany?",
     "How many hours per week can international students work in Germany?"),
    ("What is the fee for issuing a residence permit for the first time?",
     "What is the fee for extending a residence permit?"),
    ("What does PEP 484 say about the Any type?", "What does PEP 484 say about the Optional type?"),
    ("What is a job in GitHub Actions?", "What is a step in GitHub Actions?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-llm", action="store_true")
    args = parser.parse_args()
    emb = get_embedder()
    rows = []
    for label, pairs in (("positive", POSITIVE), ("cross_lingual", CROSS_LINGUAL), ("negative", NEGATIVE)):
        for a, b in pairs:
            va, vb = emb.encode_queries([a, b])
            rows.append({"label": label, "a": a, "b": b, "cosine": round(sum(x * y for x, y in zip(va, vb)), 4),
                         "keys_equal": key_tokens(a) == key_tokens(b),
                         "same_language": detect_language(a) == detect_language(b)})

    def hit(r, t, cross_ok):
        return r["cosine"] >= t and r["keys_equal"] and (cross_ok or r["same_language"])

    print(f"{'label':14s} {'cos':>6s} keys  pair")
    for r in sorted(rows, key=lambda r: -r["cosine"]):
        print(f"{r['label']:14s} {r['cosine']:6.3f} {'=' if r['keys_equal'] else '!'}     {r['a'][:50]} | {r['b'][:50]}")

    print("\nthreshold  recall(pos)  false_pos(neg)")
    best = None
    for t in [x / 100 for x in range(80, 100)]:
        pos = [r for r in rows if r["label"] == "positive"]
        neg = [r for r in rows if r["label"] == "negative"]
        recall = sum(hit(r, t, False) for r in pos) / len(pos)
        fp = sum(hit(r, t, True) for r in neg)
        print(f"  {t:.2f}      {recall:5.2f}        {fp}")
        if fp == 0 and best is None:
            best = (t, recall)
    LOGS_DIR.mkdir(exist_ok=True)
    with open(LOGS_DIR / "cache_threshold_calibration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nEmbedder: {get_settings().embed_model}; current SIMILARITY_THRESHOLD={get_settings().similarity_threshold}")
    if best:
        print(f"Cosine alone: lowest threshold with zero false positives = {best[0]:.2f} (paraphrase recall {best[1]:.2f})")

    if args.verify_llm:
        t = get_settings().similarity_threshold
        total_cost = 0.0
        print(f"\nFull cache rule at SIMILARITY_THRESHOLD={t}:")
        for r in rows:
            r["full_rule_hit"] = False
            if r["cosine"] >= t and r["keys_equal"] and r["same_language"]:
                if content_words(r["a"]) == content_words(r["b"]):
                    r["full_rule_hit"] = True
                else:
                    check = questions_equivalent(r["a"], r["b"])
                    total_cost += check["cost"]
                    r["full_rule_hit"] = check["same"]
            if r["label"] == "negative" and r["full_rule_hit"]:
                print(f"  FALSE POSITIVE: {r['a']} | {r['b']}")
        pos = [r for r in rows if r["label"] == "positive"]
        neg = [r for r in rows if r["label"] == "negative"]
        print(f"  paraphrase recall {sum(r['full_rule_hit'] for r in pos)}/{len(pos)}, "
              f"false positives {sum(r['full_rule_hit'] for r in neg)}/{len(neg)}, verification cost ${total_cost:.6f}")
        with open(LOGS_DIR / "cache_threshold_calibration.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
