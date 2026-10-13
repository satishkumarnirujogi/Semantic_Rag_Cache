"""Evaluation on data/eval/gold_set.json.

Retrieval metrics (free, local):
  doc_hit@1        the top-ranked document is a gold document
  recall@context   a chunk sent to the LLM is from a gold document AND contains gold evidence
  recall@cand      same, over all hybrid-search candidates (separates search misses from rerank misses)
  MRR              reciprocal rank of the first relevant chunk after reranking
  domain_acc       detected domain == gold domain
Answer metrics (calls the LLMs; --retrieval-only skips them):
  answered / false refusals on answerable questions, correct refusals on unanswerable ones,
  judge correctness (1-5, judged by a different model family), citation accuracy, cost, latency.
Judge failures are recorded with judge_status="error" and excluded from the score average,
never silently replaced by a default score (finding A10/H14).

Run:  python -m semantic_rag.evaluate [--retrieval-only] [--ids fin-01,na-02]
"""
import argparse
import json
import platform
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from semantic_rag import pipeline
from semantic_rag.embeddings import get_embedder
from semantic_rag.index_meta import read_manifest
from semantic_rag.prompts import JUDGE_PROMPT
from semantic_rag.retrieval import retrieve
from semantic_rag.router import LLMError, _call, parse_model_json
from semantic_rag.settings import DATA_DIR, LOGS_DIR, get_settings

GOLD_PATH = DATA_DIR / "eval" / "gold_set.json"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def is_relevant(chunk: Dict[str, Any], item: Dict[str, Any]) -> bool:
    if chunk.get("source") not in item.get("source_documents", []):
        return False
    body = _norm(chunk.get("text", ""))
    return any(_norm(e) in body for e in item.get("evidence", []))


def first_relevant_rank(chunks: List[Dict[str, Any]], item: Dict[str, Any]) -> Optional[int]:
    for i, c in enumerate(chunks, start=1):
        if is_relevant(c, item):
            return i
    return None


def judge(question: str, reference: str, answer: str) -> Dict[str, Any]:
    s = get_settings()
    try:
        raw = _call(s.judge_model, [{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer)}])
        data = parse_model_json(raw["content"])
        score = int(data["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score out of range: {score}")
        return {"judge_status": "ok", "judge_score": score, "judge_reason": data.get("reason", ""),
                "judge_cost": raw["cost"]}
    except (LLMError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return {"judge_status": "error", "judge_score": None, "judge_reason": f"{type(e).__name__}: {e}",
                "judge_cost": 0.0}


def evaluate_item(item: Dict[str, Any], run_llm: bool) -> Dict[str, Any]:
    q = item["question"]
    row: Dict[str, Any] = {"id": item["id"], "question": q, "domain": item["domain"], "language": item["language"],
                           "answerable": item["answerable"]}
    q_vec = get_embedder().encode_query(q)
    r = retrieve(q, q_vec)
    row["detected_domain"] = r.domain
    row["top_document"] = r.primary_document
    row["selected_documents"] = "; ".join(r.selected_documents)
    row["top_relevance"] = round(r.top_relevance, 4)
    if item["answerable"]:
        rank = first_relevant_rank(r.candidates, item)
        row["doc_hit@1"] = int(r.primary_document in item["source_documents"])
        row["recall@context"] = int(first_relevant_rank(r.context, item) is not None)
        row["recall@cand"] = int(rank is not None)
        row["mrr"] = round(1.0 / rank, 4) if rank else 0.0
        row["domain_correct"] = int(r.domain == item["domain"])

    if run_llm:
        a = pipeline.answer(q, use_cache=False, log=False)
        row.update(status=a.status, answer=a.answer, model=a.model_used, route=a.route, escalated=a.escalated,
                   cost=a.cost, latency_ms=a.latency_ms, error=a.error,
                   cited_documents="; ".join(sorted({s["source"] for s in a.sources})))
        if item["answerable"]:
            cited_ok = any(s["source"] in item["source_documents"] for s in a.sources)
            row["citation_correct"] = int(a.status == pipeline.STATUS_ANSWERED and cited_ok)
            if a.status == pipeline.STATUS_ANSWERED:
                row.update(judge(q, item["reference_answer"], a.answer))
            else:
                row.update(judge_status="skipped", judge_score=1, judge_reason="refused or failed", judge_cost=0.0)
        else:
            row["correct_refusal"] = int(a.status == pipeline.STATUS_NOT_FOUND)
    return row


def summarize(df: pd.DataFrame, run_llm: bool) -> Dict[str, Any]:
    s = get_settings()
    ans = df[df["answerable"]]
    una = df[~df["answerable"]]
    out: Dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d"), "n_questions": int(len(df)), "n_answerable": int(len(ans)),
        "n_unanswerable": int(len(una)), "hardware": f"{platform.system()} {platform.machine()} "
                                                     f"({platform.processor() or 'cpu'}), CPU inference",
        "corpus_version": (read_manifest() or {}).get("corpus_version"),
        "embed_model": s.embed_model, "rerank_model": s.rerank_model if s.rerank_enabled else None,
        "retrieval": {
            "doc_hit@1": round(ans["doc_hit@1"].mean(), 3),
            f"recall@context(k<={s.context_k})": round(ans["recall@context"].mean(), 3),
            f"recall@candidates(k={s.candidate_k})": round(ans["recall@cand"].mean(), 3),
            "mrr": round(ans["mrr"].mean(), 3),
            "domain_accuracy": round(ans["domain_correct"].mean(), 3),
            "top_relevance_answerable_median": round(ans["top_relevance"].median(), 3),
            "top_relevance_unanswerable_median": round(una["top_relevance"].median(), 3) if len(una) else None,
        },
        "by_domain": {d: {"n": int(len(g)), "recall@context": round(g["recall@context"].mean(), 3),
                          "doc_hit@1": round(g["doc_hit@1"].mean(), 3)} for d, g in ans.groupby("domain")},
    }
    if run_llm:
        judged = ans[ans["judge_status"].isin(["ok", "skipped"])]
        out["answers"] = {
            "models": {"small": s.small_model.id, "large": s.large_model.id, "judge": s.judge_model.id,
                       "prices_as_of": s.prices_as_of},
            "answered_rate_on_answerable": round((ans["status"] == "answered").mean(), 3),
            "false_refusal_rate": round((ans["status"] == "not_found").mean(), 3),
            "correct_refusal_rate_on_unanswerable": round(una["correct_refusal"].mean(), 3) if len(una) else None,
            "hallucinated_answers_on_unanswerable": int((una["status"] == "answered").sum()) if len(una) else 0,
            "mean_judge_score": round(judged["judge_score"].mean(), 2) if len(judged) else None,
            "judge_errors": int((ans["judge_status"] == "error").sum()),
            "citation_accuracy": round(ans["citation_correct"].mean(), 3),
            "errors": int((df["status"] == "error").sum()),
            "escalations": int(df["escalated"].sum()),
            "routes": {k: int(v) for k, v in df["route"].value_counts().items()},
            "total_generation_cost_usd": round(df["cost"].sum(), 6),
            "total_judge_cost_usd": round(ans["judge_cost"].fillna(0).sum(), 6),
            "latency_ms_median": round(df["latency_ms"].median(), 1),
            "latency_ms_p95": round(df["latency_ms"].quantile(0.95), 1),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers on the gold set.")
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM calls (free)")
    parser.add_argument("--ids", default="", help="comma-separated item ids to run")
    parser.add_argument("--gold", default=str(GOLD_PATH))
    args = parser.parse_args()

    items = json.loads(Path(args.gold).read_text(encoding="utf-8"))["items"]
    if args.ids:
        wanted = set(args.ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    run_llm = not args.retrieval_only
    rows = []
    for n, item in enumerate(items, start=1):
        row = evaluate_item(item, run_llm)
        rows.append(row)
        flag = row.get("status", "") or ""
        print(f"[{n}/{len(items)}] {item['id']:7s} rel={row['top_relevance']:.2f} "
              f"top={row['top_document'][:40]:40s} {flag} {row.get('judge_score', '')}", flush=True)
    df = pd.DataFrame(rows)
    summary = summarize(df, run_llm)

    suffix = "retrieval" if args.retrieval_only else "full"
    LOGS_DIR.mkdir(exist_ok=True)
    df.to_csv(LOGS_DIR / f"eval_results_{suffix}.csv", index=False)
    (LOGS_DIR / f"eval_summary_{suffix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(df) and "top_relevance" in df:
        a = df[df["answerable"]]["top_relevance"].tolist()
        u = df[~df["answerable"]]["top_relevance"].tolist()
        if a and u:
            print(f"\nGate calibration: answerable top_relevance min={min(a):.3f} p10={statistics.quantiles(a, n=10)[0]:.3f}; "
                  f"unanswerable max={max(u):.3f} median={statistics.median(u):.3f} "
                  f"(current MIN_RELEVANCE={get_settings().min_relevance})")


if __name__ == "__main__":
    main()
