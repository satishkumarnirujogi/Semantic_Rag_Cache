"""Hybrid retrieval, reranking and document discrimination.

1. Hybrid search: dense (multilingual embeddings) + sparse (BM25-style, Qdrant IDF) fused
   with reciprocal-rank fusion. Sparse search catches exact terms ("§ 16b", "RFC 9111",
   "Cache-Control") that dense search blurs.
2. Rerank: a multilingual cross-encoder scores (question, passage) pairs; its probability
   is the relevance signal used for every later decision.
3. Document discrimination: candidates are grouped by document; only documents scoring
   within DOC_MARGIN of the best one are kept (max MAX_CONTEXT_DOCS). This keeps chunks
   from the wrong document (e.g. Microsoft's 10-K for an Apple question) out of the prompt.
4. The domain of the best document is reported as the question's detected domain.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qdrant_client.http import models

from semantic_rag.catalog import domain_label
from semantic_rag.embeddings import get_reranker
from semantic_rag.qdrant_db import get_client
from semantic_rag.settings import get_settings
from semantic_rag.textutils import sparse_query_vector


@dataclass
class RetrievalResult:
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # reranked, best first
    context: List[Dict[str, Any]] = field(default_factory=list)  # chunks sent to the LLM
    documents: List[Dict[str, Any]] = field(default_factory=list)  # ranked documents
    selected_documents: List[str] = field(default_factory=list)
    domain: str = ""
    domain_label: str = ""
    top_relevance: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def primary_document(self) -> str:
        return self.selected_documents[0] if self.selected_documents else ""


def _domain_filter(domains: Optional[List[str]]) -> Optional[models.Filter]:
    if not domains:
        return None
    return models.Filter(must=[models.FieldCondition(key="domain", match=models.MatchAny(any=domains))])


def hybrid_search(query: str, q_vec: List[float], limit: int, domains: Optional[List[str]] = None) -> List[Dict]:
    s = get_settings()
    idx, vals = sparse_query_vector(query)
    prefetch = [models.Prefetch(query=q_vec, using="dense", limit=limit * 2)]
    if idx:
        prefetch.append(models.Prefetch(query=models.SparseVector(indices=idx, values=vals), using="sparse",
                                        limit=limit * 2))
    res = get_client().query_points(s.chunks_collection, prefetch=prefetch,
                                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                                    query_filter=_domain_filter(domains), limit=limit, with_payload=True)
    out = []
    for rank, p in enumerate(res.points):
        d = dict(p.payload or {})
        d.update(id=str(p.id), fusion_score=float(p.score), fusion_rank=rank)
        out.append(d)
    return out


def top_documents_fast(query: str, q_vec: List[float], n: int = 3) -> List[str]:
    """Cheap (no rerank) list of the most likely documents; used to sanity-check cache hits."""
    seen: List[str] = []
    for c in hybrid_search(query, q_vec, limit=15):
        if c["source"] not in seen:
            seen.append(c["source"])
        if len(seen) >= n:
            break
    return seen


def retrieve(query: str, q_vec: List[float], domains: Optional[List[str]] = None,
             candidate_k: Optional[int] = None, context_k: Optional[int] = None) -> RetrievalResult:
    s = get_settings()
    candidate_k = candidate_k or s.candidate_k
    context_k = context_k or s.context_k
    result = RetrievalResult()

    t = time.perf_counter()
    candidates = hybrid_search(query, q_vec, candidate_k, domains)
    result.search_ms = (time.perf_counter() - t) * 1000
    if not candidates:
        return result

    t = time.perf_counter()
    reranker = get_reranker()
    if reranker is not None:
        scores = reranker.score(query, [f"{c['header']}\n{c['text']}" for c in candidates])
    else:  # no reranker: fall back to a rank-based score so the pipeline still works
        scores = [1.0 / (1 + c["fusion_rank"]) for c in candidates]
    for c, sc in zip(candidates, scores):
        c["score"] = float(sc)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    result.rerank_ms = (time.perf_counter() - t) * 1000
    result.candidates = candidates
    result.top_relevance = candidates[0]["score"]

    # Rank documents by their best chunk, with a small bonus for supporting chunks
    docs: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        d = docs.setdefault(c["source"], {"source": c["source"], "title": c.get("title", c["source"]),
                                          "domain": c.get("domain", ""), "best": c["score"], "support": 0.0})
        d["support"] += c["score"]
    for d in docs.values():
        d["score"] = d["best"] + 0.05 * min(d["support"] - d["best"], 1.0)
    ranked = sorted(docs.values(), key=lambda d: d["score"], reverse=True)
    result.documents = ranked

    best = ranked[0]["score"]
    selected = [d["source"] for d in ranked if d["score"] >= best - s.doc_margin and d["best"] >= s.min_relevance]
    result.selected_documents = selected[: s.max_context_docs] or [ranked[0]["source"]]
    result.domain = ranked[0]["domain"]
    result.domain_label = domain_label(result.domain)

    chosen = set(result.selected_documents)
    result.context = [c for c in candidates if c["source"] in chosen and c["score"] >= s.min_relevance][:context_k]
    if not result.context:
        result.context = [candidates[0]]
    return result
