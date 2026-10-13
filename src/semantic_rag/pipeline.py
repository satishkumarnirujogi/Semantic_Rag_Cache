"""The one question-answering pipeline. FastAPI, Streamlit and the evaluator all call answer().

    question
      -> embed once (reused by cache lookup, retrieval and cache write)        [E2]
      -> semantic cache (tenant scope, version/TTL filters, key-token + document guards)
      -> hybrid retrieval + rerank + document discrimination (detects the domain)
      -> relevance gate: nothing relevant enough -> "not found" without calling an LLM
      -> cost-aware routing + grounded JSON generation (+ escalation)
      -> not grounded -> "not found";  grounded -> answer with citations, cached
      -> telemetry
"""
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from semantic_rag import telemetry
from semantic_rag.cache import check_cache, put_cache
from semantic_rag.catalog import domain_label
from semantic_rag.embeddings import get_embedder
from semantic_rag.prompts import NOT_FOUND
from semantic_rag.retrieval import retrieve, top_documents_fast
from semantic_rag.router import LLMError, route_and_generate
from semantic_rag.settings import get_settings
from semantic_rag.textutils import detect_language

STATUS_ANSWERED = "answered"
STATUS_NOT_FOUND = "not_found"
STATUS_CACHE_HIT = "cache_hit"
STATUS_ERROR = "error"


@dataclass
class Answer:
    query: str
    answer: str
    status: str
    language: str
    model_used: str = ""
    route: str = ""
    cache_hit: bool = False
    escalated: bool = False
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    domain: str = ""
    domain_label: str = ""
    primary_document: str = ""
    top_relevance: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _source_entry(c: Dict[str, Any], label: str) -> Dict[str, Any]:
    return {"label": label, "source": c.get("source"), "title": c.get("title"), "domain": c.get("domain"),
            "location": c.get("location"), "page": c.get("page_start"), "heading": c.get("heading"),
            "chunk_id": c.get("chunk_id"), "score": round(float(c.get("score", 0.0)), 4)}


def _chunk_view(c: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("chunk_id", "source", "title", "domain", "location", "heading", "page_start", "score", "text")
    return {k: c.get(k) for k in keys}


def answer(query: str, tenant_id: str = "default", use_cache: bool = True, domains: Optional[List[str]] = None,
           log: bool = True) -> Answer:
    s = get_settings()
    started = time.perf_counter()
    query = (query or "").strip()
    language = detect_language(query)
    result = Answer(query=query, answer="", status=STATUS_ERROR, language=language)
    timings: Dict[str, float] = {}

    def finish() -> Answer:
        if result.status == STATUS_NOT_FOUND:
            # nothing was answered, so don't report the closest-but-wrong document as "the" document
            result.primary_document, result.domain, result.domain_label = "", "", ""
        result.timings_ms = {k: round(v, 1) for k, v in timings.items()}
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if log:
            telemetry.log_request({
                "query": query, "model": result.model_used or "none", "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens, "cost": result.cost, "latency_ms": result.latency_ms,
                "cache_hit": result.cache_hit, "route": result.route or "none", "escalated": result.escalated,
                "status": result.status, "tenant_id": tenant_id, "domain": result.domain,
                "primary_document": result.primary_document, "top_relevance": result.top_relevance,
                "embed_ms": timings.get("embed"), "cache_ms": timings.get("cache"),
                "search_ms": timings.get("search"), "rerank_ms": timings.get("rerank"),
                "generation_ms": timings.get("generation"), "error": result.error,
            })
        return result

    if not query:
        result.error = "empty question"
        return finish()

    t = time.perf_counter()
    q_vec = get_embedder().encode_query(query)
    timings["embed"] = (time.perf_counter() - t) * 1000

    # 1. Semantic cache
    if use_cache:
        t = time.perf_counter()
        hit, verify_cost = check_cache(query, q_vec, tenant_id=tenant_id)
        result.cost += verify_cost
        if hit and hit["primary_document"] not in top_documents_fast(query, q_vec):
            hit = None  # similar wording, but the question points at a different document
        timings["cache"] = (time.perf_counter() - t) * 1000
        if hit:
            result.answer, result.sources = hit["answer"], hit["sources"]
            result.status, result.cache_hit, result.route = STATUS_CACHE_HIT, True, "cache"
            result.model_used = "semantic-cache"
            result.primary_document = hit["primary_document"]
            if result.sources:
                result.domain = result.sources[0].get("domain", "")
                result.domain_label = domain_label(result.domain)
            return finish()

    # 2. Retrieval, reranking, document discrimination
    retrieval = retrieve(query, q_vec, domains=domains)
    timings["search"], timings["rerank"] = retrieval.search_ms, retrieval.rerank_ms
    result.domain, result.domain_label = retrieval.domain, retrieval.domain_label
    result.primary_document = retrieval.primary_document
    result.top_relevance = round(retrieval.top_relevance, 4)
    result.retrieved_chunks = [_chunk_view(c) for c in retrieval.context]

    # 3. Relevance gate: nothing relevant -> don't ask an LLM to improvise
    if not retrieval.context or retrieval.top_relevance < s.min_relevance:
        result.status, result.answer, result.route = STATUS_NOT_FOUND, NOT_FOUND[language], "gate"
        return finish()

    # 4. Routed, grounded generation
    t = time.perf_counter()
    try:
        gen = route_and_generate(query, retrieval.context, language, retrieval.top_relevance)
    except LLMError as e:
        timings["generation"] = (time.perf_counter() - t) * 1000
        result.error = str(e)
        result.answer = "The answer service is temporarily unavailable. Please try again."
        return finish()
    timings["generation"] = (time.perf_counter() - t) * 1000
    result.model_used, result.escalated = gen.model, gen.escalated
    result.route = "escalated" if gen.escalated else ("large" if gen.model == s.large_model.id else "small")
    result.cost += gen.cost
    result.prompt_tokens, result.completion_tokens = gen.prompt_tokens, gen.completion_tokens

    if not gen.answerable:
        result.status, result.answer = STATUS_NOT_FOUND, NOT_FOUND[language]
        return finish()

    cited = {int(c[1:]) for c in gen.citations}
    result.sources = [_source_entry(c, f"S{i}") for i, c in enumerate(retrieval.context, start=1) if i in cited]
    result.answer, result.status = gen.answer, STATUS_ANSWERED
    result.primary_document = result.sources[0]["source"] if result.sources else result.primary_document
    result.domain = result.sources[0]["domain"] if result.sources else result.domain

    # 5. Cache only grounded answers (never cache "not found" or errors)
    if use_cache:
        put_cache(query, q_vec, result.answer, result.sources, result.model_used, result.primary_document, tenant_id)
    return finish()
