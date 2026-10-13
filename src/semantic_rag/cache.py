"""Semantic answer cache in Qdrant.

- Scope: entries are keyed by tenant_id. This is a cache *scope*, not access control:
  tenant_id is supplied by the caller and is not authenticated (finding A8).
- Validity: corpus version and expiry are filtered inside the Qdrant query, so an expired or
  stale top hit can never hide a valid entry behind it (finding A7). Expired points are
  purged at startup via purge_expired().
- Precision: calibration (scripts/calibrate_cache_threshold.py) showed that no cosine
  threshold separates paraphrases from near-misses: "Apple's net income in 2025" vs
  "Apple's net sales in 2025" scores 0.97, higher than most true paraphrases. So cosine
  only selects candidates, and a candidate must also pass:
    * key tokens (numbers, years, § references) equal: "§ 16b" never serves "§ 18b";
    * same language by default (CACHE_ALLOW_CROSS_LINGUAL=true to relax), so a German
      question is not answered with a cached English text;
    * same content words (word order/punctuation/casing variants), OR a cheap LLM
      equivalence check ("do both questions ask for exactly the same information?"),
      ~100 tokens instead of a full RAG answer (CACHE_VERIFY=false disables the LLM check,
      leaving only the exact-content tier);
    * in pipeline.py: the cached answer's document must be among the top documents
      retrieved for the new question.
"""
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.http import models

from semantic_rag.embeddings import get_embedder
from semantic_rag.index_meta import current_corpus_version
from semantic_rag.prompts import CACHE_EQUIVALENCE_PROMPT
from semantic_rag.qdrant_db import get_client, reset_collection
from semantic_rag.settings import get_settings
from semantic_rag.textutils import content_words, detect_language, key_tokens, normalize_query

_initialized = False


def _create_cache_collection() -> None:
    s = get_settings()
    client = get_client()
    client.create_collection(
        s.cache_collection,
        vectors_config=models.VectorParams(size=get_embedder().dim, distance=models.Distance.COSINE),
    )
    if s.qdrant_mode == "server":
        for key, schema in (("tenant_id", models.PayloadSchemaType.KEYWORD),
                            ("corpus_version", models.PayloadSchemaType.KEYWORD),
                            ("expires_at", models.PayloadSchemaType.FLOAT)):
            client.create_payload_index(s.cache_collection, key, schema)


def ensure_cache_collection() -> None:
    global _initialized
    if _initialized:
        return
    if not get_client().collection_exists(get_settings().cache_collection):
        _create_cache_collection()
    _initialized = True


def reset_initialized() -> None:
    global _initialized
    _initialized = False


def clear_cache() -> None:
    reset_collection(get_settings().cache_collection, _create_cache_collection)
    global _initialized
    _initialized = True


def purge_expired() -> None:
    ensure_cache_collection()
    get_client().delete(
        get_settings().cache_collection,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="expires_at", range=models.Range(lt=time.time()))])),
    )


def put_cache(query: str, q_vec: List[float], answer: str, sources: List[Dict[str, Any]], model_used: str,
              primary_document: str, tenant_id: str = "default") -> None:
    ensure_cache_collection()
    normalized = normalize_query(query)
    if not normalized or not answer.strip():
        return
    now = time.time()
    payload = {
        "query": query, "normalized_query": normalized, "answer": answer, "sources": sources,
        "model_used": model_used, "primary_document": primary_document,
        "key_tokens": sorted(key_tokens(query)), "language": detect_language(query),
        "corpus_version": current_corpus_version(), "tenant_id": tenant_id,
        "created_at": now, "expires_at": now + get_settings().cache_ttl_seconds,
    }
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}|{normalized}"))
    get_client().upsert(get_settings().cache_collection,
                        points=[models.PointStruct(id=point_id, vector=q_vec, payload=payload)])


def questions_equivalent(a: str, b: str) -> Dict[str, Any]:
    """Ask the small model whether two questions request the same information. Fails closed."""
    from semantic_rag.router import LLMError, _call, parse_model_json

    try:
        raw = _call(get_settings().small_model,
                    [{"role": "user", "content": CACHE_EQUIVALENCE_PROMPT.format(a=a, b=b)}])
        same = parse_model_json(raw["content"]).get("same") is True
        return {"same": same, "cost": raw["cost"]}
    except (LLMError, ValueError, json.JSONDecodeError):
        return {"same": False, "cost": 0.0}


def check_cache(query: str, q_vec: List[float], tenant_id: str = "default",
                threshold: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], float]:
    """Return (best valid cache entry that passes every guard or None, LLM cost spent verifying)."""
    ensure_cache_collection()
    s = get_settings()
    threshold = s.similarity_threshold if threshold is None else threshold
    flt = models.Filter(must=[
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
        models.FieldCondition(key="corpus_version", match=models.MatchValue(value=current_corpus_version())),
        models.FieldCondition(key="expires_at", range=models.Range(gt=time.time())),
    ])
    hits = get_client().query_points(s.cache_collection, query=q_vec, query_filter=flt, limit=5,
                                     score_threshold=threshold, with_payload=True).points
    q_keys = key_tokens(query)
    q_lang = detect_language(query)
    q_words = content_words(query)
    verify_cost = 0.0
    for hit in hits[:2]:  # at most two LLM checks per lookup
        p = hit.payload or {}
        if set(p.get("key_tokens", [])) != q_keys:
            continue
        if not s.cache_allow_cross_lingual and p.get("language") != q_lang:
            continue
        if content_words(p.get("query", "")) == q_words:
            method = "exact-content"
        elif s.cache_verify:
            check = questions_equivalent(p.get("query", ""), query)
            verify_cost += check["cost"]
            if not check["same"]:
                continue
            method = "llm-verified"
        else:
            continue
        return {"answer": p["answer"], "sources": p.get("sources", []), "cached_query": p.get("query", ""),
                "similarity": float(hit.score), "model": p.get("model_used", ""),
                "primary_document": p.get("primary_document", ""), "match": method}, verify_cost
    return None, verify_cost
