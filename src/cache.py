import os
import sys
import time
import string
import uuid
from typing import Optional, Dict, Any, List
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qdrant_db import get_shared_qdrant_client

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CACHE_COLLECTION = "german_semantic_cache"
VECTOR_DIM = 768  # 768 for paraphrase-multilingual-mpnet-base-v2

DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.88))
CURRENT_CORPUS_VERSION = "v1.0"
CACHE_TTL_SECONDS = 86400 * 7  # 7-day TTL

client = get_shared_qdrant_client()
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def init_cache_collection(reset: bool = False):
    """Ensure multilingual 768-dim semantic cache collection exists in Qdrant."""
    collections = [c.name for c in client.get_collections().collections]
    if reset and CACHE_COLLECTION in collections:
        client.delete_collection(CACHE_COLLECTION)
        collections.remove(CACHE_COLLECTION)

    if CACHE_COLLECTION not in collections:
        client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE)
        )
        print(f"Created Qdrant cache collection: '{CACHE_COLLECTION}' (dim={VECTOR_DIM})")
    else:
        print(f"Cache collection '{CACHE_COLLECTION}' already exists.")

def normalize_query(query: str) -> str:
    """Normalize text: lowercase, strip whitespace and basic punctuation."""
    text = query.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def put_cache(
    query: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    model_used: str = "cache",
    tenant_id: str = "default_tenant"
):
    """Store a query-answer pair in the multilingual semantic cache with metadata and TTL."""
    init_cache_collection()
    normalized = normalize_query(query)
    if not normalized or not answer.strip():
        return

    q_vec = embedder.encode(normalized).tolist()
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{normalized}_{tenant_id}"))

    payload = {
        "query": query,
        "normalized_query": normalized,
        "answer": answer,
        "sources": sources or [],
        "model_used": model_used,
        "corpus_version": CURRENT_CORPUS_VERSION,
        "tenant_id": tenant_id,
        "expires_at": time.time() + CACHE_TTL_SECONDS
    }

    client.upsert(
        collection_name=CACHE_COLLECTION,
        points=[models.PointStruct(id=point_id, vector=q_vec, payload=payload)]
    )

def check_cache(
    query: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    tenant_id: str = "default_tenant"
) -> Optional[Dict[str, Any]]:
    """Check Qdrant semantic cache for a similar previous query with multilingual cosine search."""
    init_cache_collection()
    normalized = normalize_query(query)
    if not normalized:
        return None

    q_vec = embedder.encode(normalized).tolist()
    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id)
            )
        ]
    )

    try:
        res = client.query_points(
            collection_name=CACHE_COLLECTION,
            query=q_vec,
            query_filter=search_filter,
            limit=1
        )
        hits = res.points
    except Exception as e:
        print(f"Cache query error: {e}")
        return None

    if not hits or hits[0].score < threshold:
        return None

    best_hit = hits[0]
    payload = best_hit.payload or {}

    # Check corpus version
    if payload.get("corpus_version") != CURRENT_CORPUS_VERSION:
        return None

    # Check TTL expiration
    if time.time() > payload.get("expires_at", float("inf")):
        return None

    return {
        "answer": payload.get("answer", ""),
        "sources": payload.get("sources", []),
        "cached_query": payload.get("query", ""),
        "similarity": best_hit.score,
        "model": payload.get("model_used", "semantic-cache"),
        "corpus_version": payload.get("corpus_version"),
        "tenant_id": payload.get("tenant_id"),
        "source": "cache"
    }

if __name__ == "__main__":
    init_cache_collection()
    print("Multilingual Semantic Cache initialized successfully.")
