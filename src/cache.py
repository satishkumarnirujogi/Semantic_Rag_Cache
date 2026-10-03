import os
import sys
import time
import string
import uuid
from typing import Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE_COLLECTION_NAME = "semantic_cache"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.85))

CURRENT_CORPUS_VERSION = "v1.0"  # Increment whenever documents are re-ingested
CACHE_TTL_SECONDS = 86400        # 24-hour default TTL

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def init_cache_collection():
    """Ensure semantic_cache Qdrant collection exists."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if CACHE_COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=CACHE_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created Qdrant cache collection: '{CACHE_COLLECTION_NAME}'")

def normalize_query(query: str) -> str:
    """Normalize text: lowercase, strip whitespace and punctuation."""
    text = query.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def check_cache(
    query: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    tenant_id: str = "default_tenant"
) -> Optional[Dict[str, Any]]:
    """Check Qdrant semantic cache for a similar previous query with tenant filter, TTL, and versioning."""
    init_cache_collection()
    normalized = normalize_query(query)
    if not normalized:
        return None

    query_vector = embedding_model.encode(normalized).tolist()

    # Pre-filter search by tenant_id to prevent cross-tenant data leaks
    search_filter = Filter(
        must=[
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            )
        ]
    )

    try:
        results = qdrant_client.query_points(
            collection_name=CACHE_COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=1
        ).points
    except Exception as e:
        print(f"Cache query error: {e}")
        return None

    if not results:
        return None

    best_hit = results[0]
    similarity = best_hit.score
    payload = best_hit.payload

    # 1. Check similarity threshold
    if similarity < threshold:
        return None

    # 2. Check Corpus Version Invalidation
    if payload.get("corpus_version") != CURRENT_CORPUS_VERSION:
        print(f"Cache invalidated due to corpus version mismatch ({payload.get('corpus_version')} != {CURRENT_CORPUS_VERSION})")
        return None

    # 3. Check TTL Expiration
    if time.time() > payload.get("expires_at", float("inf")):
        print("Cache item expired.")
        return None

    return {
        "answer": payload.get("answer", ""),
        "cached_query": payload.get("original_query", ""),
        "similarity": similarity,
        "model": payload.get("model_used", "semantic-cache"),
        "corpus_version": payload.get("corpus_version"),
        "tenant_id": payload.get("tenant_id"),
        "source": "cache"
    }

def put_cache(
    query: str,
    answer: str,
    model_used: str = "cache",
    tenant_id: str = "default_tenant"
):
    """Store a query answer pair in the Qdrant semantic cache with tenant_id, TTL, and corpus_version."""
    init_cache_collection()
    normalized = normalize_query(query)
    if not normalized or not answer.strip():
        return

    query_vector = embedding_model.encode(normalized).tolist()
    # Stable ID generation per query + tenant combination
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{normalized}_{tenant_id}"))

    payload = {
        "normalized_query": normalized,
        "original_query": query,
        "answer": answer,
        "model_used": model_used,
        "corpus_version": CURRENT_CORPUS_VERSION,
        "tenant_id": tenant_id,
        "expires_at": time.time() + CACHE_TTL_SECONDS
    }

    qdrant_client.upsert(
        collection_name=CACHE_COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=query_vector,
                payload=payload
            )
        ]
    )

if __name__ == "__main__":
    init_cache_collection()
    print("Semantic cache module initialized successfully with tenant isolation, TTL, and versioning.")
