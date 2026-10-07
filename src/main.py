import os
import sys
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure src directory and .venv site-packages are in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

VENV_SITE_PACKAGES = os.path.abspath(os.path.join(SRC_DIR, "..", ".venv", "Lib", "site-packages"))
if os.path.exists(VENV_SITE_PACKAGES) and VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

from baseline_rag import retrieve_chunks, ingest_corpus, init_collection
from cache import check_cache, put_cache, init_cache_collection
from router import route_and_generate, classify_query_route
from logger import log_request, init_db

app = FastAPI(
    title="German Law Multilingual RAG & Semantic Cache API",
    version="2.0",
    description="Multilingual German Law RAG platform with Qdrant semantic caching, cost-aware routing, and source provenance."
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.88
    use_cache: Optional[bool] = True
    tenant_id: Optional[str] = "default_tenant"

class QueryResponse(BaseModel):
    query: str
    answer: str
    model_used: str
    cache_hit: bool
    escalated: bool
    cost: float
    latency_ms: float
    retrieved_chunks: List[Dict[str, Any]]
    sources: Optional[List[Dict[str, Any]]] = []

@app.on_event("startup")
def startup_event():
    """Initialize DB and Qdrant collections on FastAPI start."""
    init_db()
    init_collection()
    init_cache_collection()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "German Law Multilingual RAG & Semantic Cache API",
        "version": "2.0"
    }

@app.post("/ingest")
def trigger_ingest(reset: bool = False):
    total = ingest_corpus(reset=reset)
    return {"status": "success", "total_points": total}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    start_time = time.time()
    query_str = request.query.strip()
    tenant_id = request.tenant_id or "default_tenant"

    if not query_str:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # 1. Check Semantic Cache
    if request.use_cache:
        cached_result = check_cache(query_str, threshold=request.similarity_threshold, tenant_id=tenant_id)
        if cached_result:
            latency_ms = (time.time() - start_time) * 1000.0
            log_request(
                query=query_str,
                model="semantic-cache",
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                latency_ms=latency_ms,
                cache_hit=True,
                route="cache",
                escalated=False
            )
            return QueryResponse(
                query=query_str,
                answer=cached_result["answer"],
                model_used="semantic-cache",
                cache_hit=True,
                escalated=False,
                cost=0.0,
                latency_ms=round(latency_ms, 2),
                retrieved_chunks=[],
                sources=cached_result.get("sources", [])
            )

    # 2. Retrieve Relevant Context Chunks
    chunks = retrieve_chunks(query_str, top_k=request.top_k)

    sources = []
    seen = set()
    for c in chunks:
        key = (c.get("source", "unknown"), c.get("page", 1))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})

    # 3. Route & Generate Answer (with Escalation Safety Net)
    answer, model_used, stats, escalated = route_and_generate(query_str, chunks)

    # 4. Update Semantic Cache with fresh response, sources, and tenant scope
    if request.use_cache:
        put_cache(query_str, answer, sources=sources, model_used=model_used, tenant_id=tenant_id)

    latency_ms = (time.time() - start_time) * 1000.0
    route_name = "escalated" if escalated else ("large" if "gpt" in model_used or "luna" in model_used else "small")

    # 5. Log telemetry metrics to SQLite DB
    log_request(
        query=query_str,
        model=model_used,
        prompt_tokens=stats.get("prompt_tokens", 0),
        completion_tokens=stats.get("completion_tokens", 0),
        cost=stats.get("cost", 0.0),
        latency_ms=latency_ms,
        cache_hit=False,
        route=route_name,
        escalated=escalated
    )

    return QueryResponse(
        query=query_str,
        answer=answer,
        model_used=model_used,
        cache_hit=False,
        escalated=escalated,
        cost=stats.get("cost", 0.0),
        latency_ms=round(latency_ms, 2),
        retrieved_chunks=chunks,
        sources=sources
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
