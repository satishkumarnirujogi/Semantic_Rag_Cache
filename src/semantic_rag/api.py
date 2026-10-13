"""FastAPI service.  Run:  uvicorn semantic_rag.api:app --port 8000"""
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from semantic_rag import catalog, pipeline, telemetry
from semantic_rag.cache import ensure_cache_collection, purge_expired
from semantic_rag.embeddings import get_embedder, get_reranker
from semantic_rag.index_meta import read_manifest
from semantic_rag.ingest import ensure_chunks_collection, ingest


@asynccontextmanager
async def lifespan(_: FastAPI):
    telemetry.init_db()
    get_embedder()
    get_reranker()
    ensure_chunks_collection()
    ensure_cache_collection()
    purge_expired()
    yield


app = FastAPI(title="Multi-Domain RAG & Semantic Cache API", version="3.0", lifespan=lifespan,
              description="Grounded multi-domain RAG: hybrid retrieval, reranking, document discrimination, "
                          "semantic cache and cost-aware routing. Answers only from indexed documents.")


class QueryRequest(BaseModel):
    query: str
    tenant_id: str = Field("default", description="Cache scope key. Not authenticated; not access control.")
    use_cache: bool = True
    domains: Optional[List[str]] = Field(None, description="Restrict retrieval to these domain keys")


class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str
    language: str
    model_used: str
    route: str
    cache_hit: bool
    escalated: bool
    cost: float
    latency_ms: float
    domain: str
    domain_label: str
    primary_document: str
    top_relevance: float
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    timings_ms: Dict[str, float]
    error: Optional[str] = None


@app.get("/")
def root():
    return {"status": "online", "service": app.title, "version": app.version}


@app.get("/health")
def health():
    manifest = read_manifest() or {}
    return {"status": "ok", "corpus_version": manifest.get("corpus_version"),
            "indexed_documents": len(manifest.get("files", {})), "total_chunks": manifest.get("total_chunks", 0)}


@app.get("/documents")
def documents():
    """Indexed documents grouped by domain, plus files that were rejected at ingestion."""
    manifest = read_manifest(fresh=True) or {}
    grouped: Dict[str, Dict[str, Any]] = {}
    for name, info in sorted(manifest.get("files", {}).items()):
        g = grouped.setdefault(info["domain"], {"domain": info["domain"], "label": catalog.domain_label(info["domain"]),
                                                "documents": []})
        g["documents"].append({"source": name, "title": info["title"], "format": info["format"],
                               "chunks": info["chunks"], "language": info["language"]})
    return {"corpus_version": manifest.get("corpus_version"), "domains": list(grouped.values()),
            "rejected": {k: v["reason"] for k, v in manifest.get("rejected", {}).items()}}


@app.post("/ingest")
def trigger_ingest(reset: bool = False):
    return ingest(reset=reset).as_dict()


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = pipeline.answer(request.query, tenant_id=request.tenant_id, use_cache=request.use_cache,
                             domains=request.domains)
    if result.status == pipeline.STATUS_ERROR:
        raise HTTPException(status_code=503, detail=result.error or "generation failed")
    return QueryResponse(**{k: v for k, v in result.to_dict().items() if k in QueryResponse.model_fields})
