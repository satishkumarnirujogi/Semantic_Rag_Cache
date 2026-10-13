"""Central configuration. Every tunable value is read here and nowhere else."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


def _resolve(path: str) -> Path:
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class ModelSpec:
    id: str
    input_per_m: float
    output_per_m: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round((prompt_tokens * self.input_per_m + completion_tokens * self.output_per_m) / 1_000_000, 8)


@dataclass(frozen=True)
class Settings:
    # Corpus
    docs_dirs: List[Path]
    catalog_path: Path

    # Qdrant
    qdrant_mode: str  # "server" or "local"
    qdrant_host: str
    qdrant_port: int
    qdrant_path: Path
    chunks_collection: str
    cache_collection: str
    meta_collection: str

    # Embedding, chunking, retrieval
    embed_model: str
    rerank_model: str
    rerank_enabled: bool
    rerank_max_length: int
    chunk_tokens: int
    chunk_overlap_tokens: int
    candidate_k: int
    context_k: int
    max_context_docs: int
    doc_margin: float
    min_relevance: float
    escalation_relevance: float

    # Semantic cache
    similarity_threshold: float
    cache_ttl_seconds: int
    cache_allow_cross_lingual: bool
    cache_verify: bool

    # LLM
    openrouter_api_key: str
    site_url: str
    site_name: str
    small_model: ModelSpec
    large_model: ModelSpec
    judge_model: ModelSpec
    prices_as_of: str
    llm_timeout_seconds: float
    llm_num_retries: int
    llm_temperature: float
    llm_max_output_tokens: int

    # Telemetry
    telemetry_db: Path


def _load_models(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    models_cfg = _load_models(_resolve(_env("MODELS_CONFIG", "config/models.yaml")))
    llm_cfg = models_cfg.get("llm", {})

    def spec(key: str) -> ModelSpec:
        m = models_cfg[key]
        return ModelSpec(id=m["id"], input_per_m=float(m["input_per_m"]), output_per_m=float(m["output_per_m"]))

    docs_dirs = [_resolve(p.strip()) for p in _env("DOCS_DIRS", "data/pdfs,data/corpus").split(",") if p.strip()]

    return Settings(
        docs_dirs=docs_dirs,
        catalog_path=_resolve(_env("CORPUS_CATALOG", "config/corpus_catalog.yaml")),
        qdrant_mode=_env("QDRANT_MODE", "server").lower(),
        qdrant_host=_env("QDRANT_HOST", "localhost"),
        qdrant_port=int(_env("QDRANT_PORT", "6333")),
        qdrant_path=_resolve(_env("QDRANT_PATH", "qdrant_storage")),
        chunks_collection=_env("CHUNKS_COLLECTION", "corpus_chunks"),
        cache_collection=_env("CACHE_COLLECTION", "semantic_cache"),
        meta_collection=_env("META_COLLECTION", "index_meta"),
        embed_model=_env("EMBED_MODEL", "intfloat/multilingual-e5-base"),
        rerank_model=_env("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        rerank_enabled=_env_bool("RERANK_ENABLED", True),
        rerank_max_length=int(_env("RERANK_MAX_LENGTH", "384")),
        chunk_tokens=int(_env("CHUNK_TOKENS", "320")),
        chunk_overlap_tokens=int(_env("CHUNK_OVERLAP_TOKENS", "48")),
        candidate_k=int(_env("CANDIDATE_K", "20")),
        context_k=int(_env("CONTEXT_K", "6")),
        max_context_docs=int(_env("MAX_CONTEXT_DOCS", "3")),
        doc_margin=float(_env("DOC_MARGIN", "0.25")),
        min_relevance=float(_env("MIN_RELEVANCE", "0.30")),
        escalation_relevance=float(_env("ESCALATION_RELEVANCE", "0.50")),
        similarity_threshold=float(_env("SIMILARITY_THRESHOLD", "0.90")),
        cache_ttl_seconds=int(_env("CACHE_TTL_SECONDS", str(7 * 86400))),
        cache_allow_cross_lingual=_env_bool("CACHE_ALLOW_CROSS_LINGUAL", False),
        cache_verify=_env_bool("CACHE_VERIFY", True),
        openrouter_api_key=_env("OPENROUTER_API_KEY", ""),
        site_url=_env("SITE_URL", "http://localhost:8000"),
        site_name=_env("SITE_NAME", "Semantic-RAG-Cache"),
        small_model=spec("small"),
        large_model=spec("large"),
        judge_model=spec("judge"),
        prices_as_of=str(models_cfg.get("prices_as_of", "unknown")),
        llm_timeout_seconds=float(llm_cfg.get("timeout_seconds", 60)),
        llm_num_retries=int(llm_cfg.get("num_retries", 2)),
        llm_temperature=float(llm_cfg.get("temperature", 0.0)),
        llm_max_output_tokens=int(llm_cfg.get("max_output_tokens", 900)),
        telemetry_db=_resolve(_env("TELEMETRY_DB", "logs/request_logs.db")),
    )
