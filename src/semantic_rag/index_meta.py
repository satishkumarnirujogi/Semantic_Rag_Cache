"""The index manifest (corpus version + per-file hashes) lives in Qdrant next to the vectors,
so every process that reads the index also sees the matching corpus version."""
import time
import uuid
from typing import Any, Dict, Optional

from qdrant_client.http import models

from semantic_rag.qdrant_db import get_client
from semantic_rag.settings import get_settings

_MANIFEST_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "semantic-rag/index-manifest"))
_cache: Dict[str, Any] = {"at": 0.0, "manifest": None}
_CACHE_SECONDS = 10.0


def _ensure() -> str:
    name = get_settings().meta_collection
    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT))
    return name


def read_manifest(fresh: bool = False) -> Optional[Dict[str, Any]]:
    if not fresh and time.time() - _cache["at"] < _CACHE_SECONDS:
        return _cache["manifest"]
    name = _ensure()
    points = get_client().retrieve(name, ids=[_MANIFEST_ID], with_payload=True)
    manifest = points[0].payload if points else None
    _cache.update(at=time.time(), manifest=manifest)
    return manifest


def write_manifest(manifest: Dict[str, Any]) -> None:
    name = _ensure()
    get_client().upsert(name, points=[models.PointStruct(id=_MANIFEST_ID, vector=[1.0], payload=manifest)])
    _cache.update(at=time.time(), manifest=manifest)


def current_corpus_version() -> str:
    manifest = read_manifest()
    return manifest.get("corpus_version", "none") if manifest else "none"
