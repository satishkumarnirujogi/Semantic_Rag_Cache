"""Shared Qdrant client.

QDRANT_MODE=server (default) connects to the container from docker-compose.yml and
fails loudly if it is not reachable. QDRANT_MODE=local uses embedded on-disk storage,
which only ONE process can open at a time (do not run the API and Streamlit together).
"""
import atexit
import threading
from typing import Optional

from qdrant_client import QdrantClient, models

from semantic_rag.settings import get_settings

_client: Optional[QdrantClient] = None
_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            s = get_settings()
            if s.qdrant_mode == "local":
                s.qdrant_path.mkdir(parents=True, exist_ok=True)
                _client = QdrantClient(path=str(s.qdrant_path))
            elif s.qdrant_mode == "server":
                client = QdrantClient(host=s.qdrant_host, port=s.qdrant_port, timeout=10.0)
                try:
                    client.get_collections()
                except Exception as e:
                    raise RuntimeError(
                        f"Cannot reach Qdrant at {s.qdrant_host}:{s.qdrant_port}. "
                        "Start it with `docker compose up -d qdrant`, or set QDRANT_MODE=local "
                        "to use embedded single-process storage."
                    ) from e
                _client = client
            else:
                raise ValueError(f"QDRANT_MODE must be 'server' or 'local', got '{s.qdrant_mode}'")
            atexit.register(_client.close)
    return _client


def set_client(client: Optional[QdrantClient]) -> None:
    """Override the shared client (used by tests with an in-memory instance)."""
    global _client
    _client = client


def collection_exists(name: str) -> bool:
    return get_client().collection_exists(name)


def reset_collection(name: str, create) -> None:
    """Drop and recreate a collection, guaranteeing it is empty afterwards.

    Embedded (local) Qdrant on Windows does not remove a deleted collection's storage, and
    re-creating it under the same name reloads the old points. So after recreating we also
    delete every point and verify the collection is empty."""
    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        create()
    client.delete(name, points_selector=models.FilterSelector(filter=models.Filter(must=[])), wait=True)
    left = client.count(name, exact=True).count
    if left:
        raise RuntimeError(f"Could not empty Qdrant collection '{name}' ({left} points remain)")
