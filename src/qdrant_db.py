import os
import sys
from qdrant_client import QdrantClient

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

_CLIENT_INSTANCE = None

def get_shared_qdrant_client() -> QdrantClient:
    """Return a singleton QdrantClient instance shared across all modules."""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is not None:
        return _CLIENT_INSTANCE

    # Try connecting to remote/Docker Qdrant first
    try:
        c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=1.0)
        c.get_collections()
        _CLIENT_INSTANCE = c
        return _CLIENT_INSTANCE
    except Exception:
        pass

    # Fall back to local embedded disk storage
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    storage_dir = os.path.join(base_dir, "qdrant_storage")
    os.makedirs(storage_dir, exist_ok=True)
    _CLIENT_INSTANCE = QdrantClient(path=storage_dir)
    return _CLIENT_INSTANCE
