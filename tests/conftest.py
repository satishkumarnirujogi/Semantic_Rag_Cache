import hashlib
import math
import re

import pytest
from qdrant_client import QdrantClient

from semantic_rag import cache, catalog, embeddings, index_meta, qdrant_db


class FakeEmbedder:
    """Deterministic bag-of-words embedder: same words -> same vector. No model download."""

    model_name = "fake"
    dim = 64
    max_seq_length = 120
    query_prefix = passage_prefix = ""

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def count_tokens(self, text):
        return len(text.split())

    def max_text_tokens(self):
        return self.max_seq_length - 2

    def encode_queries(self, texts):
        return [self._vec(t) for t in texts]

    def encode_query(self, text):
        return self._vec(text)

    def encode_passages(self, texts, batch_size=16):
        return [self._vec(t) for t in texts]


class FakeReranker:
    """Relevance = share of question words found in the passage."""

    def score(self, query, passages):
        q = set(re.findall(r"\w+", query.lower()))
        return [len(q & set(re.findall(r"\w+", p.lower()))) / max(1, len(q)) for p in passages]


@pytest.fixture
def memory_qdrant(monkeypatch):
    """In-memory Qdrant + fake models; isolated per test."""
    def reset():
        cache.reset_initialized()
        index_meta._cache.update(at=0.0, manifest=None)
        catalog._domain_vectors.cache_clear()

    qdrant_db.set_client(QdrantClient(":memory:"))
    previous = embeddings.set_models(FakeEmbedder(), FakeReranker())
    reset()
    yield qdrant_db.get_client()
    embeddings.set_models(*previous)
    qdrant_db.set_client(None)
    reset()
