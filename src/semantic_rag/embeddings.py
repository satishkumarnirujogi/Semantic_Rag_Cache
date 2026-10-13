"""One lazily loaded embedder and reranker per process, shared by every module."""
import math
import threading
from typing import List, Optional, Sequence, Tuple

from semantic_rag.settings import get_settings

_lock = threading.Lock()
_embedder = None
_reranker = None


class Embedder:
    """Wraps a SentenceTransformer and applies the query/passage prefixes some models need."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.max_seq_length = int(self.model.max_seq_length)
        is_e5 = "e5" in model_name.lower()
        self.query_prefix = "query: " if is_e5 else ""
        self.passage_prefix = "passage: " if is_e5 else ""

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def max_text_tokens(self) -> int:
        """Tokens available for passage text after special tokens and the prefix."""
        return self.max_seq_length - 2 - self.count_tokens(self.passage_prefix)

    def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self.model.encode([self.query_prefix + t for t in texts], normalize_embeddings=True,
                                 show_progress_bar=False)
        return vecs.tolist()

    def encode_query(self, text: str) -> List[float]:
        return self.encode_queries([text])[0]

    def encode_passages(self, texts: Sequence[str], batch_size: int = 16) -> List[List[float]]:
        vecs = self.model.encode([self.passage_prefix + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


class Reranker:
    """Cross-encoder; returns relevance as a probability in [0, 1]."""

    def __init__(self, model_name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        logits = self.model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = Embedder(get_settings().embed_model)
    return _embedder


def get_reranker() -> Optional[Reranker]:
    global _reranker
    if not get_settings().rerank_enabled:
        return None
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = Reranker(get_settings().rerank_model, get_settings().rerank_max_length)
    return _reranker


def set_models(embedder=None, reranker=None) -> Tuple:
    """Override the shared models (used by tests with lightweight fakes)."""
    global _embedder, _reranker
    previous = (_embedder, _reranker)
    _embedder, _reranker = embedder, reranker
    return previous
