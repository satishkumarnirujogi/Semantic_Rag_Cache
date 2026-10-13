"""Checks that need the real models. Run with:  pytest -m slow"""
from pathlib import Path

import pytest

from semantic_rag.chunking import chunk_document
from semantic_rag.embeddings import Embedder, Reranker
from semantic_rag.loaders import load_document
from semantic_rag.settings import get_settings

pytestmark = pytest.mark.slow

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "data" / "pdfs" / "202406_Studierende_Fachkraefteeinwanderung.pdf"


@pytest.fixture(scope="module")
def embedder():
    return Embedder(get_settings().embed_model)


@pytest.fixture(scope="module")
def reranker():
    return Reranker(get_settings().rerank_model)


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="German sample PDF not present")
def test_real_chunks_fit_the_embedder_window(embedder):
    s = get_settings()
    doc = load_document(SAMPLE_PDF)
    chunks = chunk_document(doc, "Studierende und Fachkräfteeinwanderung", embedder.count_tokens,
                            embedder.max_text_tokens(), s.chunk_tokens, s.chunk_overlap_tokens)
    assert chunks
    assert max(embedder.count_tokens(embedder.passage_prefix + c.embed_text) for c in chunks) <= embedder.max_seq_length - 2


def test_cross_lingual_question_matches_german_passage(embedder):
    q = embedder.encode_query("How many days per year can international students work in Germany?")
    right, wrong = embedder.encode_passages([
        "Arbeit neben dem Studium: Bis zu 140 ganze oder 280 halbe Tage im Jahr.",
        "Die Krankenversicherung kostet für Studierende 87,38 Euro im Monat.",
    ])
    assert _cos(q, right) > _cos(q, wrong)


def test_reranker_separates_relevant_from_unrelated(reranker):
    scores = reranker.score("What were Apple's total net sales in 2025?", [
        "Apple Inc. Form 10-K | Item 7\nTotal net sales | 416,161 | 391,035 | 383,285",
        "RFC 9111: HTTP Caching | 5.2.1.5. no-store\nThe no-store request directive indicates that a cache MUST NOT store",
    ])
    assert scores[0] > 0.5 > scores[1]
