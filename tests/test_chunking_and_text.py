import pytest

from semantic_rag.chunking import chunk_document
from semantic_rag.loaders import LoadedDocument, Section
from semantic_rag.textutils import content_words, detect_language, key_tokens, sparse_query_vector


def _count(text):
    return len(text.split())


def _doc(sections):
    return LoadedDocument(source="d.md", path="d.md", format="markdown", title="T", sha256="x", sections=sections)


def test_chunks_never_exceed_model_limit_including_header():
    long_para = " ".join(f"word{i}" for i in range(1000))
    doc = _doc([Section(text=long_para, heading="A very long section heading"),
                Section(text="short tail paragraph " * 5, heading="Tail")])
    chunks = chunk_document(doc, "Doc title", _count, max_tokens=100, chunk_tokens=80, overlap_tokens=10)
    assert len(chunks) > 10
    for c in chunks:
        assert _count(c.embed_text) <= 100 + 2


def test_chunk_metadata_and_page_ranges():
    doc = _doc([Section(text="alpha " * 30, heading="Page 1", page=1),
                Section(text="beta " * 30, heading="Page 2", page=2)])
    chunks = chunk_document(doc, "Doc", _count, max_tokens=200, chunk_tokens=150, overlap_tokens=0)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    assert chunks[0].header.startswith("Doc | Page 1")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_language_detection():
    assert detect_language("Wie viele Tage dürfen Studierende arbeiten?") == "de"
    assert detect_language("How many days can students work?") == "en"


@pytest.mark.parametrize("a,b,same", [
    ("What does § 16b AufenthG say?", "What does § 18b AufenthG say?", False),
    ("Apple net sales in fiscal 2024", "Apple net sales in fiscal 2025", False),
    ("Apple net sales in 2025", "What were Apple's 2025 net sales?", True),
])
def test_key_tokens_distinguish_near_misses(a, b, same):
    assert (key_tokens(a) == key_tokens(b)) is same


def test_sparse_vector_ignores_stopwords():
    idx, vals = sparse_query_vector("What is the Cache-Control header?")
    assert len(idx) == 3 and all(v == 1.0 for v in vals)  # cache, control, header


def test_content_words_ignore_order_case_and_question_words():
    assert content_words("Where are Delta checkpoints stored?") == content_words("delta checkpoints: where stored")
    assert content_words("Apple net income 2025") != content_words("Apple net sales 2025")


def test_no_paragraph_is_ever_lost():
    """Regression: a size check once dropped already-packed paragraphs when overlap was involved."""
    import random

    rng = random.Random(7)
    sections = []
    for si in range(40):
        paras = [" ".join(f"s{si}p{pi}w{wi}" for wi in range(rng.randint(5, 150))) for pi in range(rng.randint(1, 6))]
        sections.append(Section(text="\n\n".join(paras), heading=f"H{si}"))
    chunks = chunk_document(_doc(sections), "Doc", _count, max_tokens=200, chunk_tokens=160, overlap_tokens=30)
    emitted = set(" ".join(c.text for c in chunks).split())
    for s in sections:
        for word in s.text.split():
            assert word in emitted, word
