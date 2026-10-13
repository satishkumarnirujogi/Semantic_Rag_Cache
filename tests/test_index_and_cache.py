"""Ingestion idempotency, corpus versioning and cache invalidation, on in-memory Qdrant with fake models."""
import dataclasses
import time

import pytest

from semantic_rag import cache, index_meta, ingest, pipeline, router, settings
from semantic_rag.embeddings import get_embedder

DOC_A = "# Caching\n\n" + "HTTP caches store responses. The no-store directive forbids storing responses. " * 8
DOC_B = "# Tables\n\n" + "Delta Lake checkpoints live in the delta log directory of the table. " * 8


@pytest.fixture
def corpus(tmp_path, memory_qdrant, monkeypatch):
    (tmp_path / "a.md").write_text(DOC_A, encoding="utf-8")
    (tmp_path / "b.md").write_text(DOC_B, encoding="utf-8")
    base = settings.get_settings()
    patched = dataclasses.replace(base, docs_dirs=[tmp_path], chunk_tokens=60, chunk_overlap_tokens=0,
                                  min_relevance=0.3, similarity_threshold=0.9)
    monkeypatch.setattr(settings, "get_settings", lambda: patched)
    for mod in ("ingest", "cache", "index_meta", "retrieval", "pipeline", "router", "catalog", "embeddings",
                "qdrant_db"):
        m = __import__(f"semantic_rag.{mod}", fromlist=["get_settings"])
        if hasattr(m, "get_settings"):
            monkeypatch.setattr(m, "get_settings", lambda: patched)
    return tmp_path


def test_reingest_is_idempotent_no_duplicates(corpus):
    r1 = ingest.ingest()
    r2 = ingest.ingest()
    assert r1.total_chunks == r2.total_chunks > 0
    assert r2.unchanged == ["a.md", "b.md"] and r2.indexed == []
    assert r1.corpus_version == r2.corpus_version and not r2.cache_cleared


def test_changed_file_bumps_version_and_clears_cache(corpus):
    r1 = ingest.ingest()
    q = "what does no-store forbid"
    cache.put_cache(q, get_embedder().encode_query(q), "Storing [S1].", [{"source": "a.md"}], "m", "a.md")
    assert cache.check_cache(q, get_embedder().encode_query(q))[0] is not None

    (corpus / "a.md").write_text(DOC_A + "\nNew paragraph about stale responses and freshness.", encoding="utf-8")
    r2 = ingest.ingest()
    assert r2.indexed == ["a.md"] and r2.corpus_version != r1.corpus_version and r2.cache_cleared
    assert cache.check_cache(q, get_embedder().encode_query(q))[0] is None


def test_removed_file_is_deleted_from_index(corpus):
    ingest.ingest()
    (corpus / "b.md").unlink()
    r = ingest.ingest()
    assert r.removed == ["b.md"]
    assert "b.md" not in index_meta.read_manifest(fresh=True)["files"]


def test_cache_tenant_scope_key_tokens_and_expiry(corpus):
    ingest.ingest()
    emb = get_embedder()
    q = "what does section 16b say"
    cache.put_cache(q, emb.encode_query(q), "answer [S1]", [], "m", "a.md", tenant_id="t1")
    assert cache.check_cache(q, emb.encode_query(q), tenant_id="t1")[0] is not None
    assert cache.check_cache(q, emb.encode_query(q), tenant_id="t2")[0] is None
    q2 = "what does section 18b say"  # near-identical wording, different key token
    assert cache.check_cache(q2, emb.encode_query(q2), tenant_id="t1", threshold=0.5)[0] is None


def test_expired_entry_does_not_hide_valid_one(corpus, monkeypatch):
    ingest.ingest()
    emb = get_embedder()
    q = "where are delta checkpoints stored"
    cache.put_cache(q, emb.encode_query(q), "valid [S1]", [], "m", "b.md")
    # an expired near-duplicate that would rank first
    real_time = time.time
    monkeypatch.setattr(cache.time, "time", lambda: real_time() - 10 * 86400)
    cache.put_cache(q + " exactly", emb.encode_query(q), "expired [S1]", [], "m", "b.md")
    monkeypatch.setattr(cache.time, "time", real_time)
    hit, _ = cache.check_cache(q, emb.encode_query(q))
    assert hit is not None and hit["answer"] == "valid [S1]"
    cache.purge_expired()


def test_pipeline_refuses_without_llm_when_nothing_is_relevant(corpus, monkeypatch):
    ingest.ingest()
    monkeypatch.setattr(router, "_call", lambda *a, **k: pytest.fail("LLM must not be called"))
    r = pipeline.answer("Who won the football world cup in 1990?", use_cache=False, log=False)
    assert r.status == pipeline.STATUS_NOT_FOUND and r.cost == 0.0
    assert r.answer == "I couldn't find an answer to this question in the indexed documents."


def test_pipeline_answers_with_citations_and_caches(corpus, monkeypatch):
    ingest.ingest()
    monkeypatch.setattr(router, "_call", lambda model, messages, json_mode=True: {
        "content": '{"answerable": true, "answer": "Storing responses [S1].", "citations": ["S1"]}',
        "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0})
    r = pipeline.answer("What does the no-store directive forbid for HTTP caches?", log=False)
    assert r.status == pipeline.STATUS_ANSWERED and r.sources and r.sources[0]["source"] == "a.md"
    monkeypatch.setattr(router, "_call", lambda *a, **k: pytest.fail("should be served from cache"))
    r2 = pipeline.answer("What does the no-store directive forbid for HTTP caches?", log=False)
    assert r2.status == pipeline.STATUS_CACHE_HIT and r2.answer == r.answer


def test_paraphrase_needs_llm_verification_and_fails_closed(corpus, monkeypatch):
    ingest.ingest()
    emb = get_embedder()
    cached_q, new_q = "apple net income 2025", "apple net sales 2025"
    cache.put_cache(cached_q, emb.encode_query(cached_q), "income [S1]", [], "m", "a.md")
    calls = []

    def fake_call(model, messages, json_mode=True):
        calls.append(messages[0]["content"])
        return {"content": '{"same": false}', "prompt_tokens": 50, "completion_tokens": 5, "cost": 0.00001}

    monkeypatch.setattr(router, "_call", fake_call)
    hit, cost = cache.check_cache(new_q, emb.encode_query(new_q), threshold=0.5)
    assert hit is None and len(calls) == 1 and cost == 0.00001

    monkeypatch.setattr(router, "_call", lambda *a, **k: (_ for _ in ()).throw(router.LLMError("down")))
    hit, _ = cache.check_cache(new_q, emb.encode_query(new_q), threshold=0.5)
    assert hit is None  # verification failure must never produce a hit


def test_reset_really_empties_on_disk_collection(tmp_path):
    """Regression: embedded Qdrant on Windows reloads old points after delete+recreate."""
    from qdrant_client import QdrantClient, models

    from semantic_rag import qdrant_db

    client = QdrantClient(path=str(tmp_path / "q"))
    qdrant_db.set_client(client)
    try:
        def create():
            client.create_collection("c", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        create()
        client.upsert("c", [models.PointStruct(id=i, vector=[1.0, float(i)]) for i in range(5)])
        qdrant_db.reset_collection("c", create)
        assert client.count("c", exact=True).count == 0
    finally:
        client.close()
        qdrant_db.set_client(None)
