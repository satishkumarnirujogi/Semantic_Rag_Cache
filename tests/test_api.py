from fastapi.testclient import TestClient

from semantic_rag import api, pipeline


def _client():
    return TestClient(api.app)  # no `with`: skips the lifespan (no model loading)


def test_query_returns_pipeline_result(monkeypatch):
    def fake_answer(query, tenant_id, use_cache, domains):
        assert (tenant_id, use_cache, domains) == ("t1", False, ["finance_sec_filings"])
        return pipeline.Answer(query=query, answer="$416,161 million [S1].", status="answered", language="en",
                               model_used="small", route="small", domain="finance_sec_filings",
                               domain_label="Corporate finance (SEC 10-K filings)",
                               primary_document="04_Apple_10K_FY2025.htm",
                               sources=[{"label": "S1", "source": "04_Apple_10K_FY2025.htm"}])

    monkeypatch.setattr(pipeline, "answer", fake_answer)
    r = _client().post("/query", json={"query": "Apple net sales 2025?", "tenant_id": "t1", "use_cache": False,
                                       "domains": ["finance_sec_filings"]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["primary_document"] == "04_Apple_10K_FY2025.htm"


def test_empty_query_is_400():
    assert _client().post("/query", json={"query": "  "}).status_code == 400


def test_generation_failure_is_503_not_a_fake_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "answer", lambda *a, **k: pipeline.Answer(
        query="q", answer="", status="error", language="en", error="timeout"))
    r = _client().post("/query", json={"query": "anything"})
    assert r.status_code == 503 and "timeout" in r.text
