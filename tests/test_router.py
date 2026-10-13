from semantic_rag import router
from semantic_rag.prompts import NOT_FOUND
from semantic_rag.settings import get_settings


def test_route_classification():
    assert router.classify_query_route("How many days can I work?") == "small"
    assert router.classify_query_route("Compare Delta Lake and Iceberg checkpoints") == "large"
    assert router.classify_query_route("Vergleichen Sie § 16b und § 18b AufenthG.") == "large"


def test_pricing_makes_routing_worthwhile():
    s = get_settings()
    assert s.large_model.input_per_m > s.small_model.input_per_m
    assert s.large_model.output_per_m > s.small_model.output_per_m
    assert s.judge_model.id.split("/")[1] != s.small_model.id.split("/")[1]  # judge from another family


def test_parse_model_json_tolerates_code_fences():
    assert router.parse_model_json('```json\n{"answerable": true, "answer": "x [S1]"}\n```')["answerable"] is True


def test_citations_must_point_to_real_excerpts():
    data = {"answer": "Fact [S1]. Other [S9].", "citations": ["S2", "S7"]}
    assert router.validate_citations(data, n_sources=3) == ["S2", "S1"]


def _fake_call_factory(responses, calls):
    def fake_call(model, messages, json_mode=True):
        calls.append(model.id)
        return {"content": responses.pop(0), "prompt_tokens": 100, "completion_tokens": 20,
                "cost": model.cost(100, 20)}
    return fake_call


CONTEXT = [{"title": "Doc", "domain": "cloud_devops", "location": "p. 1", "text": "RFC 9111 obsoletes RFC 7234."}]


def test_uncited_answer_is_not_grounded(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "_call", _fake_call_factory(
        ['{"answerable": true, "answer": "It obsoletes RFC 7234.", "citations": []}'], calls))
    gen = router.route_and_generate("Which RFC?", CONTEXT, "en", top_relevance=0.2)
    assert gen.answerable is False and gen.answer == ""


def test_small_model_refusal_escalates_when_retrieval_is_strong(monkeypatch):
    s = get_settings()
    calls = []
    monkeypatch.setattr(router, "_call", _fake_call_factory([
        '{"answerable": false, "answer": "", "citations": []}',
        '{"answerable": true, "answer": "RFC 7234 [S1].", "citations": ["S1"]}'], calls))
    gen = router.route_and_generate("Which RFC does it obsolete?", CONTEXT, "en", top_relevance=0.95)
    assert calls == [s.small_model.id, s.large_model.id]
    assert gen.escalated and gen.answerable and gen.model == s.large_model.id
    assert gen.cost == round(s.small_model.cost(100, 20) + s.large_model.cost(100, 20), 8)


def test_no_escalation_when_retrieval_is_weak(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "_call", _fake_call_factory(
        ['{"answerable": false, "answer": "", "citations": []}'], calls))
    gen = router.route_and_generate("Something unrelated?", CONTEXT, "en", top_relevance=0.2)
    assert len(calls) == 1 and not gen.escalated and not gen.answerable


def test_not_found_messages_exist_for_both_languages():
    assert "couldn't find" in NOT_FOUND["en"] and "keine Antwort" in NOT_FOUND["de"]
