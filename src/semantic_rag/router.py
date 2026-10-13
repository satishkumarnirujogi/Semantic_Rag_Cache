"""Cost-aware model routing and grounded generation.

Routing: simple questions go to the small (cheap) model, complex/comparative ones to the
large model. Escalation: if the small model reports it cannot ground an answer although
retrieval found strongly relevant passages, the large model gets one try.

Grounding: the model must return JSON with an explicit `answerable` flag and the [S#]
citations it used (finding A9 replaced phrase matching). An answer without at least one
valid citation is treated as not grounded.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from semantic_rag.catalog import domain_label
from semantic_rag.prompts import SYSTEM_PROMPT, USER_PROMPT, language_name
from semantic_rag.settings import ModelSpec, get_settings

COMPLEXITY_KEYWORDS = (
    "compare", "comparison", "contrast", "difference between", "differences", "versus", " vs ",
    "why", "trade-off", "tradeoff", "relationship", "analyze", "analyse", "evaluate", "implications",
    "explain in detail", "step by step", "pros and cons",
    "vergleich", "unterschied", "warum", "voraussetzungen", "rechtsgrundlage", "ausnahme", "ermessen",
)


class LLMError(RuntimeError):
    pass


@dataclass
class Generation:
    answerable: bool
    answer: str
    citations: List[str]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    escalated: bool = False
    attempts: List[Dict[str, Any]] = field(default_factory=list)


def classify_query_route(query: str) -> str:
    """Return 'small' or 'large'."""
    q = f" {query.lower()} "
    if any(k in q for k in COMPLEXITY_KEYWORDS) or len(q.split()) > 30:
        return "large"
    return "small"


def format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[S{i}] Document: {c.get('title', c.get('source'))} | Domain: {domain_label(c.get('domain', ''))}"
                     f" | Location: {c.get('location', '')}\n{c.get('text', '')}")
    return "\n\n".join(parts)


def parse_model_json(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    return json.loads(m.group(0))


def validate_citations(data: Dict[str, Any], n_sources: int) -> List[str]:
    """Citations the model listed or used inline that refer to a real excerpt."""
    listed = [str(c).strip("[] ") for c in data.get("citations", []) or []]
    inline = re.findall(r"\[(S\d+)\]", data.get("answer", "") or "")
    valid = []
    for c in listed + inline:
        m = re.fullmatch(r"S(\d+)", c)
        if m and 1 <= int(m.group(1)) <= n_sources and c not in valid:
            valid.append(c)
    return valid


def _call(model: ModelSpec, messages: List[Dict[str, str]], json_mode: bool = True) -> Dict[str, Any]:
    from litellm import completion

    s = get_settings()
    if not s.openrouter_api_key:
        raise LLMError("OPENROUTER_API_KEY is not set (see .env.example)")
    os.environ.setdefault("OPENROUTER_API_KEY", s.openrouter_api_key)
    kwargs = dict(model=model.id, messages=messages, temperature=s.llm_temperature,
                  max_tokens=s.llm_max_output_tokens, timeout=s.llm_timeout_seconds,
                  num_retries=s.llm_num_retries, api_key=s.openrouter_api_key,
                  extra_headers={"HTTP-Referer": s.site_url, "X-Title": s.site_name})
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = completion(**kwargs)
    except Exception as e:
        raise LLMError(f"{model.id} call failed: {type(e).__name__}: {e}") from e
    usage = resp.usage
    return {"content": resp.choices[0].message.content or "", "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost": model.cost(usage.prompt_tokens, usage.completion_tokens)}


def _attempt(model: ModelSpec, query: str, context: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(language=language_name(language))},
                {"role": "user", "content": USER_PROMPT.format(context=format_context(context), question=query)}]
    raw = _call(model, messages)
    try:
        data = parse_model_json(raw["content"])
        citations = validate_citations(data, len(context))
        answerable = bool(data.get("answerable")) and bool(str(data.get("answer", "")).strip()) and bool(citations)
        raw.update(answerable=answerable, answer=str(data.get("answer", "")).strip(), citations=citations,
                   parse_error=None)
    except (ValueError, json.JSONDecodeError) as e:
        raw.update(answerable=False, answer="", citations=[], parse_error=str(e))
    raw["model"] = model.id
    return raw


def route_and_generate(query: str, context: List[Dict[str, Any]], language: str, top_relevance: float,
                       force_route: Optional[str] = None) -> Generation:
    s = get_settings()
    route = force_route or classify_query_route(query)
    model = s.large_model if route == "large" else s.small_model

    first = _attempt(model, query, context, language)
    attempts = [first]
    final = first
    escalated = False
    if route == "small" and not first["answerable"] and top_relevance >= s.escalation_relevance:
        escalated = True
        final = _attempt(s.large_model, query, context, language)
        attempts.append(final)

    return Generation(
        answerable=final["answerable"], answer=final["answer"] if final["answerable"] else "",
        citations=final["citations"], model=final["model"],
        prompt_tokens=sum(a["prompt_tokens"] for a in attempts),
        completion_tokens=sum(a["completion_tokens"] for a in attempts),
        cost=round(sum(a["cost"] for a in attempts), 8), escalated=escalated,
        attempts=[{k: a[k] for k in ("model", "answerable", "parse_error", "cost")} for a in attempts],
    )
