import os
import sys
from typing import Dict, Any, Tuple, List
from litellm import completion

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SMALL_MODEL, LARGE_MODEL, OPENROUTER_API_KEY, SITE_URL, SITE_NAME, calculate_cost

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY or ""

COMPLEXITY_KEYWORDS = [
    "compare", "contrast", "why", "difference", "relationship",
    "synthesis", "tradeoff", "analyze", "evaluate", "explain in detail",
    "vergleichen", "unterschied", "voraussetzungen", "rechtsgrundlage",
    "ausnahme", "ermessen", "dauerhafter aufenthalt", "einbuergerung"
]

SYSTEM_PROMPT = """You are an expert legal & administrative assistant for international students and immigrants in Germany.
Answer the user's question based strictly on the provided German document excerpts.
- If the question is in English, reply in English. If in German, reply in German.
- If the context does not contain the answer, explicitly state that you cannot find it in the provided documents.
- Always cite the document filename and page number from the context for every factual assertion.
"""

def classify_query_route(query: str) -> str:
    """Classify incoming query as 'small' or 'large' model based on complexity heuristics."""
    q_lower = query.lower()
    words = q_lower.split()

    # Rule 1: Keyword-based complexity trigger
    for kw in COMPLEXITY_KEYWORDS:
        if kw in q_lower:
            return LARGE_MODEL

    # Rule 2: Token/word length threshold
    if len(words) > 25:
        return LARGE_MODEL

    # Default to small model for fast/cost-efficient queries
    return SMALL_MODEL

def is_answer_weak(answer: str, max_retrieval_score: float = 1.0) -> bool:
    """Check if generated answer is weak or lacks sufficient grounding."""
    if max_retrieval_score < 0.25:
        return True

    ans_lower = answer.lower()
    weak_phrases = [
        "i don't know",
        "does not provide information",
        "no information provided",
        "context does not state",
        "insufficient context",
        "cannot answer",
        "not mentioned in the context",
        "kann ich nicht beantworten",
        "keine informationen",
        "wird nicht erwhnt"
    ]
    for phrase in weak_phrases:
        if phrase in ans_lower:
            return True

    return False

def route_and_generate(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    force_model: str = None
) -> Tuple[str, str, Dict[str, Any], bool]:
    """
    Route query to appropriate model, format bilingual prompt with citations, and escalate if needed.
    Returns: (answer, model_used, stats_dict, escalated_flag)
    """
    selected_model = force_model if force_model else classify_query_route(query)
    max_score = max([c.get("score", 0.0) for c in retrieved_chunks]) if retrieved_chunks else 0.0

    context_str = "\n\n".join([
        f"[Document: {c.get('source', c.get('doc_id', 'unknown.pdf'))}, Page: {c.get('page', 1)}]\n{c.get('text', '')}"
        for c in retrieved_chunks
    ])

    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer with document citations:"

    # First Attempt
    response = completion(
        model=selected_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        extra_headers={
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        },
        temperature=0.1
    )
    answer = response.choices[0].message.content
    usage = response.usage
    cost = calculate_cost(selected_model, usage.prompt_tokens, usage.completion_tokens)

    escalated = False

    # Check for Escalation Safety Net (if small model was used and answer is weak)
    if selected_model == SMALL_MODEL and is_answer_weak(answer, max_score):
        escalated = True
        selected_model = LARGE_MODEL

        response_esc = completion(
            model=selected_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            temperature=0.1
        )
        answer = response_esc.choices[0].message.content
        usage_esc = response_esc.usage
        cost += calculate_cost(selected_model, usage_esc.prompt_tokens, usage_esc.completion_tokens)
        usage.prompt_tokens += usage_esc.prompt_tokens
        usage.completion_tokens += usage_esc.completion_tokens

    stats = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost": cost,
        "model": selected_model
    }

    return answer, selected_model, stats, escalated

if __name__ == "__main__":
    print("Testing router classification...")
    print("German simple ->", classify_query_route("Wie viele Tage darf ich arbeiten?"))
    print("German complex ->", classify_query_route("Vergleichen Sie die Voraussetzungen von § 16b und § 18b AufenthG."))
