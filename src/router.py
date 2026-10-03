import os
import sys
from typing import Dict, Any, Tuple
from litellm import completion

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SMALL_MODEL, LARGE_MODEL, OPENROUTER_API_KEY, SITE_URL, SITE_NAME, calculate_cost

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

COMPLEXITY_KEYWORDS = [
    "compare", "contrast", "why", "difference", "relationship",
    "synthesis", "tradeoff", "analyze", "evaluate", "explain in detail",
    "architectural differences", "deep dive", "implications"
]

def classify_query_route(query: str) -> str:
    """Classify incoming query as 'small' or 'large' based on complexity heuristics."""
    q_lower = query.lower()
    words = q_lower.split()

    # Rule 1: Keyword-based complexity trigger
    for kw in COMPLEXITY_KEYWORDS:
        if kw in q_lower:
            return LARGE_MODEL

    # Rule 2: Token/word length threshold
    if len(words) > 25:
        return LARGE_MODEL

    # Default to small model for fast/cheap baseline queries
    return SMALL_MODEL

def is_answer_weak(answer: str, max_retrieval_score: float = 1.0) -> bool:
    """Check if generated answer is weak or lacks sufficient context."""
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
        "not mentioned in the context"
    ]
    for phrase in weak_phrases:
        if phrase in ans_lower:
            return True

    return False

def route_and_generate(
    query: str,
    retrieved_chunks: list,
    force_model: str = None
) -> Tuple[str, str, Dict[str, Any], bool]:
    """
    Route query to appropriate model, generate answer, and escalate if needed.
    Returns: (answer, model_used, stats_dict, escalated_flag)
    """
    selected_model = force_model if force_model else classify_query_route(query)
    max_score = max([c.get("score", 0.0) for c in retrieved_chunks]) if retrieved_chunks else 0.0

    context_str = ""
    for chunk in retrieved_chunks:
        context_str += f"\n--- Document [{chunk['doc_id']}] Chunk {chunk['chunk_index']} ---\n{chunk['text']}\n"

    system_prompt = (
        "You are an expert AI research assistant. Answer the user question based on the provided context.\n"
        "Always include inline citations referring to the source document ID (e.g. [Doc: filename.pdf]).\n"
        "If the answer is not in the context, state that clearly."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

    # First Attempt
    response = completion(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        extra_headers={
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        }
    )
    answer = response.choices[0].message.content
    usage = response.usage
    cost = calculate_cost(selected_model, usage.prompt_tokens, usage.completion_tokens)

    escalated = False

    # Check for Escalation Safety Net (if small model was used)
    if selected_model == SMALL_MODEL and is_answer_weak(answer, max_score):
        print(f"Escalation triggered for query: '{query[:40]}...'. Rerunning with LARGE_MODEL...")
        escalated = True
        selected_model = LARGE_MODEL

        response_esc = completion(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
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
    print("Simple query ->", classify_query_route("What is Nemotron 3?"))
    print("Complex query ->", classify_query_route("Compare and contrast Mamba vs Transformer attention tradeoffs."))
