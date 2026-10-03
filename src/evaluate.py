import os
import sys
import json
import pandas as pd
from typing import Dict, Any, List
from litellm import completion

# Ensure src path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SMALL_MODEL, OPENROUTER_API_KEY, SITE_URL, SITE_NAME, calculate_cost
from baseline_rag import retrieve_chunks, generate_rag_answer

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

EVAL_SET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.json"))
OUTPUT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "baseline_eval_results.csv"))

def judge_answer(question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
    """Use LiteLLM to judge correctness on a scale of 1 to 5."""
    judge_prompt = (
        "You are an expert evaluator for RAG systems.\n"
        "Compare the GENERATED ANSWER against the REFERENCE ANSWER for the given QUESTION.\n"
        "Assign a Correctness Score from 1 to 5 based on factual accuracy, completeness, and alignment with reference answer:\n"
        "5 = Perfectly accurate and complete\n"
        "4 = Mostly accurate with minor omissions\n"
        "3 = Partially accurate\n"
        "2 = Mostly inaccurate or missing key facts\n"
        "1 = Completely incorrect or non-responsive\n\n"
        "Respond strictly in JSON format as follows:\n"
        '{"score": <integer 1-5>, "reason": "<one sentence explanation>"}\n\n'
        f"QUESTION: {question}\n"
        f"REFERENCE ANSWER: {reference_answer}\n"
        f"GENERATED ANSWER: {generated_answer}\n"
    )

    try:
        response = completion(
            model=SMALL_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        score = float(data.get("score", 3))
        reason = data.get("reason", "No rationale provided")
        return {"score": score, "reason": reason}
    except Exception as e:
        print(f"Error in LLM judging: {e}")
        return {"score": 3.0, "reason": f"Evaluation error: {str(e)}"}

def run_evaluation():
    if not os.path.exists(EVAL_SET_PATH):
        print(f"Eval set not found at {EVAL_SET_PATH}")
        return

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Running baseline evaluation on {len(eval_set)} questions...")
    results = []

    total_cost = 0.0
    hits = 0

    for idx, item in enumerate(eval_set, 1):
        q_id = item.get("id", idx)
        question = item["question"]
        ref_answer = item["reference_answer"]
        target_doc = item["source_document"]

        print(f"\n[{idx}/{len(eval_set)}] Question: {question}", flush=True)
        
        # 1. Retrieval
        retrieved_chunks = retrieve_chunks(question, top_k=5)
        retrieved_doc_ids = [c["doc_id"] for c in retrieved_chunks]

        # Check retrieval hit
        is_hit = 1 if any(target_doc in doc_id or doc_id in target_doc for doc_id in retrieved_doc_ids) else 0
        hits += is_hit

        # 2. Generation
        gen_result = generate_rag_answer(question, retrieved_chunks)
        gen_answer = gen_result["answer"]
        query_cost = gen_result["cost"]
        total_cost += query_cost

        # 3. LLM-as-a-judge Evaluation
        judge_res = judge_answer(question, ref_answer, gen_answer)
        score = judge_res["score"]
        reason = judge_res["reason"]

        print(f"  Hit Rate: {is_hit} | Score: {score}/5 | Cost: ${query_cost:.6f}", flush=True)
        print(f"  Judge Reason: {reason}", flush=True)


        results.append({
            "question_id": q_id,
            "question": question,
            "source_document": target_doc,
            "hit_rate": is_hit,
            "correctness_score": score,
            "query_cost": query_cost,
            "generated_answer": gen_answer,
            "judge_reason": reason
        })

    # Save to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    avg_hit_rate = (hits / len(eval_set)) * 100
    avg_score = df["correctness_score"].mean()

    print("\n" + "=" * 50)
    print("      BASELINE RAG EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Questions Evaluated: {len(eval_set)}")
    print(f"Retrieval Hit Rate:        {avg_hit_rate:.2f}% ({hits}/{len(eval_set)})")
    print(f"Average Correctness Score: {avg_score:.2f} / 5.0")
    print(f"Total LLM Cost:            ${total_cost:.6f}")
    print(f"Results saved to:          {OUTPUT_CSV_PATH}")
    print("=" * 50)

if __name__ == "__main__":
    run_evaluation()
