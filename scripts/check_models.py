"""Check that every configured OpenRouter model answers, and print what one call costs.

Usage:  python scripts/check_models.py
"""
from semantic_rag.router import LLMError, _call
from semantic_rag.settings import get_settings


def main() -> None:
    s = get_settings()
    print(f"Prices as of {s.prices_as_of}")
    for role, model in (("small", s.small_model), ("large", s.large_model), ("judge", s.judge_model)):
        try:
            r = _call(model, [{"role": "user", "content": "Reply with the single word OK."}], json_mode=False)
            print(f"{role:5s} {model.id}: {r['content'].strip()[:40]!r} "
                  f"({r['prompt_tokens']} in / {r['completion_tokens']} out, ${r['cost']:.6f})")
        except LLMError as e:
            print(f"{role:5s} {model.id}: FAILED - {e}")


if __name__ == "__main__":
    main()
