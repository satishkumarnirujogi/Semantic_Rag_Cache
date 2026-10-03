import os
import sys

# Ensure current directory is in sys.path when running from src or root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm import completion
from config import SMALL_MODEL, LARGE_MODEL, OPENROUTER_API_KEY, SITE_URL, SITE_NAME, calculate_cost

# Pass OpenRouter headers globally via environment or extra headers
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

def test_model(model_name: str, prompt: str):
    print(f"\nTesting model: {model_name}...")
    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            }
        )
        answer = response.choices[0].message.content
        usage = response.usage
        cost = calculate_cost(model_name, usage.prompt_tokens, usage.completion_tokens)
        
        print(f"  Response: {answer.strip()}")
        print(f"  Tokens -> Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}")
        print(f"  Calculated Cost: ${cost}")
    except Exception as e:
        print(f"  Error connecting to {model_name}: {e}")

if __name__ == "__main__":
    print("--- Verifying OpenRouter Integration ---")
    test_model(SMALL_MODEL, "Say 'Small model active' in 3 words.")
    test_model(LARGE_MODEL, "Say 'Large model active' in 3 words.")
