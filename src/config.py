import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter configuration parameters
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")
SITE_NAME = os.getenv("SITE_NAME", "RAG-Semantic-Cache")

# Model Slugs matching OpenRouter naming convention
SMALL_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
LARGE_MODEL = "openrouter/openai/gpt-6-luna"

# Pricing table per 1K tokens for cost tracking (Project 1 requirement)
PRICING_TABLE = {
    "openrouter/deepseek/deepseek-v4.1-flash": {
        "input": 0.00015,   # ~$0.15 per 1M tokens
        "output": 0.0006    # ~$0.60 per 1M tokens
    },
    "openrouter/openai/gpt-6-luna": {
        "input": 0.0001,    # ~$0.10 per 1M tokens
        "output": 0.0005    # ~$0.50 per 1M tokens
    }
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING_TABLE.get(model_name, {"input": 0.0002, "output": 0.0008})
    cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1000.0
    return round(cost, 6)
