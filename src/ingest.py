import sys
import os

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_rag import ingest_corpus, init_collection
from cache import init_cache_collection

if __name__ == "__main__":
    print("=== Initializing Multilingual German Law RAG Ingestion ===")
    init_collection(reset=True)
    init_cache_collection()
    count = ingest_corpus(reset=False, pdf_dir="data/pdfs")
    print(f"=== Ingestion Complete! Total indexed chunks: {count} ===")
