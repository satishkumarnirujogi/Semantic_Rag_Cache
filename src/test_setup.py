import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def main():
    print("1. Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333)
    collections = client.get_collections()
    print(f"   Connected! Collections present: {collections.collections}")

    print("2. Testing local embedding model download...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode("Hello RAG baseline")
    print(f"   Embedding model ready! Vector dimension: {len(vec)}")

    print("\nEnvironment and infrastructure ready for Step 0 (Shared Baseline).")

if __name__ == "__main__":
    main()
