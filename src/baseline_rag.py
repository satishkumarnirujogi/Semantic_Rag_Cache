import os
import sys
import glob
import uuid
import pypdf
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from litellm import completion

# Ensure src path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SMALL_MODEL, OPENROUTER_API_KEY, SITE_URL, SITE_NAME, calculate_cost

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

COLLECTION_NAME = "baseline_docs"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
CORPUS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "corpus"))

# Initialize clients & embedding model
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def init_collection():
    """Ensure Qdrant collection exists."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created Qdrant collection: '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract clean text from a PDF file using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into word-based chunks of roughly chunk_size words with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_corpus(force: bool = False):
    """Ingest documents from data/corpus/ into Qdrant if collection is empty or force=True."""
    init_collection()
    info = qdrant_client.get_collection(COLLECTION_NAME)
    if info.points_count > 0 and not force:
        print(f"Collection '{COLLECTION_NAME}' already contains {info.points_count} points. Skipping ingestion.")
        return info.points_count

    pdf_files = glob.glob(os.path.join(CORPUS_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {CORPUS_DIR}")
        return 0

    print(f"Starting ingestion of {len(pdf_files)} files from {CORPUS_DIR}...")
    points = []
    point_id_counter = 1

    for file_path in pdf_files:
        doc_id = os.path.basename(file_path)
        full_text = extract_text_from_pdf(file_path)
        if not full_text.strip():
            continue
        chunks = chunk_text(full_text, chunk_size=500, overlap=50)

        # Batch encode chunks for performance
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=point_id_counter,
                vector=vector.tolist(),
                payload={
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "text": chunk
                }
            ))
            point_id_counter += 1

            if len(points) >= 100:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    total_count = qdrant_client.get_collection(COLLECTION_NAME).points_count
    print(f"Ingestion complete. Total points in '{COLLECTION_NAME}': {total_count}")
    return total_count

def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant chunks from Qdrant."""
    query_vector = embedding_model.encode(query).tolist()
    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    results = []
    for hit in response.points:
        results.append({
            "score": hit.score,
            "doc_id": hit.payload.get("doc_id", "unknown"),
            "chunk_index": hit.payload.get("chunk_index", 0),
            "text": hit.payload.get("text", "")
        })
    return results


def generate_rag_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate LLM answer with inline citations using LiteLLM."""
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, 1):
        context_str += f"\n--- Document [{chunk['doc_id']}] Chunk {chunk['chunk_index']} ---\n{chunk['text']}\n"

    system_prompt = (
        "You are a helpful AI research assistant. Answer the user question based strictly on the provided context.\n"
        "Always include inline citations referring to the source document ID (e.g. [Doc: filename.pdf]).\n"
        "If the answer is not in the context, state that clearly."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

    response = completion(
        model=SMALL_MODEL,
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
    cost = calculate_cost(SMALL_MODEL, usage.prompt_tokens, usage.completion_tokens)

    return {
        "answer": answer,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost": cost
    }

# FastAPI App setup
app = FastAPI(title="Baseline RAG API", version="1.0")

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    cost: float

@app.on_event("startup")
def startup_event():
    init_collection()

@app.get("/")
def root():
    return {"message": "Baseline RAG API is operational"}

@app.post("/ingest")
def trigger_ingest(force: bool = False):
    total = ingest_corpus(force=force)
    return {"status": "success", "total_points": total}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chunks = retrieve_chunks(request.query, top_k=request.top_k)
    result = generate_rag_answer(request.query, chunks)

    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        retrieved_chunks=chunks,
        cost=result["cost"]
    )

if __name__ == "__main__":
    print("Ingesting corpus...")
    ingest_corpus(force=False)
