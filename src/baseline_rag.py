import os
import sys
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from litellm import completion

# Ensure src in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SMALL_MODEL, LARGE_MODEL, OPENROUTER_API_KEY, SITE_URL, SITE_NAME, calculate_cost
from loader import load_german_pdfs
from qdrant_db import get_shared_qdrant_client

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY or ""

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
COLLECTION_NAME = "german_docs_baseline"
VECTOR_DIM = 768  # 768 for paraphrase-multilingual-mpnet-base-v2

# Initialize clients & embedding model
qdrant = get_shared_qdrant_client()
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def init_collection(reset: bool = False):
    """Ensure multilingual Qdrant collection exists with 768-dim vector space."""
    collections = [c.name for c in qdrant.get_collections().collections]
    if reset and COLLECTION_NAME in collections:
        qdrant.delete_collection(COLLECTION_NAME)
        collections.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE)
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}).")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into word-based chunks of roughly chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max(1, chunk_size - overlap)):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    return chunks

def ingest_corpus(reset: bool = False, pdf_dir: str = "data/pdfs") -> int:
    """Ingest German law PDFs with provenance metadata into Qdrant."""
    init_collection(reset=reset)
    docs = load_german_pdfs(pdf_dir)
    if not docs:
        print(f"No PDF documents found in {pdf_dir}")
        return 0

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["content"], chunk_size=500, overlap=50)
        for idx, chunk in enumerate(chunks):
            source_name = doc["metadata"]["source"]
            page_num = doc["metadata"]["page"]
            all_chunks.append({
                "text": chunk,
                "source": source_name,
                "page": page_num,
                "chunk_id": f"{source_name}_p{page_num}_c{idx}",
                "language": doc["metadata"].get("language", "de")
            })

    print(f"Ingesting {len(all_chunks)} chunks from {len(docs)} pages into '{COLLECTION_NAME}'...")

    # Embed and upsert in batches
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [b["text"] for b in batch]
        vectors = embedder.encode(texts, show_progress_bar=False).tolist()

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j],
                payload={
                    "text": batch[j]["text"],
                    "source": batch[j]["source"],
                    "page": batch[j]["page"],
                    "chunk_id": batch[j]["chunk_id"],
                    "language": batch[j]["language"],
                    "doc_id": batch[j]["source"],
                    "chunk_index": j
                }
            )
            for j in range(len(batch))
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    total_count = qdrant.get_collection(COLLECTION_NAME).points_count
    print(f"Ingestion complete. Total points in '{COLLECTION_NAME}': {total_count}")
    return total_count

def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant chunks from Qdrant with multilingual embeddings."""
    init_collection(reset=False)
    q_vec = embedder.encode(query).tolist()
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=top_k
    )
    chunks = []
    for hit in response.points:
        payload = hit.payload or {}
        chunks.append({
            "score": hit.score,
            "text": payload.get("text", ""),
            "source": payload.get("source", payload.get("doc_id", "unknown.pdf")),
            "page": payload.get("page", 1),
            "chunk_id": payload.get("chunk_id", ""),
            "language": payload.get("language", "de"),
            "doc_id": payload.get("source", payload.get("doc_id", "unknown.pdf")),
            "chunk_index": payload.get("chunk_index", 0)
        })
    return chunks

def generate_rag_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate bilingual RAG answer strictly citing German document sources."""
    context_str = "\n\n".join([
        f"[Document: {c['source']}, Page: {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    ])

    system_prompt = (
        "You are an expert legal & administrative assistant for international students and immigrants in Germany.\n"
        "Answer the user's question based strictly on the provided German document excerpts.\n"
        "- If the question is in English, reply in English. If in German, reply in German.\n"
        "- If the context does not contain the answer, explicitly state that you cannot find it in the provided documents.\n"
        "- Always cite the document filename and page number from the context for every factual assertion."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer with document citations:"

    response = completion(
        model=SMALL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
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
    cost = calculate_cost(SMALL_MODEL, usage.prompt_tokens, usage.completion_tokens)

    return {
        "answer": answer,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost": cost
    }

# FastAPI App setup
app = FastAPI(title="Multilingual German Law RAG API", version="2.0")

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
    return {"message": "Multilingual German Law RAG API is operational"}

@app.post("/ingest")
def trigger_ingest(reset: bool = False):
    total = ingest_corpus(reset=reset)
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
    print("Ingesting German law corpus into Qdrant...")
    ingest_corpus(reset=False)
