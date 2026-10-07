import os
import fitz  # PyMuPDF

def load_german_pdfs(pdf_dir: str = "data/pdfs"):
    """
    Extract text page-by-page from German legal/administrative PDFs
    and retain detailed metadata (source, page, total_pages, language).
    """
    documents = []
    # Resolve path relative to project root or workspace if relative
    if not os.path.isabs(pdf_dir):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        resolved_dir = os.path.join(base_path, pdf_dir)
    else:
        resolved_dir = pdf_dir

    if not os.path.exists(resolved_dir):
        # Fallback to absolute docs directory
        fallback_dir = r"C:\Users\niruj\german_law_rag_docs"
        if os.path.exists(fallback_dir):
            resolved_dir = fallback_dir
        else:
            os.makedirs(resolved_dir, exist_ok=True)
            return documents

    for filename in os.listdir(resolved_dir):
        if not filename.endswith(".pdf"):
            continue
        file_path = os.path.join(resolved_dir, filename)
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if len(text) < 50:  # Skip blank pages or footers
                    continue
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "language": "de"
                    }
                })
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents

if __name__ == "__main__":
    docs = load_german_pdfs("data/pdfs")
    print(f"Loaded {len(docs)} pages from German PDFs.")
    if docs:
        print("Sample Metadata:", docs[0]["metadata"])
        print("Sample Text:", docs[0]["content"][:200])
