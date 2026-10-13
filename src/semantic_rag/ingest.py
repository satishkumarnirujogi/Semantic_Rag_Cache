"""Incremental, hash-based ingestion.

- Each file is hashed; unchanged files are skipped, changed files are re-indexed, deleted
  files are removed from the index.
- Point IDs are deterministic (uuid5 of source, chunk index and text hash), so re-running
  ingestion never duplicates chunks (finding A4).
- The corpus version is a hash of all file hashes plus the embedding/chunking settings. When it
  changes, the semantic cache is cleared, so stale answers are never served (finding A3).
- Files that fail validation (empty, bot-challenge pages, unparseable) are reported, not indexed.

Run:  python -m semantic_rag.ingest            (incremental)
      python -m semantic_rag.ingest --reset    (rebuild everything)
"""
import argparse
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from qdrant_client.http import models

from semantic_rag import catalog
from semantic_rag.cache import clear_cache
from semantic_rag.chunking import CHUNKER_VERSION, chunk_document
from semantic_rag.embeddings import get_embedder
from semantic_rag.index_meta import read_manifest, write_manifest
from semantic_rag.loaders import LOADER_VERSION, DocumentRejected, is_supported, load_document
from semantic_rag.qdrant_db import get_client, reset_collection
from semantic_rag.settings import get_settings
from semantic_rag.textutils import detect_language, sparse_doc_vector


@dataclass
class IngestReport:
    corpus_version: str = ""
    indexed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    total_chunks: int = 0
    cache_cleared: bool = False
    seconds: float = 0.0

    def as_dict(self) -> Dict:
        return self.__dict__.copy()


def settings_fingerprint() -> str:
    s = get_settings()
    parts = [s.embed_model, str(s.chunk_tokens), str(s.chunk_overlap_tokens), LOADER_VERSION, CHUNKER_VERSION,
             s.catalog_path.read_text(encoding="utf-8")]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def ensure_chunks_collection(reset: bool = False) -> None:
    s = get_settings()
    client = get_client()

    def create():
        client.create_collection(
            s.chunks_collection,
            vectors_config={"dense": models.VectorParams(size=get_embedder().dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if s.qdrant_mode == "server":  # embedded mode has no payload indexes; filtering still works
            for key in ("source", "domain", "doc_sha256"):
                client.create_payload_index(s.chunks_collection, key, models.PayloadSchemaType.KEYWORD)

    if reset:
        reset_collection(s.chunks_collection, create)
    elif not client.collection_exists(s.chunks_collection):
        create()


def discover_files() -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for d in get_settings().docs_dirs:
        if not d.exists():
            print(f"[ingest] docs dir not found, skipping: {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and is_supported(p):
                if p.name in files:
                    print(f"[ingest] duplicate file name {p.name} in {d}; keeping {files[p.name]}")
                    continue
                files[p.name] = p
    return files


def _point_id(source: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{chunk_index}|{digest}"))


def _delete_source(source: str) -> None:
    get_client().delete(
        get_settings().chunks_collection,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])),
    )


def _index_document(path: Path) -> Dict:
    s = get_settings()
    embedder = get_embedder()
    doc = load_document(path)
    info = catalog.describe(doc)
    chunks = chunk_document(doc, info.title, embedder.count_tokens, embedder.max_text_tokens(),
                            s.chunk_tokens, s.chunk_overlap_tokens)
    language = detect_language(" ".join(c.text for c in chunks[:5]))

    batch = 32
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        dense = embedder.encode_passages([c.embed_text for c in part])
        points = []
        for c, vec in zip(part, dense):
            idx, vals = sparse_doc_vector(c.embed_text)
            points.append(models.PointStruct(
                id=_point_id(doc.source, c.chunk_index, c.text),
                vector={"dense": vec, "sparse": models.SparseVector(indices=idx, values=vals)},
                payload={
                    "text": c.text, "header": c.header, "heading": c.heading,
                    "page_start": c.page_start, "page_end": c.page_end, "location": c.location,
                    "chunk_index": c.chunk_index, "chunk_id": f"{doc.source}#{c.chunk_index}",
                    "source": doc.source, "title": info.title, "domain": info.domain,
                    "organization": info.organization, "format": doc.format, "language": language,
                    "doc_sha256": doc.sha256,
                },
            ))
        get_client().upsert(s.chunks_collection, points=points)
    return {"sha256": doc.sha256, "chunks": len(chunks), "title": info.title, "domain": info.domain,
            "domain_source": info.domain_source, "format": doc.format, "language": language,
            "organization": info.organization}


def ingest(reset: bool = False) -> IngestReport:
    from semantic_rag.loaders import file_sha256

    started = time.time()
    report = IngestReport()
    fingerprint = settings_fingerprint()
    manifest = None if reset else read_manifest(fresh=True)
    if manifest and manifest.get("settings_fingerprint") != fingerprint:
        print("[ingest] embedding/chunking/catalog settings changed: full rebuild")
        manifest, reset = None, True
    ensure_chunks_collection(reset=reset or manifest is None)

    previous: Dict[str, Dict] = dict((manifest or {}).get("files", {}))
    rejected_before: Dict[str, Dict] = dict((manifest or {}).get("rejected", {}))
    files = discover_files()
    current: Dict[str, Dict] = {}
    rejected: Dict[str, Dict] = {}

    for name in sorted(set(previous) - set(files)):
        _delete_source(name)
        report.removed.append(name)

    for name, path in files.items():
        sha = file_sha256(path)
        if name in previous and previous[name]["sha256"] == sha:
            current[name] = previous[name]
            report.unchanged.append(name)
            continue
        if name in rejected_before and rejected_before[name]["sha256"] == sha:
            rejected[name] = rejected_before[name]
            report.rejected[name] = rejected_before[name]["reason"]
            continue
        if name in previous:
            _delete_source(name)
        try:
            t = time.time()
            current[name] = _index_document(path)
            report.indexed.append(name)
            print(f"[ingest] indexed {name}: {current[name]['chunks']} chunks, domain={current[name]['domain']} "
                  f"({time.time() - t:.1f}s)", flush=True)
        except DocumentRejected as e:
            rejected[name] = {"sha256": sha, "reason": str(e)}
            report.rejected[name] = str(e)
            print(f"[ingest] REJECTED {name}: {e}", flush=True)

    version_basis = json.dumps({k: v["sha256"] for k, v in sorted(current.items())}) + fingerprint
    report.corpus_version = hashlib.sha256(version_basis.encode("utf-8")).hexdigest()[:16]
    old_version = (manifest or {}).get("corpus_version")
    if report.corpus_version != old_version:
        clear_cache()
        report.cache_cleared = True

    report.total_chunks = get_client().count(get_settings().chunks_collection, exact=True).count
    write_manifest({
        "corpus_version": report.corpus_version, "settings_fingerprint": fingerprint,
        "embed_model": get_settings().embed_model, "files": current, "rejected": rejected,
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "total_chunks": report.total_chunks,
    })
    report.seconds = round(time.time() - started, 1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the document corpus into Qdrant.")
    parser.add_argument("--reset", action="store_true", help="drop the index and rebuild from scratch")
    args = parser.parse_args()
    report = ingest(reset=args.reset)
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
