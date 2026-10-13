"""Domain catalog: which industry/domain a document belongs to, plus a readable title.

Known files get their metadata from config/corpus_catalog.yaml. Unknown files are
classified zero-shot by embedding similarity between their opening text and the
domain descriptions, so new documents can be dropped into a docs folder as-is.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import yaml

from semantic_rag.embeddings import get_embedder
from semantic_rag.loaders import LoadedDocument
from semantic_rag.settings import get_settings

_JUNK_TITLES = ("microsoft word", "layout", "untitled", "document")


@dataclass(frozen=True)
class Domain:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class DocInfo:
    title: str
    domain: str
    organization: str
    domain_source: str  # "catalog" or "auto"


@lru_cache(maxsize=1)
def _load() -> Dict:
    with open(get_settings().catalog_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def domains() -> Dict[str, Domain]:
    return {k: Domain(k, v["label"], " ".join(v["description"].split())) for k, v in _load().get("domains", {}).items()}


def domain_label(key: str) -> str:
    d = domains().get(key)
    return d.label if d else key


@lru_cache(maxsize=1)
def _domain_vectors():
    ds = list(domains().values())
    vecs = get_embedder().encode_queries([f"{d.label}: {d.description}" for d in ds])
    return [d.key for d in ds], vecs


def classify_text(text: str) -> str:
    keys, dvecs = _domain_vectors()
    v = get_embedder().encode_passages([text[:2000]])[0]
    scores = [sum(a * b for a, b in zip(v, dv)) for dv in dvecs]
    return keys[max(range(len(keys)), key=scores.__getitem__)]


def _clean_title(doc: LoadedDocument) -> str:
    t = (doc.title or "").strip()
    if len(t) < 4 or t.lower().startswith(_JUNK_TITLES) or t.isdigit():
        t = doc.source.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    return t


def describe(doc: LoadedDocument) -> DocInfo:
    entry: Optional[Dict] = (_load().get("documents") or {}).get(doc.source)
    if entry and entry.get("domain") in domains():
        return DocInfo(title=entry.get("title") or _clean_title(doc), domain=entry["domain"],
                       organization=entry.get("organization", ""), domain_source="catalog")
    sample = f"{doc.title}\n" + "\n".join(s.text for s in doc.sections[:3])
    return DocInfo(title=_clean_title(doc), domain=classify_text(sample),
                   organization=doc.metadata.get("entityregistrantname", ""), domain_source="auto")
