"""Token-based, structure-aware chunking.

Chunks are measured with the embedder's own tokenizer and never exceed what the embedder
can read (finding A2: the old 500-word chunks were cut at 128 tokens). Each chunk also gets
a short contextual header (document title + section path), which is embedded and indexed
with the chunk. This is what lets a passage that only says "the Company's net sales" still
be matched to "Apple" or "Microsoft".
"""
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from semantic_rag.loaders import LoadedDocument

CHUNKER_VERSION = "2"  # part of the index fingerprint; bump when chunk boundaries change

_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÄÖÜ0-9(\"'§])")


@dataclass
class Chunk:
    text: str
    heading: str
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_index: int
    header: str

    @property
    def location(self) -> str:
        if self.page_start:
            pages = (f"p. {self.page_start}" if self.page_start == self.page_end
                     else f"pp. {self.page_start}-{self.page_end}")
            return pages
        return self.heading or "start of document"

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


@dataclass
class _Unit:
    text: str
    tokens: int
    heading: str
    page: Optional[int]
    new_section: bool


def _split_long(text: str, budget: int, count: Callable[[str], int]) -> List[str]:
    """Split text that exceeds the budget: by lines, then sentences, then word windows."""
    if count(text) <= budget:
        return [text]
    for splitter in (lambda t: t.split("\n"), lambda t: _SENTENCE_RE.split(t)):
        parts = [p for p in splitter(text) if p.strip()]
        if len(parts) > 1:
            out: List[str] = []
            for p in parts:
                out.extend(_split_long(p, budget, count))
            return out
    words = text.split()
    out, cur = [], []
    for w in words:
        cur.append(w)
        if count(" ".join(cur)) > budget:
            cur.pop()
            if cur:
                out.append(" ".join(cur))
            cur = [w]
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_document(doc: LoadedDocument, title: str, count_tokens: Callable[[str], int], max_tokens: int,
                   chunk_tokens: int, overlap_tokens: int) -> List[Chunk]:
    """`max_tokens` is the embedder's limit for header + text; `chunk_tokens` the target size."""
    units: List[_Unit] = []
    for section in doc.sections:
        header_tokens = count_tokens(f"{title} | {section.heading}") + 2
        budget = max(32, min(chunk_tokens, max_tokens - header_tokens))
        first = True
        for para in re.split(r"\n\s*\n", section.text):
            para = para.strip()
            if not para:
                continue
            for piece in _split_long(para, budget, count_tokens):
                units.append(_Unit(piece, count_tokens(piece), section.heading, section.page, first))
                first = False

    chunks: List[Chunk] = []
    cur: List[_Unit] = []

    def budget_for(heading: str) -> int:
        return max(32, min(chunk_tokens, max_tokens - count_tokens(f"{title} | {heading}") - 2))

    def flush():
        nonlocal cur
        if not cur:
            return
        heading = cur[0].heading
        pages = [u.page for u in cur if u.page]
        header = f"{title} | {heading}" if heading else title
        chunks.append(Chunk(text="\n\n".join(u.text for u in cur), heading=heading,
                            page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                            chunk_index=len(chunks), header=header))
        # carry a small overlap of trailing units into the next chunk (same section only)
        carry, total = [], 0
        for u in reversed(cur):
            if total + u.tokens > overlap_tokens:
                break
            carry.insert(0, _Unit(u.text, u.tokens, u.heading, u.page, False))
            total += u.tokens
        cur = carry if len(carry) < len(cur) else []

    def size_with(unit: _Unit) -> int:
        # tokens of the current units plus `unit`, with 2 tokens of slack per paragraph separator
        return sum(x.tokens for x in cur) + unit.tokens + 2 * len(cur)

    for u in units:
        if cur:
            section_break = u.new_section and sum(x.tokens for x in cur) >= chunk_tokens // 2
            if section_break or size_with(u) > budget_for(cur[0].heading):
                flush()
                if u.new_section:
                    cur = []  # no overlap across section boundaries
                # `cur` now holds only overlap copies of already-emitted text, so dropping it loses nothing
                if cur and size_with(u) > budget_for(cur[0].heading):
                    cur = []
        cur.append(u)
    flush()
    return chunks
