"""Small text helpers: language detection, sparse (BM25-style) vectors, cache key tokens."""
import math
import re
import string
import zlib
from collections import Counter
from typing import Dict, Set, Tuple

_GERMAN_HINTS = {
    "der", "die", "das", "und", "ist", "nicht", "ich", "wie", "was", "welche", "welcher", "wann",
    "darf", "dürfen", "kann", "können", "muss", "müssen", "für", "mit", "auf", "eine", "einen",
    "einer", "dem", "den", "des", "im", "zum", "zur", "sind", "wird", "werden", "nach", "bei",
    "über", "pro", "jahr", "gibt", "es", "sie", "wo", "warum", "studierende", "aufenthalt",
}
_ENGLISH_HINTS = {
    "the", "and", "is", "are", "what", "which", "how", "when", "can", "does", "do", "of", "for",
    "with", "on", "in", "to", "a", "an", "be", "by", "per", "year", "why", "where", "who",
}

_STOPWORDS = _ENGLISH_HINTS | {
    "der", "die", "das", "und", "ist", "ein", "eine", "zu", "von", "in", "im", "mit", "auf", "für",
    "it", "its", "this", "that", "as", "at", "or", "from", "was", "were", "has", "have", "i", "you",
}

_WORD_RE = re.compile(r"§|\w+", re.UNICODE)
_KEY_TOKEN_RE = re.compile(r"§\s*\d+[a-z]?|\b\w*\d\w*\b", re.UNICODE | re.IGNORECASE)


def detect_language(text: str) -> str:
    """Return 'de' or 'en' using umlauts and function-word counts (good enough for routing replies)."""
    words = re.findall(r"\w+", text.lower())
    de = sum(w in _GERMAN_HINTS for w in words) + 2 * len(re.findall(r"[äöüß]", text.lower()))
    en = sum(w in _ENGLISH_HINTS for w in words)
    return "de" if de > en else "en"


def normalize_query(query: str) -> str:
    text = query.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("§", "")))
    return " ".join(text.split())


def tokenize(text: str):
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def sparse_doc_vector(text: str) -> Tuple[list, list]:
    """Term-frequency vector with log damping. Qdrant applies IDF (Modifier.IDF) at query time."""
    counts: Dict[int, float] = {}
    for tok, tf in Counter(tokenize(text)).items():
        idx = _index(tok)
        counts[idx] = counts.get(idx, 0.0) + 1.0 + math.log(tf)
    return list(counts.keys()), list(counts.values())


def sparse_query_vector(text: str) -> Tuple[list, list]:
    indices = sorted({_index(t) for t in tokenize(text)})
    return indices, [1.0] * len(indices)


def key_tokens(text: str) -> Set[str]:
    """Tokens that change the meaning of a question if they differ: numbers, years, § references,
    identifiers with digits (e.g. '16b', 'fy2025', 'rfc9111')."""
    return {re.sub(r"\s+", "", t.lower()) for t in _KEY_TOKEN_RE.findall(text)}


_QUESTION_WORDS = {"how", "much", "many", "does", "did", "which", "who", "whom", "whose", "can", "could", "would",
                   "should", "will", "please", "tell", "me", "about", "wie", "viel", "viele", "welche", "welcher",
                   "welches", "wer", "was", "wann", "warum"}


def content_words(text: str) -> Set[str]:
    """Lower-cased content words without stopwords/question words, crudely singularized."""
    out = set()
    for t in tokenize(text):
        if t in _QUESTION_WORDS:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.add(t)
    return out
