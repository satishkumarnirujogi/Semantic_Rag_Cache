"""Multi-format document loaders with section-level provenance.

Every loader returns a LoadedDocument made of Sections. A section carries a heading path
(e.g. "Item 7. Management's Discussion > Net Sales") and, for PDFs, a page number, so every
chunk and every citation can point to where the text came from.

Supported: .pdf, .html/.htm/.xhtml (incl. SEC inline-XBRL 10-K filings), .md/.markdown,
.rst, .sgml/.xml (DocBook), .txt
"""
import hashlib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

LOADER_VERSION = "3"  # bump when extraction changes, so ingestion re-indexes everything

MIN_DOCUMENT_CHARS = 400
JUNK_MARKERS = (
    "verify that you're not a robot",
    "verify you are human",
    "awswafintegration",
    "enable javascript and then reload",
    "checking your browser before accessing",
    "access denied",
)


@dataclass
class Section:
    text: str
    heading: str = ""
    page: Optional[int] = None


@dataclass
class LoadedDocument:
    source: str
    path: str
    format: str
    title: str
    sha256: str
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


class DocumentRejected(Exception):
    """Raised when a file has no usable content (e.g. a bot-challenge page saved as .html)."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean_ws(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if not ln:
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


class _SectionBuilder:
    """Collects lines under a heading stack and emits Sections."""

    def __init__(self, title: str = ""):
        self.title = title
        self.stack: List[Tuple[int, str]] = []
        self.buf: List[str] = []
        self.sections: List[Section] = []

    def heading(self, level: int, text: str):
        self.flush()
        text = re.sub(r"\s+", " ", text).strip(" #:")
        if not text:
            return
        self.stack = [(lvl, t) for lvl, t in self.stack if lvl < level]
        self.stack.append((level, text))

    def line(self, text: str):
        self.buf.append(text)

    def flush(self):
        text = _clean_ws("\n".join(self.buf))
        self.buf = []
        if text:
            path = " > ".join(t for _, t in self.stack)
            self.sections.append(Section(text=text, heading=path))


# ---------------------------------------------------------------- PDF

def load_pdf(path: Path) -> Tuple[str, List[Section], Dict[str, str]]:
    import pymupdf

    sections: List[Section] = []
    image_pages = 0
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title", "") or ""
        for i, page in enumerate(doc, start=1):
            text = _clean_ws(page.get_text("text"))
            if len(text) < 50:  # blank pages, covers with a logo only
                image_pages += 1 if page.get_images() else 0
                continue
            sections.append(Section(text=text, heading=f"Page {i}", page=i))
        if not sections and image_pages:
            raise DocumentRejected(f"scanned/image-only PDF ({len(doc)} pages, no text layer); "
                                   "download the text version of the document or OCR it first")
    if not title and sections:
        title = sections[0].text.splitlines()[0][:120]
    return title, sections, {}


# ---------------------------------------------------------------- HTML (incl. SEC inline XBRL)

_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "section", "article", "pre", "blockquote",
               "dd", "dt", "dl", "table", "figure", "figcaption", "td", "th"}
_SEC_HEADING_RE = re.compile(r"^(PART\s+[IVX]+\b.*|Item\s+\d+[A-Z]?\.\s*.*)$", re.IGNORECASE)


def load_html(path: Path) -> Tuple[str, List[Section], Dict[str, str]]:
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    metadata: Dict[str, str] = {}

    # SEC inline-XBRL cover data (registrant, form type, fiscal year) before hidden blocks are removed
    for tag in soup.find_all(["ix:nonnumeric"]):
        name = (tag.get("name") or "").lower()
        for key in ("dei:entityregistrantname", "dei:documenttype", "dei:documentfiscalyearfocus",
                    "dei:documentperiodenddate"):
            if name == key and key not in metadata:
                metadata[key.split(":")[1]] = tag.get_text(" ", strip=True)

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    for tag in soup.find_all(["script", "style", "noscript", "nav", "header", "footer", "aside",
                              "svg", "form", "button", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for tag in soup.find_all(id=re.compile(r"feedback|js_error|breadcrumb|cookie", re.I)):
        tag.decompose()

    root = (soup.find(id="main-col-body") or soup.find("main") or soup.find(id="main-content")
            or soup.body or soup)

    # Tables -> one line per row, cells joined with " | " (keeps financial tables readable)
    for table in reversed(root.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", "%", ")")]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n\n" + "\n".join(rows) + "\n\n"))

    for h in root.find_all(re.compile(r"^h[1-6]$")):
        level = int(h.name[1])
        h.replace_with(soup.new_string(f"\n\n\x00H{level}\x00{h.get_text(' ', strip=True)}\n\n"))

    for tag in root.find_all(_BLOCK_TAGS):
        tag.insert_before(soup.new_string("\n"))
        tag.append(soup.new_string("\n"))

    text = root.get_text("")
    builder = _SectionBuilder(title)
    is_sec = bool(metadata.get("documenttype"))
    for ln in text.splitlines():
        stripped = re.sub(r"\s+", " ", ln).strip()
        m = re.match(r"^\x00H(\d)\x00(.*)$", stripped)
        if m:
            builder.heading(int(m.group(1)), m.group(2))
        elif is_sec and "|" not in stripped and _SEC_HEADING_RE.match(stripped) and len(stripped) < 160:
            builder.heading(1 if stripped.upper().startswith("PART") else 2, stripped)
        else:
            builder.line(ln)
    builder.flush()

    if is_sec and metadata.get("entityregistrantname"):
        year = metadata.get("documentfiscalyearfocus", "")
        title = f"{metadata['entityregistrantname']} Form {metadata['documenttype']} (fiscal year {year})".strip()
    return title, builder.sections, metadata


# ---------------------------------------------------------------- Markdown

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# GitHub-docs product variables; anything else falls back to its last name segment
_LIQUID_NAMES = {"prodname_actions": "GitHub Actions", "prodname_dotcom": "GitHub",
                 "prodname_ghe_server": "GitHub Enterprise Server", "prodname_ghe_cloud": "GitHub Enterprise Cloud",
                 "prodname_github_app": "GitHub App", "prodname_marketplace": "GitHub Marketplace"}


def _liquid_variable(match) -> str:
    name = match.group(1).split(".")[-1]
    return _LIQUID_NAMES.get(name, name.replace("prodname_", "").replace("_", " "))


def load_markdown(path: Path) -> Tuple[str, List[Section], Dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = ""
    fm = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if fm:
        t = re.search(r"^(?:displayTitle|title):\s*[\"']?(.+?)[\"']?\s*$", fm.group(1), re.M)
        title = t.group(1) if t else ""
        text = text[fm.end():]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\{%-?\s*data\s+([\w.]+)\s*-?%\}", _liquid_variable, text)
    text = re.sub(r"\{%-?.*?-?%\}", "", text, flags=re.S)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)
    text = _MD_IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"<[^>\n]+>", "", text)

    builder = _SectionBuilder(title)
    in_code = False
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            in_code = not in_code
            builder.line(ln)
            continue
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*?)\s*(?:\{#[\w-]+\})?\s*$", ln)
        if m:
            builder.heading(len(m.group(1)), m.group(2))
            if not title:
                title = m.group(2).strip(" #")
        else:
            builder.line(ln)
    builder.flush()
    return title, builder.sections, {}


# ---------------------------------------------------------------- reStructuredText

_RST_ADORN = set("=-~*^\"'`#+:._")
_RST_ROLE_RE = re.compile(r":[\w:-]+:`([^`<]*?)\s*(?:<[^>]*>)?`")


def _is_adornment(line: str) -> bool:
    s = line.strip()
    return len(s) >= 3 and len(set(s)) == 1 and s[0] in _RST_ADORN


def load_rst(path: Path) -> Tuple[str, List[Section], Dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    title = ""
    # PEP-style header ("Title: ...")
    for ln in lines[:20]:
        m = re.match(r"^Title:\s*(.+)$", ln)
        if m:
            title = m.group(1).strip()
            break

    # Drop comments and targets (".. foo" without "::", plus their indented bodies)
    cleaned: List[str] = []
    skipping = False
    for ln in lines:
        if skipping:
            if ln.strip() == "" or ln.startswith((" ", "\t")):
                continue
            skipping = False
        if ln.startswith(".. ") or ln.strip() == "..":
            directive = re.match(r"^\.\.\s+([\w:-]+)::\s*(.*)$", ln)
            if directive and directive.group(1) in ("code-block", "code", "ipython", "sourcecode",
                                                     "note", "warning", "tip", "important",
                                                     "versionadded", "versionchanged", "math"):
                cleaned.append(directive.group(2))
                continue
            skipping = True
            continue
        cleaned.append(ln)

    levels: Dict[str, int] = {}
    builder = _SectionBuilder(title)
    i = 0
    while i < len(cleaned):
        ln = cleaned[i]
        nxt = cleaned[i + 1] if i + 1 < len(cleaned) else ""
        # overline + title + underline
        if _is_adornment(ln) and i + 2 < len(cleaned) and _is_adornment(cleaned[i + 2]) and cleaned[i + 1].strip():
            key = "o" + ln.strip()[0]
            levels.setdefault(key, len(levels) + 1)
            builder.heading(levels[key], cleaned[i + 1].strip())
            title = title or cleaned[i + 1].strip()
            i += 3
            continue
        if ln.strip() and not _is_adornment(ln) and _is_adornment(nxt) and len(nxt.strip()) >= len(ln.strip()) - 1:
            key = nxt.strip()[0]
            levels.setdefault(key, len(levels) + 1)
            builder.heading(levels[key], ln.strip())
            title = title or ln.strip()
            i += 2
            continue
        ln = _RST_ROLE_RE.sub(r"\1", ln).replace("``", "")
        builder.line(ln)
        i += 1
    builder.flush()
    return title, builder.sections, {}


# ---------------------------------------------------------------- SGML / DocBook XML

def load_sgml(path: Path) -> Tuple[str, List[Section], Dict[str, str]]:
    from bs4 import BeautifulSoup

    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"&(\w+);", lambda m: {"mdash": "—", "ndash": "–", "amp": "&", "lt": "<", "gt": ">",
                                          "quot": '"'}.get(m.group(1), m.group(1)), text)
    soup = BeautifulSoup(text, "html.parser")
    depth_tags = ("preface", "chapter", "sect1", "sect2", "sect3", "section", "appendix", "part")
    for t in soup.find_all("title"):
        depth = len([p for p in t.parents if p.name in depth_tags]) or 1
        t.replace_with(soup.new_string(f"\n\x00H{depth}\x00{t.get_text(' ', strip=True)}\n"))
    for t in soup.find_all(["para", "listitem", "programlisting", "simpara", "term", "row"]):
        t.insert_before(soup.new_string("\n"))
        t.append(soup.new_string("\n"))

    builder = _SectionBuilder()
    title = ""
    for ln in soup.get_text("").splitlines():
        m = re.match(r"^\s*\x00H(\d)\x00(.*)$", ln)
        if m:
            builder.heading(int(m.group(1)), m.group(2))
            title = title or m.group(2).strip()
        else:
            builder.line(re.sub(r"\s+", " ", ln))
    builder.flush()
    return title, builder.sections, {}


# ---------------------------------------------------------------- plain text

def load_text(path: Path) -> Tuple[str, List[Section], Dict[str, str]]:
    text = _clean_ws(path.read_text(encoding="utf-8", errors="replace"))
    title = text.splitlines()[0][:120] if text else ""
    return title, [Section(text=text)] if text else [], {}


LOADERS: Dict[str, Tuple[str, Callable]] = {
    ".pdf": ("pdf", load_pdf),
    ".html": ("html", load_html),
    ".htm": ("html", load_html),
    ".xhtml": ("html", load_html),
    ".md": ("markdown", load_markdown),
    ".markdown": ("markdown", load_markdown),
    ".rst": ("rst", load_rst),
    ".sgml": ("sgml", load_sgml),
    ".xml": ("sgml", load_sgml),
    ".txt": ("text", load_text),
}


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in LOADERS


def load_document(path: Path) -> LoadedDocument:
    """Load one file. Raises DocumentRejected when it has no usable text."""
    path = Path(path)
    fmt, loader = LOADERS[path.suffix.lower()]
    if path.stat().st_size == 0:
        raise DocumentRejected("file is empty (download incomplete?)")
    try:
        title, sections, metadata = loader(path)
    except DocumentRejected:
        raise
    except Exception as e:  # corrupt or unparseable file: report it instead of crashing the whole ingest
        raise DocumentRejected(f"could not be parsed ({type(e).__name__}: {e})") from e
    doc = LoadedDocument(source=path.name, path=str(path), format=fmt, title=(title or path.stem).strip(),
                         sha256=file_sha256(path), sections=sections, metadata=metadata)
    validate_document(doc, path)
    return doc


def validate_document(doc: LoadedDocument, path: Path) -> None:
    head = path.read_bytes()[:20000].decode("utf-8", errors="ignore").lower()
    if any(m in head for m in JUNK_MARKERS) and doc.char_count < 5000:
        raise DocumentRejected("looks like a bot-challenge/error page, not the real document; download it again from a browser")
    if doc.char_count < MIN_DOCUMENT_CHARS:
        raise DocumentRejected(f"only {doc.char_count} characters of text could be extracted")
