import pytest

from semantic_rag.loaders import DocumentRejected, load_document

WAF_PAGE = """<!DOCTYPE html><html><head><script>window.awsWafCookieDomainList = [];</script></head>
<body><div id="challenge-container"></div><noscript><h1>JavaScript is disabled</h1>
In order to continue, we need to verify that you're not a robot.</noscript></body></html>"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_bot_challenge_page_is_rejected(tmp_path):
    with pytest.raises(DocumentRejected, match="bot-challenge"):
        load_document(_write(tmp_path, "law.html", WAF_PAGE))


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(DocumentRejected, match="empty"):
        load_document(_write(tmp_path, "empty.pdf", ""))


def test_corrupt_pdf_is_rejected_not_crashing(tmp_path):
    with pytest.raises(DocumentRejected):
        load_document(_write(tmp_path, "broken.pdf", "%PDF-1.4 this is not really a pdf " * 40))


def test_markdown_headings_front_matter_and_liquid(tmp_path):
    body = "Some explanatory text about workflows and runners. " * 12
    md = (f"---\ntitle: Understanding things\n---\n<!-- license -->\n# Top\n\n{body}\n\n"
          "## The components of {% data variables.product.prodname_actions %}\n\n"
          f"[link text](http://x) and ![badge](http://img) {body}\n")
    doc = load_document(_write(tmp_path, "doc.md", md))
    assert doc.title == "Understanding things"
    headings = [s.heading for s in doc.sections]
    assert "Top > The components of GitHub Actions" in headings
    text = " ".join(s.text for s in doc.sections)
    assert "link text" in text and "http://x" not in text and "license" not in text


def test_rst_headings_and_comments(tmp_path):
    para = "Arrow arrays are aligned to sixty-four bytes for vectorised access. " * 10
    rst = (".. Licensed to the Apache Software Foundation\n.. more license text\n\n"
           "Columnar Format\n===============\n\n" + para + "\n\nNull count\n----------\n\n"
           "The :ref:`null count <nulls>` is a ``int64``. " + para + "\n")
    doc = load_document(_write(tmp_path, "arrow.rst", rst))
    assert doc.title == "Columnar Format"
    assert [s.heading for s in doc.sections] == ["Columnar Format", "Columnar Format > Null count"]
    text = " ".join(s.text for s in doc.sections)
    assert "Licensed" not in text and "null count is a int64" in text


def test_html_tables_hidden_blocks_and_sec_headings(tmp_path):
    filler = "<p>" + ("The Company designs and sells smartphones and services. " * 15) + "</p>"
    html = f"""<html><body>
      <div style="display:none"><ix:header>hidden xbrl data</ix:header></div>
      <ix:nonnumeric name="dei:EntityRegistrantName">Example Corp</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentType">10-K</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentFiscalYearFocus">2025</ix:nonnumeric>
      <table><tr><td>Item 7.</td><td>MD&amp;A</td><td>21</td></tr></table>
      <div><span>Item 7. Management's Discussion and Analysis</span></div>
      {filler}
      <table><tr><td>Total net sales</td><td>$</td><td>416,161</td><td>$</td><td>391,035</td></tr></table>
    </body></html>"""
    doc = load_document(_write(tmp_path, "10k.htm", html))
    assert doc.title == "Example Corp Form 10-K (fiscal year 2025)"
    text = "\n".join(s.text for s in doc.sections)
    assert "hidden xbrl data" not in text
    assert "Total net sales | 416,161 | 391,035" in text
    assert any(s.heading == "Item 7. Management's Discussion and Analysis" for s in doc.sections)


def test_docbook_sgml(tmp_path):
    para = "PostgreSQL is an object-relational database management system. " * 10
    sgml = f"<preface id='p'><title>Preface</title><para>{para}</para>" \
           f"<sect1><title>What Is PostgreSQL?</title><para>Based on POSTGRES &mdash; {para}</para></sect1></preface>"
    doc = load_document(_write(tmp_path, "intro.sgml", sgml))
    assert doc.title == "Preface"
    assert "Preface > What Is PostgreSQL?" in [s.heading for s in doc.sections]
