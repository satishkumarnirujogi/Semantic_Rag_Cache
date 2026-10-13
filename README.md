# Multi-Domain RAG with Grounded Answers, Semantic Cache & Cost-Aware Routing

A retrieval-augmented QA system over a **multi-industry, multi-format document collection**
(law & AI governance, SEC 10-K filings, cloud/DevOps, data engineering, software, German
immigration law; PDF, HTML/iXBRL, Markdown, reStructuredText, DocBook SGML, text) that

- works out **which domain and which document** a question is about before answering,
- answers **only from the documents, with citations**, in the question's language (English/German),
- says **"I couldn't find an answer to this question in the indexed documents"** instead of guessing,
- caches verified answers semantically and routes each question to the cheapest adequate model.

Every number in this README comes from a script in this repo; see [Evaluation](#evaluation).

---

## How a question is answered

```mermaid
flowchart TD
    Q([Question, EN or DE]) --> E[Embed once<br/>multilingual-e5-base]
    E --> C{Semantic cache<br/>tenant scope · corpus version · TTL<br/>key-token + language guards<br/>exact-content or LLM equivalence check}
    C -- verified hit --> CH([Cached answer + sources])
    C -- miss --> H[Hybrid search<br/>dense + BM25-style sparse, RRF]
    H --> R[Cross-encoder rerank<br/>multilingual]
    R --> D[Document discrimination<br/>keep documents near the best one<br/>→ detected domain]
    D --> G{Relevance gate}
    G -- nothing relevant --> NF([Not found, no LLM call])
    G -- relevant --> RT{Route}
    RT -- simple --> S[Small model: gpt-6-luna]
    RT -- complex --> L[Large model: gpt-6-sol]
    S -- refuses but retrieval strong --> L
    S & L --> J{JSON: answerable + valid [S#] citations?}
    J -- no --> NF
    J -- yes --> A([Answer + cited sources]) --> PC[Write to cache]
```

### Finding the right document
1. **Contextual chunks.** Each chunk is indexed with a header naming its document and section
   (`Apple Inc. Form 10-K (fiscal year 2025) | PART II > Item 7`). A passage that only says
   "the Company's total net sales" is therefore still matched to *Apple*, not Microsoft.
2. **Hybrid search.** Dense multilingual embeddings plus a sparse BM25-style index (Qdrant IDF),
   fused with reciprocal-rank fusion. Sparse search catches exact terms such as `§ 16b`,
   `RFC 9111`, `Cache-Control`.
3. **Reranking.** A multilingual cross-encoder scores every candidate against the question.
4. **Document discrimination.** Candidates are grouped by document. Only documents scoring close
   to the best one go into the prompt (at most 3), so similar text from the wrong document is
   kept out. The best document's domain is reported as the question's domain.

### Not answering when the documents don't say
- **Relevance gate:** if no passage is relevant enough, the answer is "not found" and no LLM is called.
- **Grounded generation:** the model must return `{"answerable": ..., "answer": ..., "citations": [...]}`.
  It is told to use only the excerpts, to ignore look-alike excerpts from other documents, and to
  refuse rather than answer a different question.
- **Citation check:** an answer without at least one citation pointing to a real excerpt is
  treated as not grounded, and the user gets the "not found" message.
- "Not found" answers are never cached.

## Corpus

Documents live in the folders listed in `DOCS_DIRS` (comma-separated). Domains and readable
titles come from [`config/corpus_catalog.yaml`](config/corpus_catalog.yaml). **Files that are not
listed are classified automatically**, by comparing their opening text with the domain descriptions.

| Domain | Documents |
|---|---|
| AI governance, law & regulation | NIST AI 600-1 (EU AI Act and GDPR: see *Known gaps*) |
| Corporate finance (SEC 10-K) | Apple 10-K FY2025, Microsoft 10-K FY2025 |
| Cloud, infrastructure & DevOps | AWS Well-Architected (performance pillar intro), RFC 9111, Kubernetes architecture, GitHub Actions |
| Data engineering & databases | Delta Lake protocol, Iceberg spec, Iceberg Spark queries, Parquet, Arrow, Spark SQL guide/reference, Airflow DAGs, PostgreSQL preface |
| Software & programming | OpenAPI 3.1.0, pandas 10-minute guide, Python tutorial intro, PEP 484 |
| German immigration & student law | 8 German/English PDFs on § 16b AufenthG, work limits, fees, health insurance |

**Ingestion is incremental and validated** (`python -m semantic_rag.ingest`):
- files are hashed; only new or changed files are re-indexed, and deleted files are removed;
- point IDs are deterministic, so re-running never duplicates chunks;
- a corpus version is derived from the file hashes plus the embedding, chunking and catalog settings,
  and the semantic cache is cleared whenever it changes;
- unusable files are **rejected with a reason** instead of polluting the index: empty files,
  bot-challenge pages saved as HTML, image-only PDFs, stubs with almost no text.
  The `/documents` endpoint and the UI's *Corpus* page list them.

Chunks are measured with the embedder's own tokenizer (≤ 320 text tokens plus header, under the
model's 512-token window), follow the document structure (headings, PDF pages, 10-K items) and
keep provenance (`page` or `section path`) for citations.

## Known gaps in the current corpus

| File | Status |
|---|---|
| `01_EU_AI_Act_2024_1689.html`, `02_GDPR_2016_679.html` | Rejected: these are AWS-WAF "verify you're not a robot" pages, not the regulations. |
| `EU AI Act Regulation.pdf` | Rejected: image-only PDF (no text layer). Download the text PDF from EUR-Lex, or OCR it. |
| `GDPR Regulation.pdf` | Rejected: image-only. Its title is `32016R0679R(02)`, which is a **corrigendum** to the GDPR, not the regulation itself. |
| `16_Spark_Structured_Streaming.md` | Rejected: stub page that only links elsewhere. |
| `14_Spark_SQL_Reference.md`, `07_AWS_Well_Architected_Performance.html` | Indexed, but they are index/intro pages with little content. |

## Evaluation

Gold set: [`data/eval/gold_set.json`](data/eval/gold_set.json), **63 questions**. 53 are answerable,
spread over all 6 domains (English and German, each with gold document(s) and an exact evidence
phrase). 10 must be refused: out-of-corpus questions, a year the filing doesn't cover, content the
indexed page only links to, and a missing country breakdown.

Run on 2026-10-13, Windows laptop CPU (Intel, no GPU), corpus version `1178edf2f8c2a5fb`
(28 documents, 1,852 chunks). Raw results: `logs/eval_results_full.csv`, summary: `logs/eval_summary_full.json`.

| Retrieval (53 answerable) | Value |
|---|---|
| Correct domain detected | **100%** |
| Correct document ranked first | 96.2% |
| Evidence chunk among the passages sent to the LLM (recall@context, ≤6) | 98.1% |
| Evidence chunk among hybrid-search candidates (recall@20) | 100% |
| MRR of first evidence chunk after reranking | 0.84 |

| Answers | Value |
|---|---|
| Answerable questions answered | **53 / 53** (0 false refusals) |
| Unanswerable questions correctly refused | **10 / 10** (0 hallucinated answers) |
| ...of which refused by the relevance gate with no LLM call | 6 |
| Mean correctness, judged by `claude-sonnet-5.5` (1-5) | **4.75** (0 judge errors) |
| Citation points to a gold document | 98.1% |
| Generation cost, all 63 questions | $0.029 (judge: $0.083 extra) |
| Latency, median / p95 (CPU reranking dominates) | 6.4 s / 10.8 s |

The two imperfect cases: a NIST data-privacy answer that was correct but incomplete (judge 3/5), and a
German question answered correctly from a different German brochure than the gold one.
Retrieval-only numbers can move by about one question between runs.

Retrieval settings were chosen by measurement. Reranking 30 candidates at 512 tokens
(`logs/eval_summary_retrieval_k30_len512.json`) was **not better** than 20 candidates at 384 tokens
(the default, `logs/eval_summary_retrieval.json`) and took about twice as long. The relevance gate
(`MIN_RELEVANCE=0.30`) sits well below the lowest answerable score (0.80). Unanswerable questions
that still look relevant (e.g. "Apple's net sales in **2030**") are refused by the grounded-generation
step instead.

Reproduce: `python -m semantic_rag.evaluate --retrieval-only` (free) and `python -m semantic_rag.evaluate`.

## Semantic cache calibration

`scripts/calibrate_cache_threshold.py` scores 15 paraphrase pairs and 14 near-miss pairs
(different company, year, metric, unit, § paragraph, directive, ...).

**Finding: no cosine threshold is safe on its own.** "Apple's *net income* in 2025" and
"Apple's *net sales* in 2025" score 0.970, higher than most true paraphrases. A threshold with zero false
positives (0.98) keeps only 7% of paraphrases (`logs/cache_threshold_calibration.csv`).

The cache therefore uses cosine only to find candidates (≥ 0.90). A candidate must also have the same
numbers/years/§ references and the same language. Then either its content words must match exactly,
or the small model must confirm that both questions ask for exactly the same information. That check
costs about $0.00003, versus a full RAG call; a failed check never produces a hit. Finally, the cached
answer's document must be among the top documents retrieved for the new question.

| Rule | Paraphrases served | Near-misses served (false positives) |
|---|---|---|
| cosine ≥ 0.90 only | 14 / 15 | 9 / 14 |
| cosine ≥ 0.98 only | 1 / 15 | 0 / 14 |
| **verified cache (default)** | **14 / 15** | **0 / 14** |

A cache hit returned in about 0.2-0.9 s in the API smoke test, versus 6-20 s for a full answer.
Cross-lingual hits are off by default (`CACHE_ALLOW_CROSS_LINGUAL`), because they would serve an
English answer to a German question.

## Models and cost

[`config/models.yaml`](config/models.yaml) holds the models and their prices (with a "prices as of" date).

| Role | Model (OpenRouter) | USD per 1M tokens in / out |
|---|---|---|
| Small (default route) | `openai/gpt-6-luna` | 0.10 / 0.50 |
| Large (complex questions, escalation) | `openai/gpt-6-sol` | 2.00 / 10.00 |
| Evaluation judge (different family) | `anthropic/claude-sonnet-5.5` | 2.00 / 10.00 |

Routing sends comparison/"why"/long questions to the large model. A small-model refusal is
escalated only when retrieval found strongly relevant passages (`ESCALATION_RELEVANCE`).

## Quickstart

```bash
git clone https://github.com/satishkumarnirujogi/Semantic_Rag_Cache.git
cd Semantic_Rag_Cache
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.lock
pip install --no-deps -e ".[dev]"
copy .env.example .env            # then set OPENROUTER_API_KEY and DOCS_DIRS
```

Qdrant:
- **server mode (recommended):** `docker compose up -d qdrant`, with `QDRANT_MODE=server`.
  The API and the UI can then run at the same time.
- **local mode:** `QDRANT_MODE=local` uses embedded storage in `./qdrant_storage`. Only one
  process can open it at a time, so run ingest, the API **or** the UI.

```bash
python -m semantic_rag.ingest                     # index DOCS_DIRS (first run downloads ~1.5 GB of models)
streamlit run src/semantic_rag/app.py             # chat, corpus browser, metrics
uvicorn semantic_rag.api:app --port 8000          # REST API (POST /query, GET /documents, POST /ingest)
python -m semantic_rag.evaluate --retrieval-only  # free retrieval metrics
python -m semantic_rag.evaluate                   # full evaluation (calls the LLMs)
pytest -m "not slow"                              # fast tests: in-memory Qdrant, fake models
pytest -m slow                                    # tests with the real embedder/reranker
```

The UI uses a white & blue theme (Inter font) defined entirely in [`.streamlit/config.toml`](.streamlit/config.toml); change colors and fonts there, not in code.

Everything in Docker: `docker compose --profile app up -d --build`. API on :8000, UI on :8501.
Set `CORPUS_DIR` to your documents folder.

Example request:

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"What were Apple's total net sales in fiscal 2025?\"}"
```

The response contains `status` (`answered` | `not_found` | `cache_hit`), `answer`, `domain_label`,
`primary_document`, `sources` (document, page/section, relevance), `model_used`, `cost`,
`latency_ms` and per-stage `timings_ms`. `tenant_id` scopes the cache. It is supplied by the caller
and **not authenticated**, so it is not access control.

## Project layout

```
config/models.yaml            models, prices (dated), LLM timeouts/retries
config/corpus_catalog.yaml    domains + per-document titles/domains
data/eval/gold_set.json       63 gold questions (53 answerable, 10 must be refused)
src/semantic_rag/
  settings.py                 all configuration (env + yaml)
  loaders.py                  PDF / HTML+iXBRL / Markdown / RST / SGML / text extraction + validation
  chunking.py                 token-based, structure-aware chunks with contextual headers
  catalog.py                  domain metadata + automatic domain classification
  ingest.py                   incremental hash-based ingestion, corpus versioning
  retrieval.py                hybrid search, rerank, document discrimination
  cache.py                    verified semantic cache
  router.py, prompts.py       routing, grounded JSON generation, escalation
  pipeline.py                 the single answer() used by API, UI and evaluator
  api.py, app.py              FastAPI service, Streamlit UI (chat, corpus, metrics)
  telemetry.py                SQLite (WAL) request log with per-stage timings
  evaluate.py                 retrieval + answer evaluation
scripts/                      calibration, model check, API smoke test, German PDF downloader
tests/                        pytest suite
```
