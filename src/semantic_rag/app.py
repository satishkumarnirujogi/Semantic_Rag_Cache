"""Streamlit UI: chat, corpus browser and telemetry dashboard (the old dashboard.py is merged here).

Run:  streamlit run src/semantic_rag/app.py
"""
import pandas as pd
import streamlit as st

from semantic_rag import catalog, pipeline, telemetry
from semantic_rag.cache import ensure_cache_collection, purge_expired
from semantic_rag.embeddings import get_embedder, get_reranker
from semantic_rag.index_meta import read_manifest
from semantic_rag.ingest import ensure_chunks_collection

st.set_page_config(page_title="Multi-Domain RAG", layout="wide")


@st.cache_resource(show_spinner="Loading models and connecting to the index...")
def _startup() -> bool:
    telemetry.init_db()
    get_embedder()
    get_reranker()
    ensure_chunks_collection()
    ensure_cache_collection()
    purge_expired()
    return True


_startup()
manifest = read_manifest(fresh=True) or {}

st.sidebar.title("Multi-Domain RAG")
page = st.sidebar.radio("Navigation", ["Ask the documents", "Corpus", "Metrics"])
all_domains = catalog.domains()


def _render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s.get('label', '')}] {s.get('title') or s.get('source')}**, {s.get('location', '')}  \n"
                        f"`{s.get('source')}` · {catalog.domain_label(s.get('domain', ''))} · relevance {s.get('score', 0):.2f}")


if page == "Ask the documents":
    st.header("Ask the document collection")
    st.caption("Answers come only from the indexed documents, with citations. If the documents don't contain "
               "the answer, the assistant says so instead of guessing. English and German questions are supported.")
    selected = st.sidebar.multiselect("Limit to domains (optional)", list(all_domains),
                                      format_func=lambda k: all_domains[k].label)
    use_cache = st.sidebar.toggle("Use semantic cache", value=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            _render_sources(m.get("sources"))

    if prompt := st.chat_input("e.g. What were Apple's total net sales in fiscal 2025?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                r = pipeline.answer(prompt, use_cache=use_cache, domains=selected or None)
            if r.status == pipeline.STATUS_ERROR:
                body = f":warning: {r.answer or 'Something went wrong.'}\n\n`{r.error}`"
            else:
                meta = [f"status `{r.status}`"]
                if r.domain_label and r.status != pipeline.STATUS_NOT_FOUND:
                    meta.append(f"domain **{r.domain_label}**")
                meta += [f"model `{r.model_used or 'none'}`", f"{r.latency_ms:.0f} ms", f"${r.cost:.5f}"]
                if r.escalated:
                    meta.append("escalated")
                body = f"{r.answer}\n\n---\n<small>{' · '.join(meta)}</small>"
            st.markdown(body, unsafe_allow_html=True)
            _render_sources(r.sources)
        st.session_state.messages.append({"role": "assistant", "content": body, "sources": r.sources})

elif page == "Corpus":
    st.header("Indexed corpus")
    files = manifest.get("files", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(files))
    c2.metric("Chunks", manifest.get("total_chunks", 0))
    c3.metric("Corpus version", manifest.get("corpus_version", "none"))
    if files:
        df = pd.DataFrame([{"domain": catalog.domain_label(v["domain"]), "title": v["title"], "file": k,
                            "format": v["format"], "language": v["language"], "chunks": v["chunks"],
                            "domain assigned by": v.get("domain_source", "")} for k, v in files.items()])
        st.dataframe(df.sort_values(["domain", "title"]), width="stretch", hide_index=True)
    rejected = manifest.get("rejected", {})
    if rejected:
        st.subheader("Files not indexed")
        st.dataframe(pd.DataFrame([{"file": k, "reason": v["reason"]} for k, v in rejected.items()]),
                     width="stretch", hide_index=True)
    st.caption("Add files to a folder in DOCS_DIRS and run `python -m semantic_rag.ingest`; only changed files are re-indexed.")

else:
    st.header("Cost & performance telemetry")
    df = telemetry.load_requests()
    if df.empty:
        st.info("No requests logged yet. Ask a question first.")
    else:
        total = len(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Requests", total)
        c2.metric("Cache hit rate", f"{100 * df['cache_hit'].mean():.1f}%")
        c3.metric("Not-found rate", f"{100 * (df['status'] == 'not_found').mean():.1f}%")
        c4.metric("Total LLM cost", f"${df['cost'].sum():.4f}")
        c5.metric("p95 latency", f"{df['latency_ms'].quantile(0.95):.0f} ms")
        st.divider()
        a, b = st.columns(2)
        with a:
            st.subheader("Requests by route")
            st.bar_chart(df["route"].value_counts())
        with b:
            st.subheader("Mean latency by route (ms)")
            st.bar_chart(df.groupby("route")["latency_ms"].mean())
        stage_cols = [c for c in ("embed_ms", "cache_ms", "search_ms", "rerank_ms", "generation_ms") if c in df]
        if stage_cols:
            st.subheader("Mean time per pipeline stage (ms)")
            st.bar_chart(df[stage_cols].mean())
        if "domain" in df:
            st.subheader("Questions by detected domain")
            st.bar_chart(df["domain"].fillna("unknown").map(catalog.domain_label).value_counts())
        st.subheader("Recent requests")
        st.dataframe(df.head(200), width="stretch")
