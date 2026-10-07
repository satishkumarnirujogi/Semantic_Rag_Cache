import os
import sys
import time
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_rag import retrieve_chunks, init_collection
from cache import check_cache, put_cache, init_cache_collection
from router import route_and_generate, classify_query_route
from logger import log_request, init_db, DB_PATH

# Page configuration
st.set_page_config(
    page_title="German Law RAG Platform: Chat & Metrics",
    layout="wide"
)

# Initialize Qdrant and SQLite DB
init_db()
init_collection()
init_cache_collection()

# Sidebar Navigation
st.sidebar.title("German Law RAG Platform")
page = st.sidebar.radio("Navigation", ["Chat with Legal Corpus", "Live Metrics Dashboard"])

# ==========================================
# PAGE 1: CHAT INTERFACE
# ==========================================
if page == "Chat with Legal Corpus":
    st.header("German Legal & Administrative Assistant")
    st.markdown(
        "Ask questions about student visas (§ 16b AufenthG), work permits, healthcare, and immigration law in Germany in **English** or **German**."
    )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Verified Legal Sources & Pages"):
                    for s in message["sources"]:
                        st.markdown(f"• **{s.get('source', 'Unknown')}** (Page {s.get('page', 1)})")

    # Accept user input
    if prompt := st.chat_input("Ask a question (e.g., 'How many days can international students work in Germany?')..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process via RAG Pipeline (Cache -> Retrieval -> Router -> Escalation -> Logging)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal corpus & checking semantic cache..."):
                start_time = time.time()
                prompt_str = prompt.strip()

                # Define active tenant context
                current_tenant_id = "default_tenant"

                # 1. Check Multilingual Semantic Cache
                cached_res = check_cache(prompt_str, threshold=0.88, tenant_id=current_tenant_id)
                sources = []

                if cached_res:
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    response_text = cached_res["answer"]
                    sources = cached_res.get("sources", [])
                    model_used = "semantic-cache"
                    is_cache_hit = True
                    route_used = "cache"
                    cost = 0.0
                    escalated = False

                    # Log request
                    log_request(
                        query=prompt_str,
                        model=model_used,
                        prompt_tokens=0,
                        completion_tokens=0,
                        cost=0.0,
                        latency_ms=elapsed_ms,
                        cache_hit=True,
                        route="cache",
                        escalated=False
                    )

                    full_display = f"{response_text}\n\n---\n**[CACHE HIT]** *Served from Multilingual Semantic Cache in {elapsed_ms:.1f}ms | Cost: $0.0000*"
                else:
                    # 2. Multilingual Vector Retrieval
                    chunks = retrieve_chunks(prompt_str, top_k=5)

                    # Extract unique document source references for provenance
                    seen_sources = set()
                    for c in chunks:
                        key = (c.get("source", "Unknown"), c.get("page", 1))
                        if key not in seen_sources:
                            seen_sources.add(key)
                            sources.append({"source": key[0], "page": key[1]})

                    # 3. Route & Generate
                    answer, model_used, stats, escalated = route_and_generate(prompt_str, chunks)
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    cost = stats.get("cost", 0.0)
                    is_cache_hit = False
                    route_used = "escalated" if escalated else ("large" if "gpt" in model_used or "luna" in model_used else "small")

                    # 4. Save to Semantic Cache with Sources Scope
                    put_cache(prompt_str, answer, sources=sources, model_used=model_used, tenant_id=current_tenant_id)

                    # 5. Log Request
                    log_request(
                        query=prompt_str,
                        model=model_used,
                        prompt_tokens=stats.get("prompt_tokens", 0),
                        completion_tokens=stats.get("completion_tokens", 0),
                        cost=cost,
                        latency_ms=elapsed_ms,
                        cache_hit=False,
                        route=route_used,
                        escalated=escalated
                    )

                    badge = "ESCALATED TO FRONTIER MODEL" if escalated else f"Model: `{model_used}`"
                    full_display = f"{answer}\n\n---\n{badge} | Route: `{route_used}` | Latency: `{elapsed_ms:.1f}ms` | Cost: `${cost:.5f}`"

                st.markdown(full_display)

                if sources:
                    with st.expander("📚 Verified Legal Sources & Pages"):
                        for s in sources:
                            st.markdown(f"• **{s.get('source', 'Unknown')}** (Page {s.get('page', 1)})")

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_display,
            "sources": sources
        })

# ==========================================
# PAGE 2: METRICS DASHBOARD
# ==========================================
elif page == "Live Metrics Dashboard":
    st.header("Live Cost & Performance Dashboard")
    st.markdown("Real-time telemetry showing cache hit rates, multilingual routing distributions, and cost reductions.")

    if st.button("Refresh Data"):
        st.rerun()

    # Load logs from SQLite
    if not os.path.exists(DB_PATH):
        df = pd.DataFrame(columns=["timestamp", "query", "model", "cost", "latency_ms", "cache_hit", "route", "escalated"])
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM requests ORDER BY timestamp DESC", conn)
        conn.close()

    if df.empty:
        st.info("No telemetry logs recorded yet. Go to the **Chat with Legal Corpus** page, ask questions in German/English, and come back here!")
    else:
        total_requests = len(df)
        cache_hits = df["cache_hit"].sum() if "cache_hit" in df.columns else 0
        hit_rate = (cache_hits / total_requests) * 100 if total_requests > 0 else 0
        total_cost = df["cost"].sum() if "cost" in df.columns else 0.0
        p95_latency = df["latency_ms"].quantile(0.95) if "latency_ms" in df.columns else 0.0

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Requests", total_requests)
        col2.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        col3.metric("Total LLM Cost", f"${total_cost:.5f}")
        col4.metric("p95 Latency", f"{p95_latency:.1f} ms")

        st.divider()

        # Layout for Charts
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Route Split (Cache vs Small vs Large)")
            if "route" in df.columns and not df["route"].isnull().all():
                route_counts = df["route"].value_counts()
                st.bar_chart(route_counts)
            else:
                st.write("No routing data available.")

        with chart_col2:
            st.subheader("Latency Over Time (ms)")
            if "latency_ms" in df.columns and not df["latency_ms"].empty:
                st.line_chart(df[["latency_ms"]])
            else:
                st.write("No latency data available.")

        st.divider()
        st.subheader("Recent Request Logs")
        st.dataframe(df, use_container_width=True)
