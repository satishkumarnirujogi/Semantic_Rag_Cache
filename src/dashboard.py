import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="RAG Cost & Performance Dashboard", layout="wide")

st.title("Project 1: Semantic Caching & Cost-Aware Routing Dashboard")
st.markdown("Monitor LLM cost savings, cache hit rates, p95 latency, and route distribution.")

# Resolve absolute path to request_logs.db
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "logs", "request_logs.db")

def load_logs():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["timestamp", "query", "model", "cost", "latency_ms", "cache_hit", "route", "escalated"])
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM requests ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading SQLite database: {e}")
        return pd.DataFrame()

df = load_logs()

if df.empty:
    st.warning("No request logs found yet. Run some queries through your FastAPI app to populate the dashboard!")
else:
    col1, col2, col3, col4 = st.columns(4)
    
    total_requests = len(df)
    cache_hits = df["cache_hit"].sum() if "cache_hit" in df.columns else 0
    hit_rate = (cache_hits / total_requests) * 100 if total_requests > 0 else 0
    total_cost = df["cost"].sum() if "cost" in df.columns else 0.0
    p95_latency = df["latency_ms"].quantile(0.95) if "latency_ms" in df.columns else 0.0

    col1.metric("Total Requests", total_requests)
    col2.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    col3.metric("Total LLM Cost", f"${total_cost:.4f}")
    col4.metric("p95 Latency", f"{p95_latency:.1f} ms")

    st.divider()

    # Route Breakdown & Analytics
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Route Distribution")
        route_counts = df["route"].value_counts()
        st.bar_chart(route_counts)

    with c2:
        st.subheader("Latency Distribution by Route (ms)")
        avg_latencies = df.groupby("route")["latency_ms"].mean()
        st.bar_chart(avg_latencies)

    st.divider()
    
    st.subheader("Recent Request Logs")
    st.dataframe(df.head(50), use_container_width=True)
