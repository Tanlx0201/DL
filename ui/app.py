"""Streamlit shell app + sidebar thống kê."""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Grading Assistant", page_icon="🎓", layout="wide")

st.sidebar.title("🎓 RAG Grading Assistant")
st.sidebar.markdown("---")

try:
    config = requests.get(f"{BACKEND_URL}/admin/config", timeout=10).json().get("values", {})
    stats = requests.get(f"{BACKEND_URL}/admin/stats", timeout=10).json().get("db", {})
except Exception:
    config = {}
    stats = {}

st.sidebar.write(f"🤖 LLM Provider: **{config.get('default_provider', '-')}**")
st.sidebar.write(f"📐 Model: **{config.get('default_model', '-')}**")
st.sidebar.write(f"📊 Top-K: **{config.get('top_k', '-')}**")
st.sidebar.markdown("---")
st.sidebar.write(f"📁 Tài liệu: **{stats.get('total_documents', 0)}**")
st.sidebar.write(f"📄 Chunks: **{stats.get('total_chunks', 0)}**")
st.sidebar.write(f"💬 Queries: **{stats.get('total_queries', 0)}**")

st.title("🎓 RAG Grading Assistant")
st.info("Chọn trang ở menu bên trái: 📄 Tài liệu, 💬 Hỏi đáp, ⚙️ Admin")
