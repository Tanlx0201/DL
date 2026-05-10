from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("⚙️ Admin")

cfg_resp = requests.get(f"{BACKEND_URL}/admin/config", timeout=20)
stats_resp = requests.get(f"{BACKEND_URL}/admin/stats", timeout=20)
logs_resp = requests.get(f"{BACKEND_URL}/admin/logs", timeout=20)

values = cfg_resp.json().get("values", {}) if cfg_resp.ok else {}
provider_models = cfg_resp.json().get("provider_models", {}) if cfg_resp.ok else {}
stats = stats_resp.json().get("db", {}) if stats_resp.ok else {}
logs = logs_resp.json().get("items", []) if logs_resp.ok else []

with st.container(border=True):
    st.subheader("Cấu hình LLM")
    provider = st.selectbox("Provider", options=list(provider_models.keys()), index=list(provider_models.keys()).index(values.get("default_provider", "gemini")) if provider_models else 0)
    models = provider_models.get(provider, [])
    default_model = values.get("default_model", models[0] if models else "")
    model_index = models.index(default_model) if default_model in models else 0
    model = st.selectbox("Model", options=models, index=model_index if models else None)

    openai_key = st.text_input("OpenAI API Key", value=values.get("openai_api_key", ""), type="password")
    gemini_key = st.text_input("Gemini API Key", value=values.get("gemini_api_key", ""), type="password")
    anthropic_key = st.text_input("Anthropic API Key", value=values.get("anthropic_api_key", ""), type="password")
    ollama_url = st.text_input("Ollama URL", value=values.get("ollama_base_url", "http://host.docker.internal:11434"))

with st.container(border=True):
    st.subheader("Tham số")
    embedding_provider = st.selectbox("Embedding provider", options=["local", "openai"], index=0 if values.get("embedding_provider", "local") == "local" else 1)
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=int(values.get("top_k", 5)))
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=5000, value=int(values.get("chunk_size", 1000)))
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=int(values.get("chunk_overlap", 200)))
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(values.get("temperature", 0.1)), step=0.05)
    max_tokens = st.number_input("Max Tokens", min_value=128, max_value=8192, value=int(values.get("max_tokens", 2048)))

if st.button("Lưu cấu hình", use_container_width=True):
    payload = {
        "values": {
            "default_provider": provider,
            "default_model": model,
            "openai_api_key": openai_key,
            "gemini_api_key": gemini_key,
            "anthropic_api_key": anthropic_key,
            "ollama_base_url": ollama_url,
            "embedding_provider": embedding_provider,
            "top_k": str(top_k),
            "chunk_size": str(chunk_size),
            "chunk_overlap": str(chunk_overlap),
            "temperature": str(temperature),
            "max_tokens": str(max_tokens),
        }
    }
    resp = requests.post(f"{BACKEND_URL}/admin/config", json=payload, timeout=20)
    if resp.ok:
        st.success("Lưu cấu hình thành công.")
    else:
        st.error(resp.text)

with st.container(border=True):
    st.subheader("Thống kê")
    st.write(f"Tài liệu: {stats.get('total_documents', 0)} | Chunks: {stats.get('total_chunks', 0)} | Queries: {stats.get('total_queries', 0)}")

with st.container(border=True):
    st.subheader("Query Logs")
    for item in logs:
        with st.expander(f"{item['timestamp']} | {item['query_text'][:80]}"):
            st.write(f"Provider: {item['llm_provider']} · Model: {item['llm_model']} · Latency: {item['latency_ms']} ms")
            st.write("Câu hỏi:")
            st.code(item["query_text"])
            st.write("Câu trả lời:")
            st.write(item["answer_text"])
            st.write("Chunks retrieved:")
            st.json(item["retrieved_chunks"])
