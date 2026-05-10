from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("💬 Hỏi & Đáp")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander(f"▼ Nguồn trích dẫn ({len(message['citations'])})"):
                for c in message["citations"]:
                    with st.container(border=True):
                        st.markdown(f"📄 **{c['source']}**")
                        st.caption(f"Trang {c['page']} · Chunk #{c['chunk_index']} · Score: {c['score']*100:.2f}%")
                        st.write(c["text"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            resp = requests.post(f"{BACKEND_URL}/chat/ask", json={"query": prompt}, timeout=120)
        if not resp.ok:
            answer = f"Lỗi: {resp.text}"
            citations = []
        else:
            data = resp.json()
            answer = data.get("answer", "")
            citations = data.get("citations", [])
        st.markdown(answer)
        if citations:
            with st.expander(f"▼ Nguồn trích dẫn ({len(citations)})"):
                for c in citations:
                    with st.container(border=True):
                        st.markdown(f"📄 **{c['source']}**")
                        st.caption(f"Trang {c['page']} · Chunk #{c['chunk_index']} · Score: {c['score']*100:.2f}%")
                        st.write(c["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
