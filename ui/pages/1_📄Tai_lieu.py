from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("📄 Tài liệu")

with st.container(border=True):
    st.subheader("Upload PDF")
    files = st.file_uploader("Chọn PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("Upload", use_container_width=True):
        if not files:
            st.warning("Vui lòng chọn ít nhất 1 file PDF.")
        else:
            multipart = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
            with st.spinner("Đang xử lý PDF..."):
                resp = requests.post(f"{BACKEND_URL}/documents/upload", files=multipart, timeout=300)
            if resp.ok:
                st.success("Upload thành công.")
            else:
                st.error(resp.text)

with st.container(border=True):
    st.subheader("Thêm URL")
    url_text = st.text_area("Mỗi dòng 1 URL")
    if st.button("Thêm URL", use_container_width=True):
        urls = [u.strip() for u in url_text.splitlines() if u.strip()]
        resp = requests.post(f"{BACKEND_URL}/documents/urls", json={"urls": urls}, timeout=120)
        if resp.ok:
            st.success("Đã thêm URL.")
        else:
            st.error(resp.text)

st.subheader("Danh sách tài liệu")
resp = requests.get(f"{BACKEND_URL}/documents/", timeout=20)
items = resp.json().get("items", []) if resp.ok else []

for idx, item in enumerate(items, start=1):
    cols = st.columns([0.4, 1.2, 0.6, 0.5, 0.6, 1.2, 0.4])
    cols[0].write(idx)
    cols[1].write(item["name"])
    cols[2].write(item["doc_type"].upper())
    cols[3].write(item["page_count"])
    cols[4].write(item["chunk_count"])
    cols[5].write(item["uploaded_at"])
    if cols[6].button("🗑️", key=f"del_{item['doc_id']}"):
        requests.delete(f"{BACKEND_URL}/documents/{item['doc_id']}", timeout=60)
        st.rerun()

c1, c2 = st.columns(2)
if c1.button("Xóa tất cả", use_container_width=True):
    requests.delete(f"{BACKEND_URL}/documents/", timeout=120)
    st.rerun()
if c2.button("Re-index tất cả", use_container_width=True):
    requests.post(f"{BACKEND_URL}/documents/reindex", timeout=120)
    st.success("Đã re-index.")
