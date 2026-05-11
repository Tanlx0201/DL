"""Làm sạch văn bản đầu vào."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Chuẩn hóa khoảng trắng và loại bỏ dòng trống thừa."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
