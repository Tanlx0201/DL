"""Extract text từ PDF với OCR fallback."""

from __future__ import annotations

from typing import Any

import pdfplumber
import pytesseract


def extract_pdf(filepath: str) -> list[dict[str, Any]]:
    """Trả về danh sách trang: [{"page": n, "text": ...}]"""
    pages: list[dict[str, Any]] = []

    with pdfplumber.open(filepath) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < 50:
                try:
                    image = page.to_image(resolution=300).original
                    text = pytesseract.image_to_string(image, lang="vie+eng").strip()
                except Exception:
                    text = text or ""
            pages.append({"page": idx, "text": text})

    return pages
