"""Extract nội dung chính từ URL."""

from __future__ import annotations

from typing import Any

import requests
import trafilatura
from bs4 import BeautifulSoup


class URLExtractionError(Exception):
    """Lỗi trích xuất URL."""


def extract_url(url: str) -> list[dict[str, Any]]:
    """Trả về nội dung URL ở dạng 1 trang logic."""
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded) if downloaded else None

    if not text:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise URLExtractionError(f"Không thể truy cập URL: {url}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

    if not text or len(text.strip()) < 20:
        raise URLExtractionError(f"Không thể trích xuất nội dung từ URL: {url}")

    return [{"page": 1, "text": text.strip()}]
