"""Chunking theo recursive character splitter."""

from __future__ import annotations

from typing import Any

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, max_len: int, separators: list[str]) -> list[str]:
    if len(text) <= max_len:
        return [text]
    if not separators:
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    sep = separators[0]
    if sep == "":
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) <= max_len:
                current = part
            else:
                chunks.extend(_recursive_split(part, max_len, separators[1:]))
                current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    source: str,
    page: int,
    doc_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """Chia text thành chunks và giữ metadata vị trí ký tự."""
    if not text.strip():
        return []

    pieces = _recursive_split(text, chunk_size, SEPARATORS)
    chunks: list[dict[str, Any]] = []
    cursor = 0

    for idx, piece in enumerate(pieces):
        norm_piece = piece.strip()
        if not norm_piece:
            continue
        pos = text.find(norm_piece, max(0, cursor - chunk_overlap))
        if pos == -1:
            pos = cursor
        start = max(0, pos)
        end = start + len(norm_piece)

        chunks.append(
            {
                "doc_id": doc_id,
                "source": source,
                "page": page,
                "chunk_index": idx,
                "text": norm_piece,
                "char_start": start,
                "char_end": end,
            }
        )
        cursor = max(cursor, end - chunk_overlap)

    return chunks
