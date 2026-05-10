from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text


def test_clean_text_normalizes_whitespace() -> None:
    raw = "A   B\n\n\nC\tD"
    assert clean_text(raw) == "A B C D"


def test_chunk_text_has_metadata() -> None:
    text = " ".join(["noi_dung"] * 400)
    chunks = chunk_text(text, source="file.pdf", page=3, doc_id="doc-1", chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1
    assert chunks[0]["source"] == "file.pdf"
    assert chunks[0]["page"] == 3
    assert chunks[0]["doc_id"] == "doc-1"
