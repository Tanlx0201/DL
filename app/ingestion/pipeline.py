"""Ingestion pipeline cho PDF/URL."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from app.config import AppConfig
from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.pdf_extractor import extract_pdf
from app.ingestion.url_extractor import extract_url
from app.models.database import DatabaseManager
from app.retrieval.embedder import Embedder
from app.retrieval.faiss_store import FAISSStore


class IngestionPipeline:
    """Điều phối trích xuất, chunking, embedding, indexing."""

    def __init__(
        self,
        db: DatabaseManager,
        config: AppConfig,
        embedder: Embedder,
        faiss_store: FAISSStore,
        upload_dir: str,
    ) -> None:
        self.db = db
        self.config = config
        self.embedder = embedder
        self.faiss_store = faiss_store
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def ingest_pdf(self, filename: str, content: bytes) -> str:
        file_path = self.upload_dir / filename
        file_path.write_bytes(content)

        pages = extract_pdf(str(file_path))
        if not pages:
            raise ValueError("Không đọc được nội dung PDF.")

        doc_id = self.db.create_document(
            name=filename,
            doc_type="pdf",
            source=filename,
            page_count=len(pages),
            chunk_count=0,
        )

        chunk_size = int(self.config.get("chunk_size"))
        overlap = int(self.config.get("chunk_overlap"))
        chunks = []
        for page in pages:
            cleaned = clean_text(page["text"])
            chunks.extend(
                chunk_text(
                    cleaned,
                    source=filename,
                    page=int(page["page"]),
                    doc_id=doc_id,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
            )

        if not chunks:
            self.db.delete_document(doc_id)
            file_path.unlink(missing_ok=True)
            raise ValueError("PDF không có nội dung để index.")

        embeddings = self.embedder.embed_texts([c["text"] for c in chunks])
        chunk_ids = self.db.insert_chunks(chunks, embeddings)
        self.db.update_document_chunk_count(doc_id, len(chunk_ids))
        self.faiss_store.add(embeddings, chunk_ids)
        self.faiss_store.save()
        return doc_id

    def ingest_url(self, url: str) -> str:
        pages = extract_url(url)
        domain = urlparse(url).netloc or url
        doc_id = self.db.create_document(
            name=domain,
            doc_type="url",
            source=url,
            page_count=1,
            chunk_count=0,
        )

        chunk_size = int(self.config.get("chunk_size"))
        overlap = int(self.config.get("chunk_overlap"))
        chunks = []
        for page in pages:
            cleaned = clean_text(page["text"])
            chunks.extend(
                chunk_text(
                    cleaned,
                    source=domain,
                    page=1,
                    doc_id=doc_id,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )
            )

        if not chunks:
            self.db.delete_document(doc_id)
            raise ValueError("URL không có nội dung để index.")

        embeddings = self.embedder.embed_texts([c["text"] for c in chunks])
        chunk_ids = self.db.insert_chunks(chunks, embeddings)
        self.db.update_document_chunk_count(doc_id, len(chunk_ids))
        self.faiss_store.add(embeddings, chunk_ids)
        self.faiss_store.save()
        return doc_id

    def delete_document(self, doc_id: str) -> None:
        doc = self.db.get_document(doc_id)
        if not doc:
            return

        if doc["doc_type"] == "pdf":
            (self.upload_dir / doc["name"]).unlink(missing_ok=True)

        self.db.delete_document(doc_id)
        all_embeddings, all_chunk_ids = self.db.get_all_chunk_embeddings()
        self.faiss_store.rebuild(all_embeddings, all_chunk_ids)
        self.faiss_store.save()

    def clear_all(self) -> None:
        for file_path in self.upload_dir.glob("*.pdf"):
            file_path.unlink(missing_ok=True)
        self.db.clear_documents()
        self.faiss_store.rebuild(None, [])
        self.faiss_store.save()

    def reindex(self) -> None:
        all_embeddings, all_chunk_ids = self.db.get_all_chunk_embeddings()
        self.faiss_store.rebuild(all_embeddings, all_chunk_ids)
        self.faiss_store.save()
