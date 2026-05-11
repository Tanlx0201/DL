# RAG Grading Assistant

Ứng dụng RAG hỗ trợ giảng viên upload PDF / thêm URL, hỏi đáp theo tài liệu và quản trị cấu hình LLM trực tiếp trên UI.

## Yêu cầu

- Docker + Docker Compose
- (Tùy chọn chạy local) Python 3.11

## Chạy nhanh bằng Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Sau khi chạy:
- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501

## Cấu trúc chính

- `app/`: backend FastAPI (ingestion, retrieval, generation, routers)
- `ui/`: Streamlit multi-page (Tài liệu, Hỏi đáp, Admin)
- `data/`: SQLite + FAISS + uploads (mount volume)

## Luồng cấu hình runtime

1. App đọc `.env` để seed dữ liệu mặc định vào bảng `config` (chỉ khi key chưa tồn tại).
2. Sau khi seed, toàn bộ runtime đọc/ghi từ SQLite.
3. Trang Admin cập nhật cấu hình trực tiếp, áp dụng ngay không cần restart.

## Chức năng chính

- Upload nhiều PDF và xóa từng tài liệu
- Thêm nhiều URL (mỗi dòng một URL)
- OCR fallback cho PDF scan (Tesseract `vie+eng`)
- Chunking recursive (`chunk_size`, `chunk_overlap` cấu hình từ Admin)
- Vector search bằng FAISS (metadata lưu SQLite)
- Chat với citation chi tiết: nguồn, trang, chunk, score, full text
- Admin cấu hình provider/model/API key + xem stats + query logs

## Test

```bash
pip install -r requirements.txt
pytest -q
```

## Screenshots

- Ảnh trang Tài liệu: (placeholder)
- Ảnh trang Hỏi đáp: (placeholder)
- Ảnh trang Admin: (placeholder)
