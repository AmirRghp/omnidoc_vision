# API & Multimodal Service Contracts (API.md)

## 1. Overview
**OmniDoc-Vision** exposes a RESTful & Streaming FastAPI backend (`src/api/routes.py`) alongside static asset serving for cropped visual citations.

---

## 2. Core Endpoints

### 2.1 `POST /api/v1/ingest`
Parses a PDF document using `IBM Docling`, reconstructs tables, crops figures, generates Approach A visual captions, and indexes all chunks into Qdrant and BM25.
- **Request (`multipart/form-data` or JSON path):**
  ```json
  {
    "pdf_path": "data/raw_pdfs/cloud_database_benchmark_2026.pdf",
    "generate_vlm_captions": true
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "status": "indexed",
    "doc_id": "doc_cloud_db_bench_2026",
    "doc_name": "cloud_database_benchmark_2026.pdf",
    "pages_processed": 6,
    "chunks_created": {
      "text": 18,
      "table": 4,
      "figure": 3,
      "total": 25
    },
    "elapsed_ms": 3420.5
  }
  ```

---

### 2.2 `POST /api/v1/query`
Executes the end-to-end Hybrid Retrieval + Cross-Encoder Reranking + CRAG Validation + Adaptive Multimodal Routing (`FAST_TEXT_TABLE_PATH` vs. `VISUAL_VLM_PATH`).
- **Request Body:**
  ```json
  {
    "query": "بر اساس نمودار تاخیر P99 در صفحه ۴، در چه تعداد اتصالی خطای سیستم جهش می‌کند؟",
    "top_k": 5,
    "modality_filter": null,
    "force_visual_route": false
  }
  ```
- **Response Body (`200 OK`):**
  ```json
  {
    "answer": "بر اساس نمودار تاخیر P99 در صفحه ۴ (`cloud_database_benchmark_2026.pdf`)، تا سقف ۲۰۰ اتصال هم‌زمان، تاخیر زیر ۴۵ میلی‌ثانیه پایدار است؛ اما دقیقاً از مرز **۲۵۰ اتصال هم‌زمان (Concurrent Connections)** شیب نمودار برای دیتابیس استاندارد به شدت صعودی شده و به ۶۲۰ میلی‌ثانیه می‌رسد.",
    "routing_path": "VISUAL_VLM_PATH",
    "crag_status": "CONFIDENT",
    "rewritten_query": null,
    "vlm_images_inspected": 1,
    "latency_ms": {
      "retrieval_ms": 84.2,
      "rerank_ms": 62.1,
      "generation_ms": 785.4,
      "total_ms": 931.7
    },
    "citations": [
      {
        "chunk_id": "doc_cloud_db_bench_2026_p04_fig01",
        "doc_name": "cloud_database_benchmark_2026.pdf",
        "page_number": 4,
        "modality": "figure",
        "section_title": "4.2 P99 Latency vs. Concurrency",
        "rerank_score": 0.942,
        "bounding_box": {
          "page": 4,
          "x0": 112.5,
          "y0": 240.0,
          "x1": 525.0,
          "y1": 598.5
        },
        "image_url": "/api/v1/crops/doc_cloud_db_bench_2026_p04_fig01.png",
        "content_preview": "[FIGURE] P99 Write Latency (ms) vs Concurrent Connections (50-500)..."
      }
    ]
  }
  ```

---

### 2.3 `GET /api/v1/crops/{crop_filename}`
Serves the high-resolution `.png` crop of a chart, diagram, or table with strict path-traversal validation.

---

### 2.4 `GET /api/v1/health`
Returns readiness status of Qdrant, BM25 index, Redis cache, and the vLLM `Qwen2.5-VL` server.
