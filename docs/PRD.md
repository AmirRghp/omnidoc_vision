# Product Requirements Document (PRD)

## 1. Product Overview
- **Product Name:** OmniDoc-Vision (Multimodal Technical Document Intelligence & Visual RAG)
- **One-Line Description:** A production-grade, bilingual (Persian/English) Multimodal Document Intelligence and Visual RAG platform that combines IBM Docling layout/table reconstruction, BGE-M3 + BM25 hybrid indexing, and adaptive Qwen2.5-VL visual reasoning to accurately extract, retrieve, and reason over complex PDFs, merged-cell tables, architecture diagrams, and telemetry charts within a 16GB VRAM budget.
- **Product Type:** Multimodal Document Intelligence Engine & Adaptive Visual RAG Microservice
- **Primary Platform:** Web Application (FastAPI + Chainlit Visual Attribution Console) & REST API

---

## 2. Problem
### What problem are we solving?
1. **Structural Destruction in Standard RAG:** Over 80% of enterprise engineering documents (cloud architecture whitepapers, database benchmark reports, network topologies, and hardware datasheets) live inside complex PDFs. Naive text parsers (`pypdf`, `pdfplumber`) flatten multi-column layouts, destroy merged-cell tables, and completely ignore charts and architecture diagrams.
2. **Prohibitive VRAM & Latency of Pure End-to-End VLMs:** Sending every PDF page as a high-resolution image to a Vision-Language Model (VLM) on every query consumes massive VRAM, inflates token costs, and increases latency to 3–5 seconds even for simple text lookups.
3. **Lack of Visual Grounding (Blind Citations):** Traditional RAG systems return raw text snippets without showing the operator the actual cropped chart, table structure, page number, or bounding box coordinates `[x0, y0, x1, y1]` where the evidence originated.
4. **Fragility on Messy Real-World Scans:** Most portfolio demos only test clean digital PDFs and fail to quantify robustness against skewed scans, low-DPI compression, or watermarks.

### Why does this problem matter?
In enterprise AI engineering, an assistant that misreads a merged column in a server capacity table or cannot read a latency spike on a P99 throughput chart causes severe engineering miscalculations. Combining structural layout parsing (`Docling`) with an **Adaptive Routing Gateway** (Fast Text/Table Path + Visual VLM Path) delivers both sub-300ms speed on text/tables and deep pixel-level reasoning on charts and diagrams.

---

## 3. Goals
### Primary Goal
Build a self-contained, production-hardened Multimodal RAG platform (`OmniDoc-Vision`) that achieves $\ge 93\%$ Table Cell Extraction F1-score, $\ge 88\%$ Visual Chart/Diagram QA Accuracy, and sub-300ms P95 latency on the Fast Text/Table Path while running comfortably inside a **16GB VRAM** ceiling.

### Secondary Goals
- Unify all three document modalities (**Text Passages**, **Reconstructed Markdown/HTML Tables**, and **High-Res Cropped Figures/Charts**) inside a single hybrid Qdrant (`BGE-M3`) + `BM25` index using Approach A (Technical Visual Captioning + Metadata Indexing).
- Provide visual grounding in the UI by rendering the exact cropped figure/table alongside page numbers and normalized bounding box coordinates.
- Quantitatively benchmark pipeline robustness against synthetic document degradation (scan skew, blur, watermarks) and compare against naive PDF parsers.

---

## 4. Target Users
### User Type 1: Cloud Architects, SREs & Systems Engineers
- **Needs:** Instant, accurate answers from 100+ page infrastructure whitepapers, benchmark tables, and architecture diagrams, backed by visual proof (cropped charts and table previews).
- **Pain Points:** Spending hours cross-referencing multi-page PDF tables and small-font performance graphs.

### User Type 2: AI Engineering Hiring Managers & Technical Interviewers
- **Needs:** Concrete proof that the candidate can handle messy multimodal PDFs, layout analysis, bounding-box attribution, VRAM optimization, and quantitative evaluation across both clean and degraded documents.

---

## 5. User Stories
- **US-01 (Complex Merged-Cell Table QA):** As an engineer, I want to ask a question about a multi-header benchmark table (in Persian or English, e.g., *"در جدول بنچمارک صفحه ۴، توان عملیاتی دیتابیس در حالت ۶۴ رشته و کش فعال چقدر است؟"*) and get the exact numeric value without column-shift hallucinations.
- **US-02 (Direct Chart & Diagram Reasoning):** As an architect, I want to ask about a visual trend or network diagram (e.g., *"بر اساس نمودار تاخیر P99، در چه تعداد اتصالی، زمان پاسخ‌دهی از مرز ۵۰۰ میلی‌ثانیه عبور می‌کند؟"*) so the system automatically routes the cropped chart image to `Qwen2.5-VL` and explains the visual inflection point.
- **US-03 (Visual Grounding & Bounding Box Attribution):** As a user, I want every answer to display the cropped chart or table image inline in the chat along with the source PDF filename, page number, and bounding box `[x0, y0, x1, y1]`.
- **US-04 (Adaptive Fast vs. Visual Routing):** As a platform owner, I want text and table queries answered in `< 300ms` without loading heavy image tensors, reserving multimodal image inference only for queries that retrieve `figure` chunks.
- **US-05 (Automated Multimodal & Noise Evaluation):** As an AI engineer, I want to run `make eval` to generate a full benchmark report comparing `Naive PyPDF RAG` vs. `Docling Text/Table RAG` vs. `OmniDoc-Vision (Full Adaptive Multimodal)` across clean and noisy PDFs.

---

## 6. Core Features

### Feature 1: IBM Docling Structural Ingestion & High-Res Cropping Pipeline
- **Purpose:** Parse complex PDFs into structured reading-order text, reconstructed Markdown/HTML tables, and cropped PNG figures with exact bounding box metadata.
- **Acceptance Criteria:**
  - Uses `Docling` (`DocumentConverter` with `do_table_structure=True` and `generate_picture_images=True`).
  - Saves cropped figures/charts at 150–200 DPI to `data/crops/{doc_id}_p{page}_fig{idx}.png`.
  - Records exact page number and bounding box `[x0, y0, x1, y1]` for every text block, table, and figure.

### Feature 2: Approach A Visual Indexing (Technical VLM Captioning + BGE-M3 Hybrid Index)
- **Purpose:** Index visual figures into the same unified `Qdrant` (Dense `BGE-M3`) + `BM25` (Sparse) retrieval engine without requiring a separate vision embedding model in VRAM.
- **Acceptance Criteria:**
  - During ingestion, each cropped figure is processed by a structured technical captioning prompt on `Qwen2.5-VL-7B-Instruct-AWQ` (with a deterministic local OCR/metadata fallback for offline CI tests) that extracts: chart title, axis labels, legend series, key numeric inflection points, and component topology.
  - All chunks (`chunk_type: "text" | "table" | "figure"`) are indexed in Qdrant + BM25 and reranked via `BAAI/bge-reranker-v2-m3`.

### Feature 3: LangGraph Adaptive Multimodal Routing Gateway
- **Purpose:** Dynamically route queries after CRAG validation between the **Fast Text/Table Path** and the **Visual VLM Reasoning Path**.
- **Acceptance Criteria:**
  - **Fast Path:** Triggered when top reranked chunks contain only `text` or `table` modalities. Calls `Qwen2.5-VL` with text-only messages (`P95 <= 300ms`).
  - **Visual Path:** Triggered when $\ge 1$ top reranked chunk has `chunk_type == "figure"` (up to `MAX_IMAGES_PER_QUERY = 2`). Encodes the cropped PNG(s) and invokes multimodal vision-language reasoning (`P95 <= 1200ms`).
  - Includes CRAG confidence grading (`threshold >= 0.35`) and query rewriting for low-relevance retrievals.

### Feature 4: Chainlit Visual Attribution Console
- **Purpose:** Render rich, verifiable responses with inline cropped charts, reconstructed Markdown tables, routing path badges, and bounding-box metadata.

### Feature 5: Three-Dimensional Multimodal Evaluation Suite (`evals/`)
- **Purpose:** Quantitatively prove superiority over standard text RAG and test resilience under document noise.
- **Benchmarks:**
  1. **Table Reconstruction Benchmark:** Table Cell F1-Score & Structural Preservation (`PyPDF` vs. `Docling`).
  2. **Visual Chart & Diagram QA Benchmark:** Accuracy on 30 visual questions (`Text-Only OCR` vs. `OmniDoc-Vision Adaptive VLM`).
  3. **Document Degradation Robustness Suite:** Accuracy retention under synthetic scan rotation ($\pm 5^\circ$), Gaussian noise, and low-DPI compression.

---

## 7. Non-Goals (Out of Scope)
- Video or audio stream processing (strictly focused on enterprise technical PDFs, tables, diagrams, and charts).
- Training or fine-tuning custom OCR weights from scratch.
- Loading multiple heavy vision models concurrently that exceed the 16GB VRAM ceiling.

---

## 8. Success Metrics
| Metric | Naive PyPDF Baseline | Docling Text/Table Only | OmniDoc-Vision Target | Production CI Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Table Cell Extraction F1** | ~0.52 | 0.94 | **$\ge 0.94$** | $\ge 0.88$ |
| **Visual Chart/Diagram QA Accuracy** | ~0.31 | ~0.48 | **$\ge 0.88$** | $\ge 0.80$ |
| **Noisy/Skewed Scan Recall@5** | ~0.45 | 0.86 | **$\ge 0.89$** | $\ge 0.80$ |
| **Faithfulness (Groundedness)** | ~0.70 | 0.92 | **$\ge 0.96$** | $\ge 0.90$ |
| **Fast Path P95 Latency (Text/Table)** | 190 ms | 260 ms | **$\le 300	ext{ ms}$** | $\le 500	ext{ ms}$ |
| **Visual Path P95 Latency (VLM)** | N/A | N/A | **$\le 1200	ext{ ms}$** | $\le 2000	ext{ ms}$ |
| **Peak VRAM Consumption** | ~8.0 GB | ~9.2 GB | **$\le 11.0	ext{ GB}$ (of 16GB)** | $\le 14.5	ext{ GB}$ |

---

## 9. Definition of Done (DoD)
- [ ] Automated sample PDF generator/downloader (`scripts/build_benchmark_pdfs.py`) creates realistic multi-page technical PDFs containing complex merged-cell tables, latency/throughput charts, and architecture diagrams.
- [ ] `Docling` ingestion pipeline extracts text, reconstructs tables, saves high-res figure crops (`data/crops/`), and generates structured captions.
- [ ] Hybrid retrieval (`BGE-M3` + `BM25` + `RRF` + `BGE-Reranker-v2-m3`) indexes and retrieves across all 3 chunk types (`text`, `table`, `figure`).
- [ ] LangGraph Adaptive Router dynamically switches between Fast Path and Visual VLM Path with CRAG validation.
- [ ] Chainlit UI displays inline cropped images (`cl.Image`), reconstructed tables, page numbers, and bounding boxes.
- [ ] Evaluation suite (`evals/run_multimodal_benchmark.py` & `evals/noise_robustness.py`) produces verifiable Markdown/JSON tables.
- [ ] `ruff`, `mypy`, and `pytest` pass with 0 errors.
