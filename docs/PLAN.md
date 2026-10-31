# Step-by-Step Implementation Plan (PLAN.md)

## Agent Execution Protocol
Before starting any task in this plan, the AI Coding Agent **MUST** adhere to the following workflow:
1. **Read First:** Open and read `AGENTS.md` (in project root), the specific **Reference Docs**, and the relevant **Local Skills** from `.agents/skills/` listed under the current task or workflow stage.
2. **One Task at a Time:** Execute strictly **one** atomic task per iteration. Never combine multiple tasks unless explicitly instructed.
3. **Test-Driven Verification:** Run the **Verification Command** specified at the end of the task before marking it as complete.
4. **Strict Typing & Formatting:** Run `ruff check` and `mypy` on all touched files before concluding any task.

---

## Overview
This plan breaks the implementation of **OmniDoc-Vision** into **8 Phases** and **22 Atomic Tasks**. Follow each task sequentially and run its verification command before moving forward.

---

## Phase 1: Project Scaffolding, Schemas & Synthetic Benchmark PDF Generator
### Task 1.1: Repository Skeleton & Dependencies
- **Reference Docs:** `docs/ARCHITECTURE.md`, `docs/CODE_STYLE.md`
- **Target Files:** `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, `Makefile`
- **Implementation Details:**
  - Create directory tree (`src/common`, `src/schemas`, `src/ingestion`, `src/retrieval`, `src/graph`, `src/services`, `src/api`, `scripts`, `evals`, `tests/unit`, `tests/integration`, `data/raw_pdfs`, `data/crops`).
  - Include dependencies: `docling`, `pypdf`, `pdfplumber`, `reportlab`, `matplotlib`, `Pillow`, `fastapi`, `uvicorn`, `pydantic>=2.9`, `pydantic-settings`, `qdrant-client`, `FlagEmbedding` (or `sentence-transformers`), `bm25s`, `langgraph`, `openai`, `httpx`, `redis`, `chainlit`, `pytest`, `pytest-asyncio`, `ruff`, `mypy`.
- **Verification:** `pip install -r requirements.txt && ruff --version`

### Task 1.2: Core Settings, Logger & Multimodal Pydantic Schemas
- **Reference Docs:** `docs/DATABASE.md`, `docs/API.md`
- **Target Files:** `src/common/config.py`, `src/common/logger.py`, `src/common/exceptions.py`, `src/schemas/document.py`, `src/schemas/api.py`
- **Implementation Details:**
  - Define `ChunkModality(StrEnum)` (`TEXT`, `TABLE`, `FIGURE`), `BoundingBox(BaseModel)`, `MultimodalChunk(BaseModel)`, `VisualCitation(BaseModel)`, and `MultimodalQueryResponse(BaseModel)`.
- **Verification:** `pytest tests/unit/test_schemas.py -v`

### Task 1.3: Automated Technical PDF & Golden Dataset Generator
- **Reference Docs:** `docs/PRD.md`, `docs/TESTING.md`
- **Target Files:** `scripts/build_benchmark_pdfs.py`, `data/golden_multimodal_dataset.json`
- **Implementation Details:**
  - Write a deterministic Python script using `reportlab` and `matplotlib` that generates 3 realistic multi-page engineering PDFs in `data/raw_pdfs/`:
    1. `cloud_database_benchmark_2026.pdf` (Contains merged-cell throughput/memory tables + P99 latency vs. concurrency line charts).
    2. `distributed_cache_architecture.pdf` (Contains cluster topology diagrams + eviction policy comparison tables).
    3. `gpu_inference_capacity_report.pdf` (Contains AWQ vs FP16 VRAM bar charts + batch size latency matrices).
  - Generate `data/golden_multimodal_dataset.json` with 30 bilingual QA pairs (10 Table, 10 Visual Chart/Diagram, 10 Text) with ground-truth answers, expected page numbers, and expected modalities.
- **Verification:** `python scripts/build_benchmark_pdfs.py && ls -lh data/raw_pdfs/`

---

## Phase 2: IBM Docling Ingestion & Approach A Visual Captioning
### Task 2.1: Docling Layout, Table Reconstruction & Figure Cropping Engine
- **Reference Docs:** `docs/ARCHITECTURE.md`, `docs/UX_FLOWS.md` (Flow 1)
- **Target Files:** `src/ingestion/docling_parser.py`, `tests/unit/test_docling_parser.py`
- **Implementation Details:**
  - Implement `DoclingParser` using IBM `docling` (`DocumentConverter` configured with `do_table_structure=True` and `generate_picture_images=True`) plus a clean fallback parser so CI runs smoothly even without heavy GPU weights.
  - Extract reading-order text blocks, reconstruct tables into both Markdown (`export_to_markdown()`) and HTML (`export_to_html()`), and save cropped PNG figures/tables into `data/crops/` with normalized/page bounding boxes `[x0, y0, x1, y1]`.
- **Verification:** `pytest tests/unit/test_docling_parser.py -v`

### Task 2.2: Approach A Visual Captioner (`Qwen2.5-VL` + Offline Fallback)
- **Reference Docs:** `docs/PRD.md` (Feature 2), `docs/SECURITY.md`
- **Target Files:** `src/ingestion/visual_captioner.py`
- **Implementation Details:**
  - Clamp image size to `<= 1280px` using Pillow.
  - Send cropped chart/diagram to `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` to produce a dense technical description (Chart title, X/Y axes, series values, key inflection points). Include a deterministic metadata/OCR fallback when running in offline CI mode.
- **Verification:** `pytest tests/unit/test_visual_captioner.py -v`

### Task 2.3: Unified Hybrid Indexer (`Qdrant BGE-M3` + `BM25s`)
- **Reference Docs:** `docs/DATABASE.md`
- **Target Files:** `src/ingestion/indexer.py`
- **Implementation Details:**
  - Index all `MultimodalChunk` items (`text`, `table`, `figure`) into Qdrant collection `omnidoc_vision_chunks` and local `BM25s` index, preserving `modality`, `bounding_box`, `image_path`, and `table_html` in the payload.
- **Verification:** `pytest tests/unit/test_indexer.py -v`

---

## Phase 3: Multimodal Hybrid Retrieval, RRF & Cross-Encoder Reranking
### Task 3.1: Parallel Dense + Sparse Search, RRF Fusion & BGE Reranker
- **Reference Docs:** `docs/ARCHITECTURE.md`
- **Target Files:** `src/retrieval/dense.py`, `src/retrieval/sparse.py`, `src/retrieval/fusion.py`, `src/retrieval/reranker.py`
- **Implementation Details:**
  - Retrieve Top-20 from Qdrant (`BGE-M3`) and Top-20 from `BM25s` concurrently (`asyncio.gather`), fuse via Reciprocal Rank Fusion ($k=60$), and rerank top 15 candidates down to Top-5 using `BAAI/bge-reranker-v2-m3`.
  - Support optional `modality_filter` (`"text" | "table" | "figure"`).
- **Verification:** `pytest tests/unit/test_retrieval.py -v`

---

## Phase 4: LangGraph CRAG & Adaptive Multimodal Routing Gateway
### Task 4.1: VLM Service Client (Text-Only Fast Mode & Base64 Multimodal Mode)
- **Reference Docs:** `docs/ARCHITECTURE.md`, `docs/SECURITY.md`
- **Target Files:** `src/services/vlm_service.py`, `configs/prompts.yaml`
- **Implementation Details:**
  - Implement `VLMService` using `AsyncOpenAI` pointed at local vLLM (`http://localhost:8000/v1`).
  - Provide `generate_fast_text(query, chunks)` (text-only messages, zero image overhead) and `generate_visual_reasoning(query, chunks, image_paths)` (attaches up to 2 Base64-encoded resized PNG crops alongside text/table context).
- **Verification:** `mypy src/services/vlm_service.py`

### Task 4.2: LangGraph State Machine (`CRAG Evaluator` + `Adaptive Router`)
- **Reference Docs:** `docs/UX_FLOWS.md` (Flows 2, 3, 4), `docs/ARCHITECTURE.md`
- **Target Files:** `src/graph/state.py`, `src/graph/nodes.py`, `src/graph/workflow.py`, `tests/unit/test_adaptive_router.py`
- **Implementation Details:**
  - Implement nodes: `retrieve_node` -> `crag_evaluate_node` (if max rerank score `< 0.35` and `rewrite_count < 1`, route to `rewrite_query_node`) -> `adaptive_router_edge`:
    - If any chunk in Top-5 has `modality == "figure"` (or `force_visual_route=True`), route to `visual_vlm_generation_node` (`VISUAL_VLM_PATH`).
    - Otherwise, route to `fast_text_table_generation_node` (`FAST_TEXT_TABLE_PATH`).
- **Verification:** `pytest tests/unit/test_adaptive_router.py -v`

---

## Phase 5: FastAPI Backend & Static Crop Asset Server
### Task 5.1: FastAPI Endpoints (`/ingest`, `/query`, `/crops/{filename}`, `/health`)
- **Reference Docs:** `docs/API.md`, `docs/SECURITY.md`
- **Target Files:** `src/api/routes.py`, `main.py`, `tests/integration/test_api.py`
- **Implementation Details:**
  - Implement `/api/v1/ingest`, `/api/v1/query`, `/api/v1/health`, and secure crop serving `/api/v1/crops/{crop_filename}` with strict path traversal protection.
- **Verification:** `pytest tests/integration/test_api.py -v`

---

## Phase 6: Chainlit Visual Attribution Console
### Task 6.1: Interactive Chainlit UI with Inline Cropped Charts & BBox Badges
- **Reference Docs:** `docs/DESIGN_SYSTEM.md`, `docs/UX_FLOWS.md`
- **Target Files:** `app.py`, `public/custom.css`
- **Implementation Details:**
  - Display route badge (`⚡ Fast Text/Table Path` vs. `👁️ Visual VLM Reasoning Path`) + latency breakdown.
  - Automatically attach `cl.Image(path=citation.image_path, name=..., display="inline")` for every retrieved figure/table crop.
  - Render reconstructed Markdown tables and expandable citation cards showing `doc_name`, `page_number`, and `bounding_box` `[x0, y0, x1, y1]`.
- **Verification:** `chainlit run app.py --help`

---

## Phase 7: Three-Way Multimodal Benchmark & Document Noise Robustness Suite
### Task 7.1: Three-Way Ablation Benchmark (`PyPDF` vs. `Docling Text/Table` vs. `OmniDoc-Vision`)
- **Reference Docs:** `docs/TESTING.md`
- **Target Files:** `evals/run_multimodal_benchmark.py`
- **Implementation Details:**
  - Evaluate all 30 questions in `data/golden_multimodal_dataset.json` across the 3 pipeline variants.
  - Compute Table Cell F1-Score, Visual Chart QA Accuracy, Context Recall@5, Faithfulness, and P95 Latency (Fast Path vs. Visual Path). Output Markdown and JSON reports to `evals/results/`.
- **Verification:** `python -m evals.run_multimodal_benchmark --output evals/results/benchmark_report.json`

### Task 7.2: Synthetic Document Degradation & Noise Robustness Suite
- **Reference Docs:** `docs/TESTING.md`, `docs/PRD.md`
- **Target Files:** `evals/noise_robustness.py`, `tests/unit/test_noise_transforms.py`
- **Implementation Details:**
  - Apply scan rotation ($\pm 5^\circ$), Gaussian blur ($\sigma=1.2$), watermark overlay, and 150 DPI re-compression to benchmark pages/crops, measuring Recall@5 and Visual QA accuracy retention.
- **Verification:** `pytest tests/unit/test_noise_transforms.py -v && python -m evals.noise_robustness`

---

## Phase 8: Docker Compose One-Command Init, CI/CD & Portfolio README
### Task 8.1: Docker Compose with Auto-Ingest Init & GitHub Actions CI
- **Reference Docs:** `docs/ARCHITECTURE.md`
- **Target Files:** `docker-compose.yml`, `Dockerfile`, `.github/workflows/ci.yml`, `README.md`
- **Implementation Details:**
  - Configure `qdrant`, `redis`, `langfuse`, and an automatic `init-ingest` step so `docker compose up -d` generates and indexes the benchmark PDFs automatically (Zero manual setup steps / low TTFR).
  - Write a standout `README.md` linking all 3 portfolio repositories (`omnidoc-rag`, `omniops-sre-agent`, `omnidoc-vision`), presenting the architecture diagram, 16GB VRAM breakdown, 3-way ablation table, noise robustness table, and `Limitations & Failure Modes`.
- **Verification:** `ruff check . && mypy src && pytest`
