# System Architecture (ARCHITECTURE.md)

## 1. System Overview
**OmniDoc-Vision** is a modular, production-grade Multimodal Document Intelligence & Visual RAG system that extends the foundation of Project 1 (`OmniDoc RAG`) into complex PDFs, multi-header tables, architecture diagrams, and telemetry charts.

It implements a **Hybrid-Multimodal Architecture**:
1. **Structural Ingestion Layer (`IBM Docling`):** Decouples PDF layout analysis, table reconstruction (`TableFormer`), and figure cropping from query-time inference.
2. **Unified Hybrid Storage (`Approach A`):** Enriches cropped figures with technical VLM descriptions at ingestion time so that text, tables, and visual figures share a single high-speed **Qdrant (`BGE-M3`) + `BM25`** index and **`BGE-Reranker-v2-m3`** cross-encoder.
3. **Adaptive Multimodal Routing Graph (`LangGraph`):** Dynamically routes queries at runtime to either a sub-300ms **Fast Text/Table Path** or a pixel-aware **Visual VLM Path** powered by **`Qwen/Qwen2.5-VL-7B-Instruct-AWQ`** on **vLLM**.

---

## 2. 16GB VRAM Hardware Budget & Allocation Strategy
A critical engineering constraint is running the entire multimodal stack locally on a **single 16GB VRAM GPU** without Out-Of-Memory (OOM) errors:

| Component | Model / Engine | Precision | VRAM Footprint | Allocation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Multimodal LLM / VLM** | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | 4-bit AWQ | **~6.8 GB** | Served via vLLM with `--gpu-memory-utilization 0.60`, `--max-model-len 8192`, `--limit-mm-per-prompt image=2`. Handles BOTH text-only and image+text requests! |
| **Dense Embedder** | `BAAI/bge-m3` | FP16 | **~1.1 GB** | Shared across text, tables, and figure captions. |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | FP16 | **~1.1 GB** | Re-scores top-15 RRF candidates down to top-5. |
| **PDF Layout & Table Engine** | `IBM Docling` (`Layout` + `TableFormer`) | FP16 / CPU | **~1.2 GB** (or 0 GB on CPU) | Runs only during document ingestion; configurable device (`cpu` or `cuda`). |
| **Vector DB & Cache** | `Qdrant` + `Redis` + `BM25` | System RAM | **0.0 GB VRAM** | Runs in Docker system memory. |
| **Total Peak VRAM** | — | — | **~10.2 GB / 16.0 GB** | Leaves **> 5.5 GB free VRAM headroom** for image vision tokens and burst traffic. |

*Note on Model Flexibility:* Just like Projects 1 and 2, the model name is configurable via `.env` (`VLM_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct-AWQ` or lightweight `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` consuming only ~3.2 GB VRAM).

---

## 3. High-Level Architecture Diagram

```
                         ┌────────────────────────────────────┐
                         │      Complex Engineering PDFs      │
                         │ (Multi-Col Text, Tables, Diagrams) │
                         └─────────────────┬──────────────────┘
                                           │
                                           ▼
                         ┌────────────────────────────────────┐
                         │     IBM Docling Parsing Engine     │
                         │ - Layout Detection & Reading Order │
                         │ - TableFormer (Merged-Cell HTML/MD)│
                         │ - High-Res Figure Cropping (.png)  │
                         └────────┬──────────────────┬────────┘
                                  │                  │
                 (Text & Tables)  │                  │ (Cropped Figures + BBox)
                                  │                  ▼
                                  │        ┌──────────────────────────┐
                                  │        │ Approach A Visual Tagger │
                                  │        │ (Qwen2.5-VL Captioning & │
                                  │        │  Axis/Entity Extraction) │
                                  │        └─────────┬────────────────┘
                                  ▼                  ▼
                         ┌────────────────────────────────────┐
                         │   Unified Hybrid Indexer (Qdrant)  │
                         │ - Dense Vectors: BAAI/bge-m3       │
                         │ - Sparse Index: BM25s              │
                         │ - Payload: {modality, bbox, crop}  │
                         └─────────────────┬──────────────────┘
                                           │
      [User Query] ────────────────────────┤
                                           ▼
                         ┌────────────────────────────────────┐
                         │  Parallel Hybrid Search (RRF k=60) │
                         │  + BGE-Reranker-v2-m3 (Top-5)      │
                         └─────────────────┬──────────────────┘
                                           │
                                           ▼
                         ┌────────────────────────────────────┐
                         │   LangGraph CRAG + Adaptive Router │
                         └────────┬──────────────────┬────────┘
                                  │                  │
         (Only Text/Table Chunks) │                  │ (Contains Figure Chunks)
                                  ▼                  ▼
                    ┌──────────────────────┐ ┌────────────────────────┐
                    │ ⚡ Fast Text/Table   │ │ 👁️ Visual VLM Path     │
                    │    Generation        │ │ (Base64 Crop + Prompt) │
                    │  Latency: ~240ms     │ │  Latency: ~920ms       │
                    └──────────┬───────────┘ └───────────┬────────────┘
                               └───────────┬─────────────┘
                                           ▼
                         ┌────────────────────────────────────┐
                         │  Chainlit Visual Grounding Console │
                         │ - Grounded Answer + Table Markdown │
                         │ - Inline Cropped Chart (`cl.Image`)│
                         │ - Page & Bounding Box `[x0,y0...]` │
                         └────────────────────────────────────┘
```

---

## 4. Repository Directory Structure

```text
omnidoc-vision/
├── .agents/
│   └── skills/                             # Local Curated Skills Suite (29 skills)
├── .github/
│   └── workflows/
│       ├── ci.yml                          # Ruff, Mypy, Pytest (Fast unit & integration tests)
│       └── multimodal-evals.yml            # Table F1, Visual QA, and Noise Robustness gates
├── configs/
│   ├── config.yaml                         # Thresholds, DPI settings, routing rules, VRAM limits
│   └── prompts.yaml                        # Visual captioning prompt, Fast-path prompt, VLM prompt
├── data/
│   ├── raw_pdfs/                           # Generated & downloaded technical benchmark PDFs
│   ├── crops/                              # Extracted high-res PNG crops of charts & tables
│   ├── bm25_index/                         # Persisted sparse index artifacts
│   └── golden_multimodal_dataset.json      # 30-item benchmark dataset (Text, Table, Chart QA)
├── AGENTS.md                               # Master AI Agent Guidelines & Skills Routing
├── CLAUDE.md                               # Claude Agent Instructions
├── GEMINI.md                               # Gemini Agent Instructions
├── docs/                                   # Complete 10-File Engineering Specification
│   ├── PRD.md
│   ├── DESIGN_SYSTEM.md
│   ├── UX_FLOWS.md
│   ├── ARCHITECTURE.md
│   ├── DATABASE.md
│   ├── API.md
│   ├── SECURITY.md
│   ├── CODE_STYLE.md
│   ├── TESTING.md
│   └── PLAN.md
├── scripts/
│   ├── build_benchmark_pdfs.py             # Thin deterministic corpus-generation entrypoint
│   └── benchmark/                          # Shared content, layout, charts, documents, dataset
├── evals/
│   ├── __init__.py
│   ├── run_multimodal_benchmark.py         # Compares PyPDF vs Docling-Text vs OmniDoc-Vision
│   ├── noise_robustness.py                 # Tests skew, blur, and low-DPI degradation resilience
│   └── results/                            # Saved Markdown & JSON benchmark tables
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── config.py                       # Pydantic V2 BaseSettings
│   │   ├── exceptions.py                   # Custom multimodal & parser exceptions
│   │   └── logger.py                       # Structured JSON logger
│   ├── schemas/
│   │   ├── document.py                     # BoundingBox, MultimodalChunk, ChunkModality enums
│   │   └── api.py                          # Request/Response & VisualCitation schemas
│   ├── ingestion/
│   │   ├── docling_parser.py               # IBM Docling wrapper (text, tables, figure cropping)
│   │   ├── visual_captioner.py             # Approach A VLM figure summarizer + offline fallback
│   │   └── indexer.py                      # Unified Qdrant (BGE-M3) + BM25s hybrid indexer
│   ├── retrieval/
│   │   ├── dense.py                        # Qdrant async vector search
│   │   ├── sparse.py                       # BM25 sparse keyword search
│   │   ├── fusion.py                       # Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py                     # Cross-Encoder BGE-Reranker-v2-m3
│   ├── graph/
│   │   ├── state.py                        # VisionGraphState TypedDict
│   │   ├── nodes.py                        # Retrieve, CRAG Grade, Rewrite, Route, FastGen, VisualGen
│   │   └── workflow.py                     # Compiled LangGraph state machine
│   ├── services/
│   │   ├── vlm_service.py                  # OpenAI-compatible vLLM client (Text + Base64 Image)
│   │   └── cache_service.py                # Redis semantic & exact response cache
│   └── api/
│       └── routes.py                       # FastAPI endpoints (/ingest, /query, /crops/{filename})
├── tests/
│   ├── unit/
│   │   ├── test_benchmark_generator.py     # Benchmark corpus & dataset contract tests
│   │   ├── test_config.py                  # Settings & environment validation tests
│   │   ├── test_docling_parser.py          # Table & BBox extraction unit tests
│   │   ├── test_indexer.py                 # Qdrant + BM25s hybrid indexer unit tests
│   │   ├── test_logger.py                  # Structured JSON logging & redaction tests
│   │   ├── test_packaging.py               # Manifest & dependency alignment guards
│   │   ├── test_schemas.py                 # MultimodalChunk & API schema contract tests
│   │   ├── test_visual_captioner.py        # Approach A VLM captioner & fallback tests
│   │   ├── test_adaptive_router.py         # Fast Path vs Visual Path routing logic tests
│   │   └── test_noise_transforms.py        # Synthetic PDF/image noise generator tests
│   └── integration/
│       ├── test_multimodal_pipeline.py     # End-to-end ingestion -> retrieval -> generation test
│       └── test_api.py                     # FastAPI endpoint tests
├── app.py                                  # Chainlit Visual Attribution UI
├── main.py                                 # FastAPI entrypoint
├── docker-compose.yml                      # Qdrant, Redis, Langfuse, App containers
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

`.github/workflows`, Docker artifacts, the API, and the Chainlit UI are delivered in later phases. Phase 1 includes the README, buildable package metadata, benchmark generator, and tests.

---

## 5. Architecture Decision Records (ADRs)

### ADR-001: Hybrid Structural + Visual Architecture (Docling + Qwen2.5-VL) Over Pure ColPali
- **Context:** Pure visual retrieval models (like ColPali) index every page as thousands of vision patch vectors, consuming high VRAM and struggling with exact keyword/error-code matching compared to BM25. Conversely, pure text parsers fail on charts.
- **Decision:** Combine **IBM Docling** (for exact text & lossless table reconstruction) with **Approach A Visual Captioning** and **Adaptive Qwen2.5-VL Routing**.
- **Consequence:** Preserves 100% of Project 1's sub-300ms hybrid search speed on text and tables while unlocking pixel-level VLM chart reasoning only when visual chunks are retrieved.

### ADR-002: Approach A (VLM Ingestion Captioning + BGE-M3) Over Dual Vision-Text Vector Spaces
- **Context:** Maintaining two separate embedding models in VRAM (`BGE-M3` for text + `SigLIP/CLIP` for images) complicates RRF score normalization and increases VRAM overhead.
- **Decision:** Generate rich, structured technical descriptions of cropped figures during ingestion and embed them alongside text and tables using `BAAI/bge-m3` + `BM25`, while storing the high-res PNG path in the Qdrant payload for query-time `Qwen2.5-VL` inspection.
- **Consequence:** Zero extra embedding VRAM at query time, seamless RRF fusion across modalities, and full visual fidelity during generation.
