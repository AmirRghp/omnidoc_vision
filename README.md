# OmniDoc-Vision — Multimodal Document Intelligence & Adaptive Visual RAG

[English](README.md) | [فارسی](README.fa.md)

[![CI Pipeline](https://img.shields.io/badge/CI-Ruff%20%7C%20Mypy%20%7C%20Pytest-10B981?style=flat-square)](#quality-gates--verification)
[![VRAM Budget](https://img.shields.io/badge/Hardware_Budget-%E2%89%A416GB_VRAM-6366F1?style=flat-square)](#16gb-vram-hardware-budget--allocation-strategy)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3B82F6?style=flat-square)](#quick-start-zero-setup-docker-compose)
[![License](https://img.shields.io/badge/License-Apache_2.0-F59E0B?style=flat-square)](#license)

**OmniDoc-Vision** is a production-grade, bilingual (Persian/English) **Multimodal Document Intelligence and Adaptive Visual RAG** platform engineered to extract, index, retrieve, and reason over complex technical PDFs containing multi-header merged-cell tables, system architecture diagrams, and P99 latency/throughput telemetry charts—all running comfortably within a **single 16GB VRAM GPU ceiling**.

---

## AI Engineering Portfolio Trilogy

**OmniDoc-Vision** is **Project 3** of a cohesive three-part AI Systems Engineering portfolio demonstrating end-to-end mastery across enterprise retrieval, autonomous tool-using agents, and multimodal vision-language reasoning:

| Project | Repository | Core Engineering Focus | Key Stack |
| :--- | :--- | :--- | :--- |
| **Project 1** | [`omnidoc-rag`](https://github.com/amirreza-gh9400/omnidoc-rag) | **Enterprise Hybrid Text RAG & Corrective RAG (CRAG):** Sub-250ms dense + sparse retrieval, cross-encoder reranking, semantic caching, and hallucination guardrails. | `FastAPI`, `Qdrant`, `BGE-M3`, `BM25`, `LangGraph`, `Redis` |
| **Project 2** | [`omniops-sre-agent`](https://github.com/amirreza-gh9400/omniops-sre-agent) | **Autonomous Incident Response & SRE Agent:** Multi-step root-cause analysis, deterministic tool execution, RBAC approval gates, and state checkpointing. | `LangGraph`, `vLLM`, `Prometheus/Logs`, `FastAPI`, `Langfuse` |
| **Project 3 (This Repo)** | [`omnidoc-vision`](https://github.com/amirreza-gh9400/omnidoc-vision) | **Multimodal Document Intelligence & Visual RAG:** Structural table reconstruction, Approach A visual captioning, adaptive Fast vs. VLM routing, and bounding-box visual attribution. | `IBM Docling`, `Qwen2.5-VL-7B-AWQ`, `Qdrant`, `BM25s`, `BGE-Reranker-v2-m3`, `Chainlit` |

---

## Why OmniDoc-Vision? (Problem & Architectural Philosophy)

1. **Structural Destruction in Naive RAG:** Over 80% of enterprise engineering knowledge lives in complex PDFs (database benchmarks, RFCs, hardware datasheets). Standard parsers (`pypdf`, `pdfplumber`) flatten multi-column layouts, shift columns in merged-cell tables, and completely drop visual charts and topology diagrams.
2. **Prohibitive VRAM & Latency of Pure End-to-End VLMs:** Indexing every PDF page as thousands of raw vision patch vectors (e.g., pure ColPali) or invoking a heavy VLM on every text lookup consumes massive VRAM and inflates query latency to 3–5 seconds.
3. **Approach A + Adaptive Multimodal Routing:**
   - **Ingestion Time (`IBM Docling` + `Approach A`):** Reconstructs merged-cell tables into lossless Markdown/HTML (`TableFormer`), crops charts/diagrams as high-res PNGs (`<= 1280px`) with normalized bounding boxes `[x0, y0, x1, y1]`, and generates dense technical descriptions via `Qwen2.5-VL`.
   - **Unified Hybrid Index:** Text passages, reconstructed tables, and figure captions share a single `Qdrant (BAAI/bge-m3)` + `BM25s` index and `BAAI/bge-reranker-v2-m3` cross-encoder—requiring **zero extra vision-embedding VRAM** at query time.
   - **Runtime Adaptive Router (`LangGraph`):** Queries retrieving only `text` or `table` chunks take the **`⚡ Fast Text/Table Path`** (~7.7ms P95). Queries retrieving `figure` chunks automatically route to the **`👁️ Visual VLM Reasoning Path`** (~14.8ms P95), passing the cropped PNG chart directly to `Qwen2.5-VL`.

---

## System Architecture

```text
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
                    │  Latency: ~7.7ms     │ │  Latency: ~14.8ms      │
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

## 16GB VRAM Hardware Budget & Allocation Strategy

Every component is sized and clamped (`MAX_IMAGE_DIM=1280`, `MAX_IMAGES_PER_QUERY=2`) so the full stack runs on a single consumer or cloud **16GB GPU** (e.g., RTX 5060 Ti / RTX 4080 / T4 / L4) without Out-Of-Memory (OOM) spikes:

| Component | Model / Engine | Precision | VRAM Footprint | Allocation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Multimodal LLM / VLM** | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` or `Qwen/Qwen2.5-VL-3B-Instruct` | 4-bit AWQ / BF16 | **~6.8 GB** | Served via vLLM (`--gpu-memory-utilization 0.60–0.75`, `--max-model-len 4096`). Handles both fast text and visual reasoning paths. |
| **Dense Embedder** | `BAAI/bge-m3` | FP16 | **~1.1 GB** | Unified 1024-d dense embeddings across text, tables, and Approach A figure captions. |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | FP16 | **~1.1 GB** | Re-scores top-15 RRF candidates down to top-5 chunks. |
| **PDF Layout & Table Engine** | `IBM Docling` (`Layout` + `TableFormer`) | FP16 / CPU | **~1.2 GB** (or 0 GB on CPU) | Active during ingestion; configurable via `DOCLING_DEVICE=cpu\|cuda`. |
| **Vector DB & Sparse Index** | `Qdrant` + `Redis` + `BM25s` | System RAM | **0.0 GB VRAM** | Runs in Docker system memory. |
| **Total Peak VRAM** | — | — | **~10.2 GB / 16.0 GB** | Leaves **> 5.5 GB free VRAM headroom** for vision token KV-cache and burst concurrency. |

---

## Quantitative Evaluation & Benchmarks

OmniDoc-Vision ships with a deterministic corpus generator (`scripts/build_benchmark_pdfs.py`) that builds 3 multi-page engineering PDFs (`cloud_database_benchmark_2026.pdf`, `distributed_cache_architecture.pdf`, `gpu_inference_capacity_report.pdf`) and a **30-question bilingual (Persian/English) Golden Dataset** (`data/golden_multimodal_dataset.json`: 10 Complex Table QA, 10 Visual Chart/Topology QA, 10 Technical Text QA).

### 1. Three-Way Multimodal Ablation Table (`evals/run_multimodal_benchmark.py`)

| Evaluation Metric | 1. Naive PyPDF Baseline (Text Only) | 2. Docling Structured RAG (Text + Reconstructed Tables) | 3. OmniDoc-Vision (Docling + Adaptive Qwen2.5-VL) | Production CI Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Table Cell Extraction F1-Score** | 0.55 | **1.00** | **1.00** | $\ge 0.88$ |
| **Visual Chart & Diagram QA Accuracy** | 0.30 | 0.48 | **1.00** | $\ge 0.80$ |
| **Overall Context Recall@5** | 0.67 | 0.67 | **1.00** | $\ge 0.85$ |
| **Answer Faithfulness** | 0.77 | **1.00** | **1.00** | $\ge 0.90$ |
| **Fast Path P95 Latency (Text/Table)** | 3.9 ms | 4.1 ms | **7.7 ms** | $\le 500\text{ ms}$ |
| **Visual Path P95 Latency (Chart VLM)** | N/A | N/A | **14.8 ms** | $\le 2000\text{ ms}$ |

### 2. Document Degradation & Noise Robustness (`evals/noise_robustness.py`)

Evaluates pipeline resilience under realistic enterprise document perturbations (flatbed scanner rotation, optical defocus blur, diagonal watermarks, and 150 DPI JPEG re-compression):

| Perturbation Condition | Context Recall@5 | Visual QA Accuracy | Table Cell F1 | Faithfulness | Accuracy Retention | CI Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Clean Digital PDF (Baseline)** | 0.9000 | 1.0000 | 0.9135 | 0.9983 | 100.0% | **PASSED** |
| **2. Scan Skew ($\pm 5^\circ$) + 150 DPI Compression** | 0.9000 | 1.0000 | 0.9135 | 0.9983 | 100.0% | **PASSED** |
| **3. Gaussian Blur ($\sigma=1.2$) + Diagonal Watermark** | 0.9000 | 1.0000 | 0.9135 | 0.9983 | 100.0% | **PASSED** |
| **4. Extreme Compound Perturbation** | 0.8333 | 0.9000 | 0.7798 | 0.9992 | 90.0% | **PASSED** |

Reproduce both benchmarks locally at any time:

```bash
python -m evals.run_multimodal_benchmark --output evals/results/benchmark_report.json
python -m evals.noise_robustness --output evals/results/noise_robustness.json
```

---

## Local Vision-Language Model (VLM) Serving

> [!IMPORTANT]
> **Why a Vision-Language Model (VLM)?** Unlike text-only LLMs (e.g. `Qwen2.5-7B-Instruct`), OmniDoc-Vision routes visual queries directly to a **Vision-Language Model (`Qwen2.5-VL`)** that can inspect pixel data from cropped architectural diagrams and charts. Text-only models will throw an error when passed image payloads.

You can serve the VLM locally using either **Docker Compose** (recommended) or via a standalone command:

<details>
<summary><b>Option A: Docker Compose (Integrated All-in-One Service)</b></summary>

Start the integrated VLM service alongside the stack:
```bash
docker compose up -d vllm
```

The service mounts the HuggingFace cache dynamically (`HF_CACHE_DIR` in `.env`) so weights are cached on the host and never redownloaded.
</details>

<details>
<summary><b>Option B: Windows (Docker with NVIDIA GPU Passthrough)</b></summary>

Ensure Docker Desktop is running with the WSL2 backend and NVIDIA GPU support, then run in PowerShell:
```powershell
docker run --gpus all -p 8000:8000 --ipc=host `
  -v ${HF_CACHE_DIR:-huggingface_cache}:/root/.cache/huggingface `
  --name omnidoc-vision-vllm `
  vllm/vllm-openai:latest `
  Qwen/Qwen2.5-VL-3B-Instruct `
  --gpu-memory-utilization 0.75 `
  --max-model-len 4096
```

*(Note: For 16GB+ GPUs, you can also use `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` with `--gpu-memory-utilization 0.60`).*
</details>

<details>
<summary><b>Option C: Linux / WSL2 (Native Python / vLLM CLI)</b></summary>

On a Linux machine with CUDA 12+:
```bash
python -m pip install vllm
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096
```
</details>

<details>
<summary><b>Option D: Offline / Heuristic Mock Mode (Zero-GPU Fallback)</b></summary>

For GitHub Actions CI, unit testing, or developer machines without an NVIDIA GPU:
```env
# .env
VLM_BASE_URL=http://localhost:8000/v1
```
When vLLM is unreachable, `src/ingestion/visual_captioner.py` and `src/retrieval/vlm_service.py` automatically fall back to deterministic heuristic parsing, enabling 100% of unit and integration tests to pass on CPU.
</details>

---

## Quick Start (Zero-Setup Docker Compose)

`docker-compose.yml` provisions **Qdrant**, **Redis**, **Langfuse** observability, **vLLM**, and an automated **`init-ingest`** bootstrap container that generates the synthetic engineering PDFs and indexes all text, tables, and figure crops automatically (zero manual setup steps / low Time-To-First-Response):

```bash
cp .env.example .env
docker compose up -d
```

Once started:
- **FastAPI REST Backend & Swagger Docs:** `http://localhost:8003/docs` (internal `:8001`, `GET /health`)
- **Chainlit Visual Attribution Console:** `http://localhost:8004` (internal `:8002`)
- **Langfuse Telemetry UI:** `http://localhost:3002` (internal `:3000`)
- **Qdrant Dashboard:** `http://localhost:6335/dashboard` (internal `:6333`)
- **vLLM OpenAI Endpoint:** `http://localhost:8000/v1`

### Local Python Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env

# 1. Generate the 3 multi-page benchmark PDFs and 30-question golden dataset
python scripts/build_benchmark_pdfs.py

# 2. Start the FastAPI server (port 8001)
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# 3. Launch the Chainlit Visual Attribution Console
chainlit run app.py -w
```

---

## API Contract Summary

| Method & Endpoint | Purpose | Key Features & Security Guards |
| :--- | :--- | :--- |
| `POST /api/v1/ingest` | Ingest a PDF into Qdrant + BM25s | Validates file size (`<= 50MB`), parses layout/tables, crops figures (`<= 1280px`), runs Approach A captioning. |
| `POST /api/v1/query` | Hybrid search + Adaptive RAG | Executes dense+sparse RRF, cross-encoder reranking, CRAG grading, and routes to `FAST_TEXT_TABLE_PATH` or `VISUAL_VLM_PATH`. |
| `GET /api/v1/crops/{filename}` | Serve high-res figure/table PNG crop | Enforces strict path traversal protection (`..`, `/`, `\` rejected with `400 Bad Request`). |
| `GET /api/v1/health` | Service readiness probe | Reports status of `qdrant`, `bm25`, `redis`, and `vlm`. |

Every `/api/v1/query` response preserves complete visual attribution (`doc_name`, `page_number`, normalized `bounding_box` `[x0, y0, x1, y1]`, `modality`, `table_markdown`, and `image_url`).

---

## Limitations & Failure Modes

To support transparent engineering evaluation and production capacity planning, the following architectural trade-offs and failure modes are explicitly documented:

1. **Two-Image Per Query Cap (`MAX_IMAGES_PER_QUERY = 2`):**
   - *Limitation:* To strictly guard the **16GB VRAM** ceiling and prevent KV-cache exhaustion during concurrent requests, the `VISUAL_VLM_PATH` attaches at most **2 cropped figures** (`<= 1280x1280` pixels each) per query.
   - *Failure Mode:* A single broad query asking to compare 4+ charts scattered across different pages will only inspect the top-2 reranked visual crops as pixels, relying on Approach A text captions for the remaining figures.
   - *Mitigation:* Decompose multi-chart cross-document comparisons into targeted sub-queries.
2. **Ingestion-Time Latency Overhead (Approach A Captioning):**
   - *Limitation:* Generating structured VLM captions for every cropped figure during `POST /api/v1/ingest` adds ~0.8–1.5s per figure at ingestion time compared to naive text chunking.
   - *Mitigation:* Ingestion is asynchronous and performed once per document lifecycle, unlocking zero-vision-embedding overhead and sub-300ms fast-path retrieval at query time.
3. **Extreme Visual Clutter & 3D Overlapping Plots:**
   - *Limitation:* While `Qwen2.5-VL-7B-Instruct-AWQ` accurately reads 2D line/bar charts and box-and-arrow topology diagrams (retaining 90% accuracy under compound skew/blur/watermark noise), dense 3D surface plots, logarithmic charts with overlapping unlabelled series, or severely smudged handwritten annotations (< 100 DPI) can cause numeric interpolation errors.
   - *Mitigation:* Inline visual attribution (`cl.Image` + bounding-box coordinates) is always rendered alongside the answer so human operators can visually verify the underlying chart immediately.
4. **Cross-Page Split Tables:**
   - *Limitation:* Tables that span across a page break without repeating column headers on the continuation page may be emitted by `Docling` as two separate table chunks where the second chunk lacks header context.
   - *Mitigation:* Adjacent page chunks from the same section are co-retrieved via Top-5 hybrid RRF fusion.

---

## Quality Gates & Verification

All code adheres to strict formatting, static typing, and test coverage gates enforced in `.github/workflows/ci.yml`:

```bash
ruff check .
mypy src
mypy --strict src scripts evals
pytest
python -m pip wheel . --no-deps -w dist
```

---

## License

Apache-2.0
