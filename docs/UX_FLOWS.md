# User Experience & Multimodal Execution Flows (UX_FLOWS.md)

## Overview
This document traces the core user and system execution flows in **OmniDoc-Vision**, illustrating how documents are parsed by `Docling`, indexed via Approach A, and dynamically routed between the **Fast Text/Table Path** and the **Visual VLM Path**.

---

## Flow 1 — Multimodal Document Ingestion & Approach A Indexing
*An engineer ingests a technical PDF containing text sections, merged-cell benchmark tables, and P99 latency charts.*

```
[Input PDF]: `distributed_storage_benchmark.pdf`
       │
       ▼
[Step 1: IBM Docling Layout & Structure Engine (`src/ingestion/docling_parser.py`)]
       ├── Detects Reading-Order Text Blocks -> Emits `MultimodalChunk(modality="text", bbox=[...])`
       ├── Detects Complex Tables (`TableFormer`) -> Reconstructs lossless Markdown + HTML table
       │     └── Emits `MultimodalChunk(modality="table", bbox=[...], table_html="...", content="|...|")`
       └── Detects Figures & Diagrams -> Crops high-res PNG (`data/crops/dist_p3_fig1.png`)
             │
             ▼
[Step 2: Approach A Visual Captioning (`src/ingestion/visual_captioner.py`)]
       └── Sends cropped PNG to `Qwen2.5-VL-7B-Instruct-AWQ` (or deterministic metadata extractor in fast/offline mode):
           Extracts: Title, X/Y axes, units, numeric peaks, and architecture nodes into structured text `content`.
       │
       ▼
[Step 3: Unified Hybrid Indexing (`src/ingestion/indexer.py`)]
       ├── Generates Dense Embeddings via `BAAI/bge-m3` -> Upserts into Qdrant (`omnidoc_vision_chunks`)
       └── Updates Sparse `BM25` Inverted Index with bilingual tokenization
```

---

## Flow 2 — Query Flow A: Fast Text & Reconstructed Table Path (`< 300ms`)
*The user asks a specific numeric question answered by a complex table on Page 2.*

```
[User Query]: "در جدول مقایسه دیتابیس‌ها، توان عملیاتی Qdrant در حالت کوانتیزاسیون Scalar چقدر است و چقدر رم مصرف می‌کند؟"
       │
       ▼
[Hybrid Retrieval (`BGE-M3` Top-20 + `BM25` Top-20 -> `RRF` -> `BGE-Reranker-v2-m3` Top-5)]
       │
       ▼
[CRAG Evaluator Node (`src/graph/nodes.py`)]
       └── Top chunk score = `0.93` (>= `0.35` threshold). Modality of top chunks: `["table", "text"]` (Zero `figure` chunks).
       │
       ▼
[Adaptive Routing Gateway selects: ⚡ FAST_TEXT_TABLE_PATH]
       ├── No Base64 images loaded into prompt (saves VRAM KV-cache & ~700ms vision encoding latency).
       └── Passes reconstructed Markdown table + text context to `Qwen2.5-VL` text completion endpoint.
       │
       ▼
[Chainlit UI Response]:
       ├── Badge: `[⚡ Route: Fast Text/Table Path | 234ms | Score: 0.93]`
       ├── Grounded Persian Answer with exact numbers from the table
       └── Expandable Citation Card showing the reconstructed Markdown table + `[Page 2 | BBox: 0.14, 0.22, 0.85, 0.49]`
```

---

## Flow 3 — Query Flow B: Visual VLM Reasoning Path (`~800–1200ms`)
*The user asks a question about a visual trend or network diagram that requires looking at the image.*

```
[User Query]: "بر اساس نمودار تاخیر P99 در صفحه ۳، از چه تعداد درخواست هم‌زمان (Concurrency) شیب خطای سیستم تند می‌شود؟"
       │
       ▼
[Hybrid Retrieval + Cross-Encoder Reranker]
       └── Retrieves `chunk_id="cloud_p03_fig01"` (`modality="figure"`, `image_path="data/crops/cloud_p03_fig01.png"`, score=`0.91`).
       │
       ▼
[Adaptive Routing Gateway detects `modality == "figure"` -> Selects: 👁️ VISUAL_VLM_PATH]
       ├── Reads `data/crops/cloud_p03_fig01.png`, validates dimensions, encodes as Base64 data URI.
       └── Constructs OpenAI-compatible Multimodal Chat Payload (`image_url` + grounded text context).
       │
       ▼
[`Qwen2.5-VL-7B-Instruct-AWQ` inspects the chart pixels + caption metadata]
       │
       ▼
[Chainlit UI Response]:
       ├── Badge: `[👁️ Route: Visual VLM Path | 915ms | 1 Chart Inspected]`
       ├── Detailed Persian explanation pinpointing the inflection point at 250 concurrent connections
       └── Inline `cl.Image` rendering `dist_p4_fig1.png` + Attribution Footer (`Page 4 | BBox: [0.19, 0.285, 0.882, 0.71]`)
```

---

## Flow 4 — CRAG Self-Correction on Ambiguous Visual Query
*The user uses vague terminology; initial retrieval scores below `0.35`, triggering query rewrite.*

```
[User Query]: "اون عکس شبکه که سرورها به هم وصلن چی میگه؟"
       │
       ▼
[Initial Retrieval Max Rerank Score = `0.24` (< `0.35` threshold)]
       │
       ▼
[CRAG Query Rewriter Node]:
       └── Rewrites query to: `"network topology architecture diagram server cluster connection flow"`
       │
       ▼
[Retry Retrieval finds `arch_p1_fig1` (`score = 0.86`)] -> Routes to `VISUAL_VLM_PATH` -> Renders diagram & answer.
```
