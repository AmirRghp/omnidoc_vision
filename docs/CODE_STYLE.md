# Code Style & Engineering Standards (CODE_STYLE.md)

## 1. Core Engineering Principles
- **Clean Architecture & Modality Agnosticism:** Retrieval and fusion modules (`src/retrieval/`) operate on unified `MultimodalChunk` objects without hardcoding PDF parsing internals.
- **Strict Type Annotations:** 100% type hints (`mypy --strict` compliant). Use `Literal["text", "table", "figure"]` or `StrEnum` for modality types.
- **Non-Blocking Async Execution:** CPU/GPU-bound tasks (`Docling` parsing, `BGE-M3` encoding, `CrossEncoder` reranking, and PIL image resizing) must be wrapped in `asyncio.to_thread(...)` when called inside FastAPI or Chainlit async event loops.
- **Pydantic V2 Data Contracts:** All bounding boxes (`BoundingBox`), chunks (`MultimodalChunk`), and API payloads use strict Pydantic V2 validators (e.g., ensuring `x1 > x0` and `y1 > y0`).

---

## 2. Naming & Formatting Conventions
- **Linter & Formatter:** `ruff check .` and `ruff format --check .` (Line length: 100, Python 3.11+).
- **Type Checker:** `mypy src`
- **Logging:** Structured JSON logging via `src/common/logger.py` logging `query_id`, `routing_path` (`FAST_TEXT_TABLE_PATH` vs `VISUAL_VLM_PATH`), `vlm_images_count`, and per-stage latency.
