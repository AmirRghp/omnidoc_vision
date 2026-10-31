# Task 3.1 — Parallel Dense + Sparse Search, RRF Fusion & BGE Reranker

پیاده‌سازی لایه بازیابی ترکیبی چندوجهی (`src/retrieval/`) شامل جستجوی موازی برداری در Qdrant (`BAAI/bge-m3` Top-20) و جستجوی واژگانی در `BM25s` (Top-20) با `asyncio.gather`، ترکیب نتایج با الگوریتم Reciprocal Rank Fusion ($k=60$)، و بازرتبه‌بندی ۱۵ کاندیدای برتر به Top-5 توسط مدل Cross-Encoder (`BAAI/bge-reranker-v2-m3` به‌همراه Fallback قطعی برای محیط‌های تست/CI).

## User Review & Critical Decisions

> [!IMPORTANT]
> بر اساس مشخصات `docs/PLAN.md`، `docs/ARCHITECTURE.md`، `docs/DATABASE.md` و `docs/UX_FLOWS.md`، نیازمندی‌های تسک ۳.۱ کاملاً شفاف هستند:

- **سطح پیچیدگی تسک:** **Medium** — شامل ۴ ماژول متمرکز در `src/retrieval/` (`dense.py`، `sparse.py`، `fusion.py`، `reranker.py`) و تست‌های واحد کامل در `tests/unit/test_retrieval.py`.
- **اجرای موازی غیرمسدودکننده (`asyncio.gather` + `asyncio.to_thread`):** اجرای هم‌زمان جستجوی Dense و Sparse با `asyncio.gather` و انتقال عملیات محاسباتی CPU/GPU (توکنایزینگ `BM25s` و استنتاج `CrossEncoder`) به `asyncio.to_thread`.
- **نرمال‌سازی امتیاز بازرتبه‌بندی (`rerank_score` در بازه `[0.0, 1.0]`):** امتیازهای خروجی مدل `BAAI/bge-reranker-v2-m3` (و `MockReranker` در حالت تست/CI) در بازه `[0.0, 1.0]` نرمال‌سازی می‌شوند تا با قرارداد `VisualCitation.rerank_score` در `src/schemas/api.py` و آستانه `0.35` در گره `CRAG Evaluator` (تسک ۴.۲) کاملاً سازگار باشند.

---

## 1. Overview & Core Concept

- **What It Does:** پرسش کاربر (فارسی یا انگلیسی) و فیلتر اختیاری مدالیته (`modality_filter: ChunkModality | str | None`) را دریافت کرده، Top-20 چانک از Qdrant و Top-20 چانک از ایندکس `BM25s` را به‌طور موازی بازیابی می‌کند، با فرمول $\text{RRF}(d) = \sum \frac{1}{k + \text{rank}(d)}$ ($k=60$) ادغام می‌نماید، ۱۵ کاندیدای اول را با `BGE-Reranker-v2-m3` بازرتبه‌بندی کرده و Top-K (پیش‌فرض ۵) چانک برتر را به‌همراه امتیاز و زمان‌بندی (`retrieval_ms`, `rerank_ms`) برمی‌گرداند.
- **Target Audience / Downstream Consumers:** ماشین وضعیت LangGraph (`retrieve_node` و `crag_evaluate_node` در تسک ۴.۲)، اندپوینت `/api/v1/query` (تسک ۵.۱)، و بنچمارک سهلایه (تسک ۷.۱).
- **Key Value:** ترکیب دقت معنایی `BGE-M3` با تطابق دقیق واژگان/اعداد فنی در `BM25s` بدون از دست رفتن متادیتای بصری (`bounding_box`, `image_path`, `table_html`).

---

## 2. User Experience & Visual Design (Retrieval & Attribution Flow)

- **Key Execution Flow:**
  1. دریافت پرسش `query`، تعداد نتایج نهایی `top_k=5`، و فیلتر اختیاری `modality_filter` (`"text" | "table" | "figure"`).
  2. اجرای موازی `DenseRetriever.search(query, top_k=20, modality_filter=...)` و `SparseRetriever.search(query, top_k=20, modality_filter=...)` از طریق `asyncio.gather`.
  3. ادغام لیست‌های رتبه‌بندی‌شده توسط `reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=15)` با حفظ کامل شیء `MultimodalChunk`.
  4. بازرتبه‌بندی ۱۵ کاندیدای برتر توسط `BGEReranker.rerank(query, candidates, top_k=5)` و بازگرداندن لیست نهایی `ScoredChunk` به‌ترتیب نزولی `rerank_score`.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: مدل داده یکپارچه برای کاندیداهای بازیابی (`ScoredChunk`)**
  - *Chosen Approach:* تعریف مدل تغییرناپذیر `ScoredChunk(BaseModel)` شامل `chunk: MultimodalChunk`، `score: float` (امتیاز فعال در هر مرحله)، `dense_score: float | None`، `sparse_score: float | None`، `rrf_score: float | None` و `rerank_score: float | None`.
  - *Why:* حفظ کامل شیء استاندارد `MultimodalChunk` و قابلیت ردیابی امتیاز هر مرحله برای CRAG، لاگینگ و `VisualCitation`.
- **Decision 2: استفاده مجدد از `BaseEmbedder` و `tokenize_bilingual` از لایه `src/ingestion/indexer.py`**
  - *Chosen Approach:* `DenseRetriever` از همان رابط `BaseEmbedder` (`BGEM3Embedder` / `MockEmbedder`) و `SparseRetriever` از همان تابع `tokenize_bilingual` و فایل `corpus_metadata.json` ایجادشده در `HybridIndexer` استفاده می‌کنند.
  - *Why:* جلوگیری از تکرار کد و تضمین سازگاری ۱۰۰٪ بین توکنایزینگ/امبدینگ زمان ایندکس و زمان پرسش.
- **Decision 3: معماری `BaseReranker` با `BGEReranker` و `MockReranker` قطعی**
  - *Chosen Approach:* در محیط تولید از `FlagEmbedding.FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=...)` (با نگاشت سیگموئید به `[0.0, 1.0]`) استفاده می‌شود و در محیط تست/CI یا عدم نصب `FlagEmbedding`، به‌طور خودکار به `MockReranker` مبتنی بر هم‌پوشانی توکن‌های دوزبانه (`normalize_bilingual_text`) و امتیاز RRF سوییچ می‌گردد.
  - *Why:* اجرای سریع و کاملاً قطعی تمام تست‌های واحد و یکپارچه‌سازی روی CPU بدون وابستگی به وزن‌های سنگین GPU (`AGENTS.md` Rule #2).

---

## 4. Technical Architecture & Data Strategy

### Architecture & Component Diagram

```
                         ┌──────────────────────────────┐
                         │  User Query + ModalityFilter │
                         └──────────────┬───────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       │         asyncio.gather          │
                       ▼                                 ▼
        ┌─────────────────────────────┐   ┌─────────────────────────────┐
        │   DenseRetriever (dense.py) │   │  SparseRetriever (sparse.py)│
        │ - Embed query via BGE-M3    │   │ - tokenize_bilingual(query) │
        │ - Qdrant query_points Top20 │   │ - BM25s retrieve Top20      │
        │ - Payload modality filter   │   │ - Metadata modality filter  │
        └──────────────┬──────────────┘   └──────────────┬──────────────┘
                       └────────────────┬────────────────┘
                                        ▼
                       ┌─────────────────────────────────┐
                       │    RRF Fusion (fusion.py)       │
                       │ - k = 60                        │
                       │ - Deduplicate by chunk_id       │
                       │ - Select Top-15 candidates      │
                       └────────────────┬────────────────┘
                                        ▼
                       ┌─────────────────────────────────┐
                       │   BGEReranker (reranker.py)     │
                       │ - BAAI/bge-reranker-v2-m3       │
                       │   (or deterministic MockRerank) │
                       │ - Sigmoid score in [0.0, 1.0]   │
                       │ - Return Top-5 ScoredChunk      │
                       └─────────────────────────────────┘
```

### Target Files & Signatures

1. **`src/retrieval/dense.py`:**
   - `ScoredChunk(BaseModel)`: نگه‌دارنده `chunk: MultimodalChunk` و امتیازهای مراحل بازیابی.
   - `DenseRetriever`:
     - `__init__(self, settings: Settings | None = None, qdrant_client: AsyncQdrantClient | None = None, embedder: BaseEmbedder | None = None) -> None`
     - `async def search(self, query: str, *, top_k: int = 20, modality_filter: ChunkModality | str | None = None, doc_name: str | None = None) -> list[ScoredChunk]`
2. **`src/retrieval/sparse.py`:**
   - `SparseRetriever`:
     - `__init__(self, settings: Settings | None = None, *, bm25_dir: Path | None = None) -> None`
     - `async def search(self, query: str, *, top_k: int = 20, modality_filter: ChunkModality | str | None = None, doc_name: str | None = None) -> list[ScoredChunk]`
3. **`src/retrieval/fusion.py`:**
   - `RRF_K_DEFAULT: Final[int] = 60`
   - `def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[ScoredChunk]], *, k: int = 60, top_n: int = 15) -> list[ScoredChunk]`
4. **`src/retrieval/reranker.py`:**
   - `BaseReranker(ABC)`، `MockReranker(BaseReranker)`، `BGEReranker(BaseReranker)`
   - `HybridRetrievalResult(BaseModel)`: شامل `chunks: list[ScoredChunk]`، `retrieval_ms: float`، `rerank_ms: float`.
   - `HybridRetriever`: ارکستریتور سطح بالا که `DenseRetriever` و `SparseRetriever` را با `asyncio.gather(return_exceptions=False)` اجرا کرده، با `reciprocal_rank_fusion` ترکیب نموده و با `reranker.rerank` به Top-5 تبدیل می‌کند.
5. **`src/retrieval/__init__.py`:**
   - صادر کردن کلاس‌ها و توابع عمومی لایه بازیابی در `__all__`.
6. **`tests/unit/test_retrieval.py`:**
   - تست `DenseRetriever` روی Qdrant درون‌حافظه‌ای (`:memory:`) با و بدون `modality_filter` (`text`, `table`, `figure`).
   - تست `SparseRetriever` روی ایندکس `BM25s` محلی با پرسش‌های فارسی و انگلیسی و فیلتر مدالیته.
   - تست ریاضی دقیق `reciprocal_rank_fusion` با $k=60$، حذف تکراری‌ها (`chunk_id`)، و محدودسازی به Top-15.
   - تست `MockReranker` و `BGEReranker` (همراه با تست Fallback زمان اجرا و نرمال‌سازی سیگموئید در `[0.0, 1.0]`).
   - تست End-to-End کلاس `HybridRetriever` (اجرای موازی Dense + Sparse -> RRF -> Rerank به Top-5 و حفظ کامل `bounding_box`، `image_path` و `table_html`).

---

## 5. Acceptance Criteria & Verification Plan

### Acceptance Criteria
- [ ] بازیابی موازی Top-20 از Qdrant (`DenseRetriever`) و Top-20 از `BM25s` (`SparseRetriever`) با `asyncio.gather` پیاده‌سازی شده است.
- [ ] فیلتر اختیاری `modality_filter` (`"text" | "table" | "figure"`) در هر دو مسیر Dense و Sparse و در `HybridRetriever` پشتیبانی می‌شود.
- [ ] ادغام نتایج با الگوریتم Reciprocal Rank Fusion ($k=60$) و انتخاب ۱۵ کاندیدای برتر برای بازرتبه‌بندی انجام می‌شود.
- [ ] بازرتبه‌بندی کاندیداها به Top-5 با `BAAI/bge-reranker-v2-m3` (و `MockReranker` قطعی در محیط بدون GPU/CI) با امتیاز نرمال‌شده در بازه `[0.0, 1.0]` انجام می‌شود.
- [ ] تمامی فیلدهای Attribution (`page_number`, `bounding_box`, `modality`, `image_path`, `table_html`) بدون تغییر حفظ می‌شوند.

### Verification Commands
```bash
pytest tests/unit/test_retrieval.py -v
ruff check src/retrieval tests/unit/test_retrieval.py
mypy --strict src/retrieval tests/unit/test_retrieval.py
```
