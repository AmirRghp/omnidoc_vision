# Task 7.1 — Three-Way Ablation Benchmark (`PyPDF` vs. `Docling Text/Table` vs. `OmniDoc-Vision`)

پیاده‌سازی بنچمارک سه‌گانه کمی (`evals/run_multimodal_benchmark.py`) و تست‌های واحد مرتبط برای ارزیابی و مقایسه سه نسخه خط لوله:
1. **`Naive PyPDF Baseline (Text Only)`** — استخراج متن خام با `pypdf` بدون بازسازی جداول و بدون درک تصاویر/نمودارها.
2. **`Docling Structured RAG (Text + Reconstructed Tables)`** — پارسینگ ساختاریافته با `DoclingParser` (شامل متن و بازسازی جداول ادغام‌شده Markdown/HTML) و مسیریابی سریع متنی/جدولی بدون استدلال بصری روی تصاویر (`figure`).
3. **`OmniDoc-Vision (Docling + Adaptive Qwen2.5-VL)`** — خط لوله کامل چندوجهی شامل `DoclingParser`، کپشن‌گذاری فنی Approach A (`VisualCaptioner`)، بازیابی ترکیبی (`Qdrant BGE-M3` + `BM25s` + `RRF` + `BGE-Reranker`) و مسیریاب تطبیقی `AdaptiveRAGWorkflow` (`FAST_TEXT_TABLE_PATH` در برابر `VISUAL_VLM_PATH`).

---

## User Review & Critical Decisions

> [!IMPORTANT]
> بر اساس مشخصات `docs/PLAN.md` (Task 7.1)، `docs/TESTING.md` (بخش ۲) و `docs/PRD.md` (بخش‌های ۶ و ۸)، نیازمندی‌های تسک ۷.۱ کاملاً مشخص و بدون ابهام هستند:

- **سطح پیچیدگی تسک:** **Medium** — پیاده‌سازی ماژول بنچمارک CLI در `evals/run_multimodal_benchmark.py`، تست‌های واحد در `tests/unit/test_multimodal_benchmark.py` و به‌روزرسانی `Makefile` در صورت نیاز. (در Plan Mode هیچ فایلی تغییر نمی‌کند و منتظر تأیید شما می‌مانم.)
- **پشتیبانی کامل از اجرای Offline / CI (قانون ۲ در `AGENTS.md`):** بنچمارک باید بدون نیاز به سرویس خارجی Qdrant یا سرور GPU vLLM، به صورت ۱۰۰٪ خودکفا با استفاده از `AsyncQdrantClient(location=":memory:")`، تولید خودکار PDFهای بنچمارک (در صورت عدم وجود در `data/raw_pdfs/`) و حالت determinisic/offline اجرا شود و تمام گیت‌های کیفیت CI (`Table Cell F1 >= 0.88`, `Visual Chart QA Accuracy >= 0.80`, `Context Recall@5 >= 0.85`, `Faithfulness >= 0.90`, `Fast Path P95 <= 500ms`, `Visual Path P95 <= 2000ms`) را محقق سازد.
- **خروجی دوگانه (JSON + Markdown):** دستور `python -m evals.run_multimodal_benchmark --output evals/results/benchmark_report.json` هم گزارش ساختاریافته JSON (شامل خلاصه متریک‌های هر ۳ واریانت، بررسی Production CI Gates و جزئیات کامل هر ۳۰ پرسش) و هم جدول مقایسه‌ای Markdown (`benchmark_report.md`) را در `evals/results/` تولید می‌کند.

---

## 1. Overview & Core Concept

- **What It Does:** تمام ۳۰ پرسش دوزبانه موجود در `data/golden_multimodal_dataset.json` (شامل ۱۰ پرسش جدول پیچیده، ۱۰ پرسش بصری نمودار/دیاگرام، و ۱۰ پرسش متنی) را روی ۳ معماری مختلف اجرا و متریک‌های استاندارد مهندسی RAG چندوجهی را محاسبه و گزارش می‌کند.
- **Target Audience:** مهندسان AI و داوران فنی که به دنبال اثبات کمی برتری معماری `OmniDoc-Vision` نسبت به `Naive PyPDF` و `Docling Text/Table` هستند.
- **Key Value:** اثبات تجربی و قابل بازتولید (Reproducible) اینکه بازسازی جداول Docling خطای جابه‌جایی ستون‌های `PyPDF` را برطرف می‌کند (`Table F1: ~0.52 -> 0.95`) و مسیریاب تطبیقی `Qwen2.5-VL` دقت پرسش‌وپاسخ روی نمودارها را بدون تحمیل سربار تاخیر به مسیر متنی، از `~0.30 / ~0.48` به `>= 0.90` ارتقا می‌دهد.

---

## 2. User Experience & Reporting Output

- **اجرای خط فرمان (CLI):**
  - پشتیبانی از آرگومان‌های `--output` (مسیر خروجی JSON، پیش‌فرض `evals/results/benchmark_report.json`)، `--markdown-output` (مسیر خروجی Markdown، پیش‌فرض همنام با پسوند `.md` در کنار فایل JSON) و `--dataset` (مسیر دیتاست طلایی).
- **ساختار گزارش خروجی:**
  1. **جدول مقایسه‌ای ۳ واریانت (Markdown & Console Summary):** نمایش دقیق متریک‌های ۶گانه در کنار آستانه‌های `Production CI Gate` مطابق جدول بخش ۲ در `docs/TESTING.md`.
  2. **فایل JSON کامل:** شامل متادیتای اجرا، نتایج تجمیعی هر واریانت (`VariantMetrics`)، وضعیت عبور از گیت‌های CI (`passed_ci_gates: bool`) و رکوردهای تک‌تک ۳۰ پرسش (`QuestionEvalResult`) به همراه مسیر انتخاب‌شده (`FAST_TEXT_TABLE_PATH` / `VISUAL_VLM_PATH`) و citationها.

---

## 3. Key Technical Decisions & Trade-Offs

- **Decision 1: ایزولاسیون کامل هر واریانت با `AsyncQdrantClient(location=":memory:")` و دایرکتوری موقت `BM25s`**
  - *Chosen Approach:* هر یک از ۳ واریانت ایندکس ترکیبی مستقل خود را در حافظه و دایرکتوری مجزا می‌سازد تا چانک‌های `PyPDF`، `Docling Text/Table` و `OmniDoc-Vision` با یکدیگر تداخل نداشته باشند.
  - *Why:* تضمین صحت علمی آزمایش Ablation و اجرای سریع و بدون وابستگی به کانتینر خارجی در محیط CI/CD و تست.
- **Decision 2: غنی‌سازی کانتکست و استخراج پاسخ دقیق در حالت Offline برای ارزیابی واقعی خط لوله**
  - *Chosen Approach:*
    - در واریانت ۱ (`pypdf_baseline`): صفحات PDF صرفاً با `pypdf.PdfReader` به چانک‌های متنی خام تبدیل می‌شوند (جداول چندسطری مسطح شده و تصاویر نمودار نادیده گرفته می‌شوند).
    - در واریانت ۲ (`docling_text_table`): چانک‌های `TEXT` و `TABLE` استخراج‌شده توسط `DoclingParser` ایندکس می‌شوند (بدون کپشن‌گذاری بصری Approach A و بدون ارسال تصویر به VLM؛ همه پرسش‌ها از مسیر سریع متنی/جدولی پاسخ داده می‌شوند).
    - در واریانت ۳ (`omnidoc_vision`): چانک‌های `TEXT`، `TABLE` و `FIGURE` به همراه کپشن‌های فنی غنی‌شده (`VisualCaptioner`) ایندکس شده و از طریق `AdaptiveRAGWorkflow` مسیریابی می‌شوند (پرسش‌های متنی/جدولی در `FAST_TEXT_TABLE_PATH` و پرسش‌های تصویری در `VISUAL_VLM_PATH`).
  - *Why:* بازتاب وفادارانه تفاوت معماری هر سه رویکرد روی داده‌های واقعی PDFهای تولیدشده (`cloud_database_benchmark_2026.pdf`, `distributed_cache_architecture.pdf`, `gpu_inference_capacity_report.pdf`).
- **Decision 3: محاسبه دقیق و استاندارد متریک‌های ۶گانه**
  - *Chosen Approach:*
    - **`table_cell_f1`:** میانگین F1 سطح توکن/مقدار عددی و ساختار سلول روی پرسش‌های دسته `table` و مقایسه جداول بازسازی‌شده با مقادیر مرجع.
    - **`visual_qa_accuracy`:** نرخ صحت پاسخ روی ۱۰ پرسش دسته `visual` (تطابق مقادیر کلیدی و نقاط عطف نمودار/توپولوژی با `ground_truth`).
    - **`context_recall_at_5`:** نسبت پرسش‌هایی (از ۳۰ پرسش) که در Top-5 چانک بازیابی‌شده آن‌ها، سند و صفحه مورد انتظار (`expected_pages`) و مدالیته مرتبط حضور دارد.
    - **`faithfulness`:** میزان مستند بودن ادعاها و اعداد پاسخ نهایی به شواهد داخل چانک‌های بازیابی‌شده (`retrieved_chunks`).
    - **`fast_path_p95_ms` و `visual_path_p95_ms`:** صدک ۹۵ تاخیر (`percentile 95`) به تفکیک مسیر `FAST_TEXT_TABLE_PATH` و `VISUAL_VLM_PATH` (برای واریانت‌های ۱ و ۲ که مسیر بصری ندارند، `visual_path_p95_ms` برابر `None` / `N/A` ثبت می‌شود).

---

## 4. Technical Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│             data/golden_multimodal_dataset.json (30 QA Pairs)           │
│           + data/raw_pdfs/*.pdf (Auto-built if missing via script)      │
└───────────────────┬─────────────────────────┬───────────────────────────┘
                    │                         │
      ┌─────────────▼─────────────┐           │           ┌─────────────────────────────┐
      │ 1. Naive PyPDF Baseline   │           │           │ 3. OmniDoc-Vision (Full)    │
      │ - pypdf.PdfReader text    │           │           │ - DoclingParser (Txt/Tbl/Fig│
      │ - Hybrid Index (Qdrant+BM)│           ▼           │ - Approach A VisualCaptioner│
      │ - Fast Text Generation    │ ┌───────────────────┐ │ - HybridRetriever + Reranker│
      └─────────────┬─────────────┘ │ 2. Docling Struct │ │ - AdaptiveRAGWorkflow       │
                    │               │ - Docling Text+Tbl│ │   (Fast Path vs Visual VLM) │
                    │               │ - Fast Text/Table │ └──────────────┬──────────────┘
                    │               └─────────┬─────────┘                │
                    └─────────────────────────┼──────────────────────────┘
                                              ▼
                    ┌─────────────────────────────────────────────────────┐
                    │         Evaluator & Metric Aggregator               │
                    │ - Table Cell Extraction F1-Score                    │
                    │ - Visual Chart & Diagram QA Accuracy                │
                    │ - Overall Context Recall@5 & Answer Faithfulness    │
                    │ - Fast Path P95 Latency & Visual Path P95 Latency   │
                    └─────────────────────────┬───────────────────────────┘
                                              ▼
                    ┌─────────────────────────────────────────────────────┐
                    │   evals/results/benchmark_report.json & .md         │
                    └─────────────────────────────────────────────────────┘
```

---

## 5. Acceptance Criteria & Verification Plan

### Acceptance Criteria
- [ ] اجرای `python -m evals.run_multimodal_benchmark --output evals/results/benchmark_report.json` تمامی ۳۰ پرسش `data/golden_multimodal_dataset.json` را روی هر ۳ واریانت ارزیابی کرده و فایل‌های گزارش JSON و Markdown را در `evals/results/` ذخیره می‌کند.
- [ ] متریک‌های `Table Cell F1-Score`، `Visual Chart QA Accuracy`، `Context Recall@5`، `Faithfulness`، `Fast Path P95 Latency` و `Visual Path P95 Latency` محاسبه شده و واریانت `OmniDoc-Vision` تمامی آستانه‌های `Production CI Gate` در `docs/TESTING.md` را پاس می‌کند.
- [ ] فیلدهای Attribution (`page_number`, `bounding_box`, `modality`, `image_url`) در نتایج ارزیابی حفظ می‌شوند.
- [ ] تست‌های واحد در `tests/unit/test_multimodal_benchmark.py` صحت محاسبه متریک‌ها، ساختار گزارش خروجی و عملکرد CLI را پوشش می‌دهند.
- [ ] دستورات `ruff check .`، `mypy --strict src scripts evals` و `pytest` بدون هیچ خطایی اجرا می‌شوند.
