# Task 8.1 — Docker Compose with Auto-Ingest Init, GitHub Actions CI & Portfolio README

پیاده‌سازی زیرساخت استقرار تک‌دستوری (`Dockerfile` و `docker-compose.yml` به همراه گام خودکار `init-ingest`)، خط لوله کیفیت و امنیت GitHub Actions CI (`.github/workflows/ci.yml`) و مستند جامع پورتفولیو (`README.md`) برای تکمیل فاز ۸ پروژه **OmniDoc-Vision**.

---

## User Review & Critical Decisions

> [!IMPORTANT]
> بر اساس مشخصات `docs/PLAN.md` (Task 8.1)، `docs/ARCHITECTURE.md`، `docs/PRD.md` و `docs/TESTING.md`، نیازمندی‌های این تسک کاملاً مشخص و بدون ابهام هستند. در حالت Plan Mode هیچ فایلی در ریپازیتوری تغییر نکرده است؛ لطفاً برنامه زیر را بررسی و تأیید فرمایید.

- **سطح پیچیدگی تسک:** **Medium** — ایجاد `Dockerfile`، `docker-compose.yml`، `.github/workflows/ci.yml`، بازنویسی جامع `README.md` و افزودن تست‌های واحد اعتبارسنجی زیرساخت در `tests/unit/test_packaging.py`.
- **راه‌اندازی Zero-Touch با گام `init-ingest`:** اجرای `docker compose up -d` به صورت خودکار سرویس‌های زیرساختی (`qdrant`، `redis`، `langfuse-db`، `langfuse`) را بالا آورده و سرویس یک‌بارمصرف `init-ingest` ابتدا PDFهای بنچمارک را تولید (`python scripts/build_benchmark_pdfs.py`) و سپس هر ۳ سند مهندسی را در ایندکس ترکیبی Qdrant + BM25s ایندکس می‌کند تا زمان رسیدن به اولین پاسخ (TTFR) به حداقل برسد.
- **مستندسازی جامع Portfolio (`README.md`):** شامل پیوند به هر ۳ پروژه سه‌گانه پورتفولیو (`omnidoc-rag`، `omniops-sre-agent`، `omnidoc-vision`)، دیاگرام معماری، جدول بودجه `16GB VRAM`، جدول مقایسه ۳گانه Ablation، جدول مقاومت در برابر نویز (`Noise Robustness`) و بخش شفاف `Limitations & Failure Modes`.

---

## 1. Overview & Core Concept

- **What It Does:** بسته نهایی استقرار، اتوماسیون CI/CD و مستندات فنی پروژه **OmniDoc-Vision** را فراهم می‌کند تا هر مهندس یا داور فنی بتواند با یک دستور (`docker compose up -d`) کل پشته چندوجهی را اجرا کند و در CI تمامی گیت‌های کیفیت (`ruff`, `mypy`, `pytest`, `wheel build`) به صورت خودکار بررسی شوند.
- **Target Audience:** مدیران استخدام مهندسی AI، معماران سیستم و مهندسان SRE که ریپازیتوری را از نظر کیفیت مهندسی، بودجه سخت‌افزاری 16GB VRAM، نتایج کمی بنچمارک و سهولت استقرار ارزیابی می‌کنند.
- **Key Value:** حذف کامل مراحل دستی آماده‌سازی داده و ایندکس اولیه (Zero Manual Setup / Low TTFR) و ارائه مستندات مهندسی مبتنی بر شواهد کمی و تصمیمات معماری (ADRs).

---

## 2. Technical Architecture & Orchestration Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        docker compose up -d                                │
└──────┬──────────────────────┬───────────────────────┬──────────────────────┘
       │                      │                       │
       ▼                      ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────────────────────┐
│    qdrant    │       │    redis     │       │   langfuse + langfuse-db     │
│  (:6333/6334)│       │   (:6379)    │       │   (Observability :3001)      │
└──────┬───────┘       └──────┬───────┘       └───────────────┬──────────────┘
       │ (healthy)            │ (healthy)                     │
       └──────────┬───────────┴───────────────────────────────┘
                  ▼
       ┌──────────────────────────────────────────────────────┐
       │                     init-ingest                      │
       │ 1. python scripts/build_benchmark_pdfs.py            │
       │ 2. Parse, Caption (Approach A) & Hybrid Index PDFs   │
       └──────────────────────────┬───────────────────────────┘
                                  │ (service_completed_successfully)
                                  ▼
       ┌──────────────────────────────────────────────────────┐
       │          omnidoc-api (:8001) & omnidoc-ui (:8002)    │
       │   FastAPI REST Backend + Chainlit Visual Console     │
       └──────────────────────────────────────────────────────┘
```

---

## 3. Key Implementation Details

### 3.1. `Dockerfile`
- پایه `python:3.11-slim` با نصب وابستگی‌های سیستمی موردنیاز برای پردازش تصویر و PDF (`libgl1`, `libglib2.0-0`, `curl`).
- نصب وابستگی‌های پایتون از `requirements.txt` با `--no-cache-dir` و کپی سورس‌کد (`src/`, `scripts/`, `evals/`, `configs/`, `public/`, `.chainlit/`, `main.py`, `app.py`, `pyproject.toml`, `README.md`).
- ایجاد کاربر غیر ریشه (`omnidoc`) و دایرکتوری‌های داده (`data/raw_pdfs`, `data/crops`, `data/bm25_index`, `evals/results`) با دسترسی مناسب جهت رعایت اصول امنیتی (`docs/SECURITY.md`).
- تعریف `HEALTHCHECK` روی اندپوینت `/health` سرویس FastAPI.

### 3.2. `docker-compose.yml`
- **`qdrant`:** ایمیج `qdrant/qdrant:v1.11.3` روی پورت `6333:6333` با volume پایدار و `healthcheck`.
- **`redis`:** ایمیج `redis:7.4-alpine` روی پورت `6379:6379` با `healthcheck` (`redis-cli ping`).
- **`langfuse-db` & `langfuse`:** سرویس‌های تله‌متری و ردیابی (`langfuse/langfuse:2` به همراه `postgres:16-alpine`) با متغیرهای محیطی استاندارد.
- **`init-ingest`:** کانتینر مقداردهی اولیه (`restart: "no"`) که پس از آماده شدن `qdrant` و `redis`:
  1. اسکریپت `python scripts/build_benchmark_pdfs.py` را اجرا می‌کند تا ۳ فایل PDF مهندسی و دیتاست طلایی ساخته شوند.
  2. یک اسکریپت خط فرمان پایتونی کوتاه با استفاده از `DoclingParser`، `VisualCaptioner` و `HybridIndexer` اسناد تولیدشده را مستقیماً در Qdrant و `data/bm25_index` ایندکس می‌کند.
- **`omnidoc-api` & `omnidoc-ui`:** وابسته به تکمیل موفق `init-ingest` (`condition: service_completed_successfully`) و سلامت `qdrant`/`redis`، اشتراک‌گذاری volume داده‌ها (`./data:/app/data`) برای دسترسی یکپارچه به فایل‌های crop و شاخص BM25.

### 3.3. `.github/workflows/ci.yml`
- اجرا روی `push` و `pull_request` به شاخه `main`.
- راه‌اندازی پایتون `3.11` با کش `pip` در `ubuntu-latest`.
- مراحل Quality Gate:
  1. نصب وابستگی‌ها (`pip install -r requirements.txt`).
  2. بررسی Linter و فرمت (`ruff check .`).
  3. بررسی تایپ استاتیک (`mypy src` و `mypy --strict src scripts evals`).
  4. تولید PDFهای بنچمارک (`python scripts/build_benchmark_pdfs.py`).
  5. اجرای تست‌های واحد و یکپارچه‌سازی (`pytest -v`).
  6. بررسی بیلد پکیج (`python -m pip wheel . --no-deps -w dist`).

### 3.4. `README.md` (Portfolio Showcase)
- هدر پروژه و معرفی جایگاه **OmniDoc-Vision** به عنوان پروژه سوم از سه‌گانه پورتفولیو مهندسی AI همراه با جدول لینک به هر ۳ ریپازیتوری:
  1. **Project 1:** `omnidoc-rag` (Hybrid Enterprise Text RAG & CRAG)
  2. **Project 2:** `omniops-sre-agent` (Autonomous Incident Response & Tool-Use Agent)
  3. **Project 3:** `omnidoc-vision` (Multimodal Document Intelligence, Table Reconstruction & Adaptive Visual RAG)
- دیاگرام معماری ASCII (مطابق `docs/ARCHITECTURE.md`) و توضیح تصمیمات معماری کلیدی (`ADR-001` و `ADR-002`: چرا Approach A + Docling به جای ColPali خالص یا دو فضای برداری مجزا).
- جدول تفکیک بودجه سخت‌افزاری **16GB VRAM** (`Qwen2.5-VL-7B-Instruct-AWQ` با ~6.8GB، `BGE-M3` با ~1.1GB، `BGE-Reranker-v2-m3` با ~1.1GB، `IBM Docling` با ~1.2GB و مجموع ~10.2GB از 16GB).
- جدول نتایج کمی **3-Way Multimodal Ablation Benchmark** (`Naive PyPDF` vs. `Docling Structured RAG` vs. `OmniDoc-Vision`) و جدول **Document Degradation & Noise Robustness** (چرخش اسکن $\pm 5^\circ$، تاری گوسین $\sigma=1.2$، واترمارک و فشرده‌سازی 150 DPI).
- راهنمای راه‌اندازی سریع با یک دستور (`docker compose up -d`) و راه‌اندازی محلی توسعه، مرجع اندپوینت‌های FastAPI و کنسول Chainlit.
- بخش مهندسی **`Limitations & Failure Modes`** (محدودیت ۲ تصویر در هر پرسش برای حفاظت از KV-Cache در 16GB VRAM، چالش نمودارهای سه‌بعدی بسیار متراکم یا دست‌نویس‌های ناخوانا، سربار اولیه کپشن‌گذاری در زمان Ingest و راهکارهای کاهش ریسک).

### 3.5. گسترش تست‌های بسته‌بندی (`tests/unit/test_packaging.py`)
- افزودن تست‌های واحد برای اعتبارسنجی وجود و ساختار `Dockerfile`، `docker-compose.yml` (حضور سرویس‌های `qdrant`, `redis`, `langfuse`, `init-ingest`), `.github/workflows/ci.yml` و بخش‌های کلیدی `README.md`.

---

## 4. Acceptance Criteria & Verification Plan

### Acceptance Criteria
- [ ] فایل `docker-compose.yml` شامل سرویس‌های `qdrant`، `redis`، `langfuse` و گام خودکار `init-ingest` است به طوری که با `docker compose up -d` تولید و ایندکس PDFهای بنچمارک بدون دخالت دستی انجام شود.
- [ ] فایل `Dockerfile` با رعایت اصول امنیتی (کاربر غیر root و محدودیت‌های محیطی) ساخته شده است.
- [ ] فایل `.github/workflows/ci.yml` شامل گیت‌های `ruff check .`، `mypy`، `pytest` و ساخت wheel است.
- [ ] فایل `README.md` شامل لینک‌های هر ۳ ریپازیتوری پورتفولیو (`omnidoc-rag`, `omniops-sre-agent`, `omnidoc-vision`)، دیاگرام معماری، جدول بودجه 16GB VRAM، جدول 3-Way Ablation، جدول Noise Robustness و بخش `Limitations & Failure Modes` است.
- [ ] دستور اعتبارسنجی تسک ۸.۱ (`ruff check . && mypy src && pytest`) بدون هیچ خطایی (0 errors / 0 failures) اجرا می‌شود.
