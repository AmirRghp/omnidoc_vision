# Task 2.1 — Docling Layout, Table Reconstruction & Figure Cropping Engine

پیاده‌سازی موتور پردازش ساختار اسناد (`DoclingParser`) برای استخراج بلوک‌های متنی با حفظ ترتیب مطالعه، بازسازی جداول ادغام‌شده (به‌صورت Markdown و HTML)، و برش تصاویر/جداول به‌همراه مختصات نرمال‌شده (`BoundingBox` در بازه `[0, 1]`) و پشتیبانی کامل از حالت Fallback آفلاین/CI بدون نیاز به وزن‌های سنگین GPU.

## User Review & Critical Decisions

> [!IMPORTANT]
> بر اساس مشخصات `docs/PLAN.md`، `docs/ARCHITECTURE.md`، `docs/DATABASE.md` و `docs/SECURITY.md`، نیازمندی‌های تسک ۲.۱ کاملاً مشخص و بدون ابهام هستند. تصمیمات کلیدی زیر در طراحی لحاظ شده‌اند:

- **سطح پیچیدگی تسک:** **Medium** — شامل پیاده‌سازی ماژول اصلی استخراج اسناد و تست‌های واحد جامع آن بدون تغییر در اسکیمای موجود.
- **معماری دوگانه (Docling Primary + Deterministic Fallback):** استفاده از `DocumentConverter` در `docling` (با `do_table_structure=True` و `generate_picture_images=True`) به‌عنوان موتور اصلی، همراه با Fallback مبتنی بر `pdfplumber` و `pypdf`/`Pillow` تا در محیط‌های تست/CI یا بدون مدل‌های سنگین GPU، استخراج متن، جدول (Markdown + HTML) و برش تصاویر (`.png`) به‌صورت کاملاً قطعی (Deterministic) کار کند.
- **حفظ امنیت و محدودیت منابع (`docs/SECURITY.md`):** اعمال اعتبارسنجی حجم فایل (`max_pdf_size_mb`)، تعداد صفحات (`max_pdf_pages`)، سقف پیکسل تصویر (`PIL.Image.MAX_IMAGE_PIXELS = 25_000_000`)، و محدودسازی ابعاد تصاویر برش‌خورده به `<= max_image_dim` (`1280px`)، و ذخیره مسیر نسبی ایمن (`data/crops/...`) در `MultimodalChunk`.

---

## 1. Overview & Core Concept

- **What It Does:** یک فایل PDF مهندسی را دریافت کرده، محدودیت‌های امنیتی آن را بررسی می‌کند و خروجی را به لیستی از اشیاء استاندارد `MultimodalChunk` (`text`، `table`، `figure`) با مختصات صفحه نرمال‌شده (`BoundingBox`) و فایل‌های برش‌خورده `.png` در `data/crops/` تبدیل می‌نماید.
- **Target Audience / Downstream Consumers:** لایه‌های بعدی پایپ‌لاین شامل `visual_captioner.py` (تسک ۲.۲)، `indexer.py` (تسک ۲.۳)، و اندپوینت `/api/v1/ingest` (تسک ۵.۱).
- **Key Value:** جداسازی کامل تحلیل چیدمان، بازسازی ساختار جداول و استخراج تصاویر از مرحله استنتاج زمان پرسش (Query-Time Inference) با حفظ ۱۰۰٪ اطلاعات Attribution (`page_number`، `bounding_box`، `modality`، `image_path`، `table_html`).

---

## 2. User Experience & Visual Design (System & Attribution Flow)

- **Key Execution Flow (Flow 1 — Ingestion):**
  1. دریافت مسیر فایل PDF و اعتبارسنجی اولیه (موجود بودن فایل، پسوند `.pdf`، حجم `<= 50MB`، تعداد صفحات `<= 100`).
  2. استخراج ساختار سند (عناوین بخش‌ها `section_title`، پاراگراف‌های متنی، جداول و تصاویر) به ترتیب مطالعه.
  3. برای هر جدول: تولید هم‌زمان نسخه Markdown در فیلد `content` و ساختار کامل `<table>...</table>` در فیلد `table_html` (بدون تگ `<script`) و ذخیره اختیاری/پشتیبانی از تصویر برش‌خورده جدول.
  4. برای هر شکل/نمودار (`figure`): ذخیره تصویر برش‌خورده با ابعاد استاندارد (`<= 1280px`) در `data/crops/{doc_id}_p{page:02d}_fig{idx:02d}.png`، ثبت مسیر نسبی `data/crops/...` در `image_path`، و مقداردهی اولیه `content` با کپشن/متادیتای شکل (که در تسک ۲.۲ توسط `VisualCaptioner` غنی‌سازی می‌شود).
  5. نرمال‌سازی تمامی مختصات `BoundingBox` به بازه `[0.0, 1.0]` با مبدأ `CoordOrigin.TOPLEFT` و تضمین `x1 > x0` و `y1 > y0`.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: استراتژی اجرای Docling و Fallback قطعی (`use_docling` / Auto-Fallback)**
  - *Chosen Approach:* کلاس `DoclingParser` در صورت فعال بودن و در دسترس بودن وزن‌های `docling` از `DocumentConverter` استفاده می‌کند و در صورت عدم دسترسی به مدل‌های محلی یا انتخاب حالت Fallback (مثلاً در تست‌های سریع CI)، به موتور Fallback داخلی مبتنی بر `pdfplumber` (برای استخراج جداول، متن، عناوین و جعبه مرزی آن‌ها) و `pypdf`/`Pillow` (برای استخراج بایت‌های تصاویر جاسازی‌شده در PDF و ذخیره در `data/crops/`) سوییچ می‌کند.
  - *Why:* اجرای بدون خطا و سریع در محیط‌های CI/CD فاقد GPU و بدون نیاز به دانلود گیگابایت‌ها وزن مدل در تست‌های واحد (`AGENTS.md` Rule #2).
- **Decision 2: اجرای غیرمسدودکننده (Async Non-Blocking Boundaries)**
  - *Chosen Approach:* ارائه متد همگام `parse_pdf(pdf_path: Path) -> list[MultimodalChunk]` برای پردازش مستقیم و متد ناهمگام `aparse_pdf(pdf_path: Path) -> list[MultimodalChunk]` با استفاده از `asyncio.to_thread`.
  - *Why:* رعایت کامل الزام `docs/CODE_STYLE.md` جهت جلوگیری از مسدود شدن Event Loop در FastAPI و Chainlit.
- **Decision 3: فرمت شناسه‌ها و مسیرهای خروجی (`chunk_id`, `doc_id`, `image_path`)**
  - *Chosen Approach:* تولید شناسه سند پایدار (`doc_id` از نام فایل نرمال‌شده) و شناسه‌های یکتا مانند `{doc_id}_p{page:02d}_{mod}{idx:02d}` و ذخیره `image_path` به‌صورت مسیر نسبی استاندارد `data/crops/<filename>.png` منطبق با ولیدیتورهای `MultimodalChunk`.
  - *Why:* تضمین سازگاری با قرارداد `docs/DATABASE.md` و جلوگیری از خطای Path Traversal در `MultimodalChunk`.

---

## 4. Technical Architecture & Data Strategy

### Architecture & Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PDF Input (data/raw_pdfs/*.pdf)                   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     PDF Security & Resource Validator                    │
│  - Existence, File Size <= max_pdf_size_mb, Page Count <= max_pdf_pages  │
│  - Raises: PdfValidationError / DocumentProcessingError                  │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
     ┌──────────────────────────────┐  ┌──────────────────────────────┐
     │   Primary: IBM Docling       │  │   Fallback: PDFPlumber+PyPDF │
     │ - PdfPipelineOptions         │  │ - Reading-order text & sects │
     │   do_table_structure=True    │  │ - Table grid -> MD & HTML    │
     │   generate_picture_images    │  │ - Embedded images -> PNG     │
     └──────────────┬───────────────┘  └──────────────┬───────────────┘
                    └────────────────┬────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             Normalization, Image Clamping & Asset Persistence            │
│  - Clamp PNG dimensions <= max_image_dim (1280px) via Pillow             │
│  - Normalize BBox [x0, y0, x1, y1] into [0.0, 1.0] (CoordOrigin.TOPLEFT) │
│  - Emit validated list[MultimodalChunk] (TEXT, TABLE, FIGURE)            │
└──────────────────────────────────────────────────────────────────────────┘
```

### Target Files & Signatures

1. **ایجاد فایل موتور پردازش (`src/ingestion/docling_parser.py`):**
   - تنظیم `Image.MAX_IMAGE_PIXELS = 25_000_000` مطابق `docs/SECURITY.md`.
   - کلاس `DoclingParser`:
     - `__init__(self, settings: Settings | None = None, *, crops_dir: Path | None = None, use_docling: bool = False) -> None` (پشتیبانی از تزریق تنظیمات، مسیر ذخیره برش‌ها، و کنترل صریح یا خودکار موتور `docling` در برابر Fallback).
     - `parse_pdf(self, pdf_path: Path | str) -> list[MultimodalChunk]`: اعتبارسنجی فایل، استخراج بلوک‌های متنی، بازسازی جداول (Markdown و HTML با حفظ ساختار)، ذخیره تصاویر برش‌خورده در `crops_dir` با تغییر اندازه به `<= max_image_dim`، و بازگرداندن لیست `MultimodalChunk`.
     - `aparse_pdf(self, pdf_path: Path | str) -> list[MultimodalChunk]`: اجرای غیرمسدودکننده `parse_pdf` با `asyncio.to_thread`.
   - توابع کمکی ماژولار و تایپ‌شده برای تبدیل جدول به Markdown و HTML ایمن، نرمال‌سازی مختصات صفحه به `[0.0, 1.0]` با تبدیل مبدأ `BOTTOMLEFT` به `TOPLEFT` در صورت نیاز، و برش/تغییر اندازه تصویر تا سقف `1280px`.
2. **به‌روزرسانی صادرات پکیج (`src/ingestion/__init__.py`):**
   - صادر کردن `DoclingParser` در `__all__`.
3. **ایجاد تست‌های واحد (`tests/unit/test_docling_parser.py`):**
   - تست استخراج روی فایل‌های PDF بنچمارک تولیدشده (`cloud_database_benchmark_2026.pdf` و غیره در `tmp_path`):
     - بررسی تولید هر سه نوع چانک (`ChunkModality.TEXT`، `ChunkModality.TABLE`، `ChunkModality.FIGURE`).
     - بررسی بازسازی جداول شامل Markdown در `content` و جدول معتبر `<table>...</table>` در `table_html`.
     - بررسی ذخیره فایل‌های `.png` در `crops/`، نسبی بودن `image_path`، و رعایت سقف ابعاد `<= 1280px`.
     - بررسی صحت `BoundingBox` نرمال‌شده (`0.0 <= x0 < x1 <= 1.0`، `0.0 <= y0 < y1 <= 1.0` و `coord_origin == CoordOrigin.TOPLEFT`).
     - بررسی مدیریت خطاها (`PdfValidationError` برای فایل ناموجود، حجم بیش از حد، یا تعداد صفحات بیش از حد مجاز).
     - بررسی کارکرد متد ناهمگام `aparse_pdf` و مسیر اقتباس از شیء خروجی `Docling` (با Mock سبک از خروجی `DocumentConverter` برای پوشش کامل کدهای مسیر Docling علاوه بر مسیر Fallback).

---

## 5. Acceptance Criteria & Verification Plan

### Acceptance Criteria
- [ ] `DoclingParser` در `src/ingestion/docling_parser.py` پیاده‌سازی شده و هر سه مدالیته (`text`، `table`، `figure`) را در قالب `list[MultimodalChunk]` معتبر برمی‌گرداند.
- [ ] جداول به هر دو فرمت Markdown (`content`) و HTML (`table_html`) بازسازی می‌شوند.
- [ ] تصاویر/نمودارها در دایرکتوری `crops` به‌صورت `.png` ذخیره شده، ابعاد آن‌ها به حداکثر `1280x1280` محدود (`clamp`) می‌شود، و `image_path` به‌صورت مسیر نسبی بدون Traversal ثبت می‌گردد.
- [ ] مختصات `BoundingBox` برای همه چانک‌ها در بازه `[0.0, 1.0]` با `CoordOrigin.TOPLEFT` نرمال‌سازی و اعتبارسنجی می‌شوند.
- [ ] محدودیت‌های امنیتی PDF (`max_pdf_size_mb`، `max_pdf_pages`، `PIL.Image.MAX_IMAGE_PIXELS`) رعایت شده و در صورت نقض، `PdfValidationError` صادر می‌شود.
- [ ] تمامی تست‌ها بدون نیاز به GPU یا سرویس خارجی (در حالت CI/Offline) اجرا و پاس می‌شوند.

### Verification Commands
```bash
pytest tests/unit/test_docling_parser.py -v
ruff check src/ingestion/docling_parser.py tests/unit/test_docling_parser.py
mypy --strict src/ingestion/docling_parser.py tests/unit/test_docling_parser.py
```
