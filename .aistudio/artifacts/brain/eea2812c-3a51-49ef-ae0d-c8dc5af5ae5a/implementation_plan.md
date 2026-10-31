# Task 4.2 — LangGraph State Machine (`CRAG Evaluator` + `Adaptive Router`)

پیاده‌سازی ماشین وضعیت چندوجهی مبتنی بر `LangGraph` در `src/graph/` شامل گره‌های بازیابی ترکیبی (`retrieve_node`)، ارزیابی خوداصلاحگر (`crag_evaluate_node`)، بازنویسی پرسش‌های مبهم (`rewrite_query_node`)، مسیریاب تطبیقی (`adaptive_router_edge`) و دو مسیر تولید پاسخ (`fast_text_table_generation_node` برای متن/جدول و `visual_vlm_generation_node` برای تحلیل بصری نمودارها و دیاگرام‌ها با سقف ۲ تصویر).

## User Review & Critical Decisions

> [!IMPORTANT]
> بر اساس مشخصات `docs/PLAN.md`، `docs/ARCHITECTURE.md`، `docs/UX_FLOWS.md` (Flows 2, 3, 4)، `docs/API.md` و `docs/TESTING.md`، نیازمندی‌های تسک ۴.۲ کاملاً شفاف و بدون ابهام هستند:

- **سطح پیچیدگی تسک:** **Medium** — تمرکز صرفاً روی ماژول‌های گراف وضعیت در `src/graph/` و مجموعه تست‌های واحد مربوطه. (نکته: فایل‌های سرور `main.py` و `app.py` طبق `docs/PLAN.md` در فازهای ۵ و ۶ پیاده‌سازی می‌شوند و در Plan Mode هیچ فایلی تغییر نمی‌کند.)
- **منطق تصمیم‌گیری CRAG (`CRAG_CONFIDENCE_THRESHOLD = 0.35`):** در گره `crag_evaluate_node`، اگر بیشینه امتیاز بازرتبه‌بندی (`max(rerank_score)`) کمتر از `0.35` باشد و `rewrite_count < 1` باشد، گراف به `rewrite_query_node` و سپس مجدداً به `retrieve_node` هدایت می‌شود (حداکثر ۱ بار بازنویسی برای جلوگیری از حلقه بی‌نهایت). در پایان ارزیابی، وضعیت `crag_status` روی `CONFIDENT` (در صورت `max_score >= 0.35`) یا `LOW_CONFIDENCE` تنظیم می‌گردد.
- **مسیریابی تطبیقی (`adaptive_router_edge`):** پس از تأیید یا تکمیل چرخه CRAG، اگر `force_visual_route=True` باشد یا حداقل یک چانک در Top-K بازیابی‌شده دارای `modality == ChunkModality.FIGURE` باشد، مسیر `VISUAL_VLM_PATH` (`visual_vlm_generation_node`) انتخاب می‌شود؛ در غیر این صورت به مسیر سریع `FAST_TEXT_TABLE_PATH` (`fast_text_table_generation_node`) هدایت می‌شود.
- **حفظ Attribution و سقف تصاویر (`max_images_per_query = 2`):** در مسیر بصری حداکثر ۲ تصویر (`vlm_images_inspected <= 2`) به `VLMService.generate_visual_reasoning` ارسال می‌شود و خروجی گراف مستقیماً قابلیت تبدیل به مدل استاندارد `MultimodalQueryResponse` (شامل `VisualCitation` با `image_url="/api/v1/crops/{filename}"`، `bounding_box` و `LatencyBreakdown`) را دارد تا در تسک ۵.۱ (`/api/v1/query`) و تسک ۶.۱ (`Chainlit UI`) بدون لایه تبدیل اضافی استفاده شود.

---

## 1. Overview & Core Concept

- **What It Does:** پرسش کاربر و پارامترهای جستجو (`top_k`, `modality_filter`, `force_visual_route`) را در قالب وضعیت تایپ‌شده `VisionGraphState` دریافت کرده، بازیابی ترکیبی (`HybridRetriever`) را اجرا می‌کند، کیفیت شواهد را با آستانه `0.35` می‌سنجد (در صورت نیاز پرسش را بازنویسی و بازیابی را تکرار می‌کند)، و بر اساس مدالیته چانک‌های بازیابی‌شده بین مسیر سریع متنی/جدولی (`< 300ms`) و مسیر استدلال بصری VLM (`~800-1200ms`) تصمیم‌گیری می‌نماید.
- **Target Audience / Downstream Consumers:** اندپوینت `/api/v1/query` در FastAPI (تسک ۵.۱)، رابط کاربری Chainlit (تسک ۶.۱)، و بنچمارک سه‌گانه ارزیابی (تسک ۷.۱).
- **Key Value:** حذف سربار رمزنگاری Base64 و مصرف حافظه KV-cache روی GPU 16GB برای پرسش‌های صرفاً متنی و جدولی، درحالی‌که برای پرسش‌های نیازمند تحلیل پیکسل‌های نمودار و دیاگرام، تصاویر برش‌خورده با دقت بالا به `Qwen2.5-VL` ارسال می‌شوند.

---

## 2. User Experience & Visual Design (Execution & Attribution Flows)

- **Flow A — مسیر سریع متنی و جدولی (`FAST_TEXT_TABLE_PATH`):**
  1. بازیابی Top-5 چانک که همگی از نوع `text` یا `table` هستند با بیشینه امتیاز `>= 0.35`.
  2. انتخاب مسیر `FAST_TEXT_TABLE_PATH` بدون بارگذاری تصویر (`vlm_images_inspected = 0`).
  3. تولید پاسخ مستند به جدول Markdown/HTML و متن، همراه با `citations` و زمان‌بندی دقیق (`retrieval_ms`, `rerank_ms`, `generation_ms`, `total_ms`).
- **Flow B — مسیر استدلال بصری (`VISUAL_VLM_PATH`):**
  1. حضور حداقل یک چانک با `modality == "figure"` در Top-5 (یا فعال بودن `force_visual_route=True`).
  2. انتخاب مسیر `VISUAL_VLM_PATH`، استخراج حداکثر ۲ مسیر تصویر یکتا از چانک‌های `figure` (یا دارای `image_path`) و فراخوانی `VLMService.generate_visual_reasoning`.
  3. ثبت تعداد تصاویر بررسی‌شده در `vlm_images_inspected` (در بازه `0..2`) و ساخت `VisualCitation` با `image_url="/api/v1/crops/{crop_name}"`.
- **Flow C — خوداصلاحگری CRAG (`rewrite_query_node`):**
  1. اگر بیشینه امتیاز بازرتبه‌بندی اولیه `< 0.35` باشد و `rewrite_count == 0`، گره `rewrite_query_node` پرسش مبهم را از طریق `VLMService.rewrite_query` به پرسش غنی از کلیدواژه‌های فنی تبدیل کرده و `rewrite_count` را به `1` افزایش می‌دهد.
  2. اجرای مجدد `retrieve_node` با پرسش بازنویسی‌شده (تجمیع زمان `retrieval_ms` و `rerank_ms`) و سپس عبور از `adaptive_router_edge`.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: ساختار `VisionGraphState` مبتنی بر `TypedDict` استاندارد LangGraph**
  - *Chosen Approach:* تعریف `VisionGraphState(TypedDict, total=False)` شامل فیلدهای ورودی (`query`, `top_k`, `modality_filter`, `force_visual_route`)، فیلدهای کنترلی CRAG (`active_query`, `rewritten_query`, `rewrite_count`, `max_rerank_score`, `crag_status`)، فیلدهای بازیابی (`retrieved_chunks`) و فیلدهای خروجی (`routing_path`, `answer`, `vlm_images_inspected`, `citations`, `retrieval_ms`, `rerank_ms`, `generation_ms`, `total_ms`).
  - *Why:* سازگاری کامل با `StateGraph(VisionGraphState)` در `langgraph>=0.2.0` و تایپینگ سخت‌گیرانه `mypy --strict`.
- **Decision 2: تزریق وابستگی (`Dependency Injection`) در سازنده گره‌ها و گراف**
  - *Chosen Approach:* پیاده‌سازی کلاس/ساختار `AdaptiveRAGWorkflow` و توابع گره‌ها به‌گونه‌ای که نمونه‌های `HybridRetriever` و `VLMService` (یا پروتکل‌های معادل در تست واحد) قابل تزریق باشند.
  - *Why:* امکان تست واحد ۱۰۰٪ ایزوله، سریع و قطعی بدون نیاز به سرورهای خارجی و همچنین استفاده آسان در FastAPI و Chainlit.
- **Decision 3: ساخت خودکار `VisualCitation` با نگاشت امن `image_path` به `/api/v1/crops/{filename}`**
  - *Chosen Approach:* تبدیل هر `ScoredChunk` به `VisualCitation` با استخراج نام فایل از `chunk.image_path` (`f"/api/v1/crops/{Path(chunk.image_path).name}"`) و پیش‌نمایش متنی خلاصه‌شده (`content_preview`).
  - *Why:* انطباق کامل با اعتبارسنج‌های `VisualCitation` در قراردادهای API و حفظ زنجیره Attribution در تمام لایه‌ها (`AGENTS.md` Rule #4).

---

## 4. Technical Architecture & Data Strategy

### Architecture & State Machine Diagram

```
                     ┌────────────────────────────────────┐
                     │        START (User Request)        │
                     │ query, top_k, modality_filter, ... │
                     └─────────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────┐
                     │           retrieve_node            │
                     │ HybridRetriever.retrieve(active_q) │
                     └─────────────────┬──────────────────┘
                                       │
                                       ▼
                     ┌────────────────────────────────────┐
                     │         crag_evaluate_node         │
                     │ Computes max_rerank_score & status │
                     └─────────────────┬──────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   │ (max_score < 0.35 & rewrite_count < 1)│ (max_score >= 0.35 OR rewrite_count >= 1)
                   ▼                                       ▼
     ┌───────────────────────────┐           ┌───────────────────────────┐
     │    rewrite_query_node     │           │   adaptive_router_edge    │
     │ VLMService.rewrite_query  │           └──────┬─────────────┬──────┘
     │ rewrite_count += 1        │                  │             │
     └─────────────┬─────────────┘    (No figure &  │             │ (Has figure chunk OR
                   │                   not forced)  │             │  force_visual_route=True)
                   └──────────────► (retrieve_node) │             │
                                                    ▼             ▼
                                     ┌───────────────────┐ ┌──────────────────────┐
                                     │ fast_text_table_  │ │ visual_vlm_          │
                                     │ generation_node   │ │ generation_node      │
                                     │ (FAST_TEXT_TABLE) │ │ (VISUAL_VLM_PATH,    │
                                     │ 0 images inspected│ │  <= 2 Base64 crops)  │
                                     └─────────┬─────────┘ └──────────┬───────────┘
                                               └──────────┬───────────┘
                                                          ▼
                                     ┌────────────────────────────────────┐
                                     │   END (MultimodalQueryResponse)    │
                                     └────────────────────────────────────┘
```

### Component & State Responsibilities

1. **State Definition:**
   - تعریف ثابت `CRAG_CONFIDENCE_THRESHOLD: Final[float] = 0.35` و `MAX_REWRITE_COUNT: Final[int] = 1`.
   - تعریف `VisionGraphState(TypedDict, total=False)` و تابع کمکی تبدیل وضعیت نهایی به `MultimodalQueryResponse` و `VisualCitation`.
2. **Graph Nodes & Edges:**
   - `retrieve_node`: فراخوانی `retriever.retrieve` روی `active_query` (یا `query`) با `top_k` و `modality_filter`، ذخیره `retrieved_chunks` و انباشت زمان‌های `retrieval_ms` و `rerank_ms`.
   - `crag_evaluate_node`: محاسبه بیشینه امتیاز بازرتبه‌بندی از `retrieved_chunks` و تعیین `crag_status` (`CONFIDENT` اگر `>= 0.35` وگرنه `LOW_CONFIDENCE`).
   - `crag_gate_edge`: یال شرطی پس از `crag_evaluate_node` که در صورت `max_rerank_score < 0.35` و `rewrite_count < 1` به `"rewrite_query"` و در غیر این صورت به مسیر مسیریاب تطبیقی هدایت می‌کند.
   - `rewrite_query_node`: فراخوانی `vlm_service.rewrite_query(query)`، به‌روزرسانی `rewritten_query`، `active_query` و افزایش `rewrite_count`.
   - `adaptive_router_edge`: بررسی `force_visual_route` و مدالیته چانک‌های داخل `retrieved_chunks`؛ بازگرداندن `RoutingPath.VISUAL_VLM_PATH` در صورت وجود حداقل یک چانک `figure` یا `force_visual_route=True`، و در غیر این صورت `RoutingPath.FAST_TEXT_TABLE_PATH`.
   - `fast_text_table_generation_node`: فراخوانی `vlm_service.generate_fast_text`، تنظیم `routing_path = RoutingPath.FAST_TEXT_TABLE_PATH`، `vlm_images_inspected = 0`، محاسبه `generation_ms` و `total_ms` و ساخت لیست `citations`.
   - `visual_vlm_generation_node`: استخراج حداکثر ۲ مسیر تصویر یکتا از چانک‌های `figure` (یا دارای `image_path` در حالت `force_visual_route`)، فراخوانی `vlm_service.generate_visual_reasoning`، تنظیم `routing_path = RoutingPath.VISUAL_VLM_PATH`، ثبت `vlm_images_inspected` (حداکثر ۲)، محاسبه `generation_ms` و `total_ms` و ساخت لیست `citations`.
3. **Workflow Builder & Runner:**
   - ساخت و کامپایل `StateGraph(VisionGraphState)` با گره‌ها و یال‌های شرطی.
   - ارائه کلاس `AdaptiveRAGWorkflow` (و تابع سازنده `build_vision_graph`) با متد `async def ainvoke_query(request: MultimodalQueryRequest) -> MultimodalQueryResponse` برای اجرای مستقیم و تایپ‌شده گراف.
4. **Unit Testing Suite:**
   - تست مسیریابی قطعی به `FAST_TEXT_TABLE_PATH` زمانی که چانک‌های بازیابی‌شده فقط `text` و `table` هستند (`vlm_images_inspected == 0`).
   - تست مسیریابی قطعی به `VISUAL_VLM_PATH` زمانی که حداقل یک چانک `figure` در Top-5 حضور دارد.
   - تست سقف حداکثر ۲ تصویر (`vlm_images_inspected == 2`) زمانی که بیش از ۲ چانک `figure` در نتایج بازیابی وجود دارد.
   - تست `force_visual_route=True` حتی در شرایطی که چانک `figure` در نتایج نباشد.
   - تست چرخه خوداصلاحگری CRAG: امتیاز اولیه `< 0.35` -> اجرای `rewrite_query_node` (`rewrite_count == 1`) -> بازیابی مجدد با امتیاز بالا -> تولید پاسخ با `rewritten_query` غیرتهی.
   - تست توقف حلقه CRAG پس از ۱ بار بازنویسی حتی اگر امتیاز باز هم `< 0.35` باقی بماند (`crag_status == CRAGStatus.LOW_CONFIDENCE`).
   - تست حفظ کامل فیلدهای Attribution (`page_number`, `bounding_box`, `modality`, `image_url`) در خروجی `MultimodalQueryResponse`.

---

## 5. Acceptance Criteria & Verification Plan

### Acceptance Criteria
- [ ] ماشین وضعیت LangGraph شامل گره‌های `retrieve_node`، `crag_evaluate_node`، `rewrite_query_node`، `fast_text_table_generation_node` و `visual_vlm_generation_node` پیاده‌سازی شده است.
- [ ] منطق CRAG در صورت `max_rerank_score < 0.35` و `rewrite_count < 1` پرسش را بازنویسی و بازیابی را دقیقاً یک بار تکرار می‌کند.
- [ ] یال `adaptive_router_edge` پرسش‌های دارای چانک `figure` یا `force_visual_route=True` را به `VISUAL_VLM_PATH` (حداکثر ۲ تصویر) و سایر پرسش‌ها را به `FAST_TEXT_TABLE_PATH` هدایت می‌کند.
- [ ] خروجی گراف تمامی فیلدهای Attribution (`page_number`, `bounding_box`, `modality`, `image_url`) و زمان‌بندی‌ها (`retrieval_ms`, `rerank_ms`, `generation_ms`, `total_ms`) را طبق قرارداد `MultimodalQueryResponse` حفظ می‌کند.
- [ ] تمامی تست‌های واحد، بررسی‌های `ruff` و تایپ‌چکینگ `mypy --strict` بدون خطا اجرا می‌شوند.
