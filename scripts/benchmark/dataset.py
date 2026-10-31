"""Golden multilingual QA dataset for the benchmark corpus."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Final

from scripts.benchmark.content import (
    CACHE_LFU_256,
    CACHE_PDF,
    CLOUD_PDF,
    CLOUD_PEAK_THROUGHPUT,
    GPU_AWQ_BATCH8_P95,
    GPU_AWQ_BATCH8_VRAM,
    GPU_PDF,
    TOTAL_PAGES,
)

_CATEGORY_MODALITY: Final[dict[str, str]] = {
    "table": "table",
    "visual": "figure",
    "text": "text",
}


def _qa(
    qa_id: str,
    category: str,
    doc_name: str,
    page: int,
    en_question: str,
    fa_question: str,
    en_answer: str,
    fa_answer: str,
) -> dict[str, Any]:
    if category not in _CATEGORY_MODALITY:
        raise ValueError(f"Unknown question category: {category}")
    return {
        "id": qa_id,
        "category": category,
        "doc_name": doc_name,
        "expected_pages": [page],
        "expected_modalities": [_CATEGORY_MODALITY[category]],
        "question": {"en": en_question, "fa": fa_question},
        "ground_truth": {"en": en_answer, "fa": fa_answer},
    }


def build_golden_dataset() -> dict[str, Any]:
    questions = [
        _qa(
            "cloud_tbl_01",
            "table",
            CLOUD_PDF,
            2,
            "In Table 1, what is the cache-on throughput of Aurora PostgreSQL "
            "db.r7g.4xlarge at 64 threads?",
            "در جدول ۱، توان عملیاتی حالت کش‌فعال برای Aurora PostgreSQL با کلاس "
            "db.r7g.4xlarge در ۶۴ رشته چقدر است؟",
            "18,400 ops/sec",
            "۱۸٬۴۰۰ عملیات بر ثانیه",
        ),
        _qa(
            "cloud_tbl_02",
            "table",
            CLOUD_PDF,
            2,
            "How much memory does the Aurora PostgreSQL db.r7g.4xlarge instance have?",
            "حافظه نمونه Aurora PostgreSQL db.r7g.4xlarge چقدر است؟",
            "128 GB",
            "۱۲۸ گیگابایت",
        ),
        _qa(
            "cloud_tbl_03",
            "table",
            CLOUD_PDF,
            2,
            "What is the cache-on P95 latency of Cloud SQL Enterprise at 64 threads?",
            "تاخیر P95 حالت کش‌فعال برای Cloud SQL Enterprise در ۶۴ رشته چقدر است؟",
            "18.2 ms",
            "۱۸٫۲ میلی‌ثانیه",
        ),
        _qa(
            "cloud_tbl_04",
            "table",
            CLOUD_PDF,
            5,
            "What is the maximum connection count for Aurora PostgreSQL with the "
            "transaction pooler?",
            "حداکثر تعداد اتصال برای Aurora PostgreSQL با pooler تراکنشی چقدر است؟",
            "5,000 connections",
            "۵٬۰۰۰ اتصال",
        ),
        _qa(
            "cloud_fig_01",
            "visual",
            CLOUD_PDF,
            3,
            "At how many concurrent connections does Standard RDS first exceed 500 ms "
            "P99 write latency?",
            "در چه تعداد اتصال هم‌زمان، تاخیر نوشتن P99 برای Standard RDS برای اولین بار "
            "از ۵۰۰ میلی‌ثانیه عبور می‌کند؟",
            "250 concurrent connections",
            "۲۵۰ اتصال هم‌زمان",
        ),
        _qa(
            "cloud_fig_02",
            "visual",
            CLOUD_PDF,
            3,
            "What is Aurora PostgreSQL's P99 write latency at 500 concurrent connections?",
            "تاخیر نوشتن P99 برای Aurora PostgreSQL در ۵۰۰ اتصال هم‌زمان چقدر است؟",
            "44 ms",
            "۴۴ میلی‌ثانیه",
        ),
        _qa(
            "cloud_fig_03",
            "visual",
            CLOUD_PDF,
            4,
            "At what thread-pool size does Aurora PostgreSQL reach its peak read "
            "throughput, and what is that peak?",
            "Aurora PostgreSQL در چه اندازه‌ای از thread pool به اوج توان عملیاتی خواندن "
            "می‌رسد و این اوج چقدر است؟",
            "At 64 threads, with 18,400 ops/sec",
            "در ۶۴ رشته، با ۱۸٬۴۰۰ عملیات بر ثانیه",
        ),
        _qa(
            "cloud_txt_01",
            "text",
            CLOUD_PDF,
            1,
            "What read/write workload mix does the benchmark use?",
            "ترکیب بار کاری خواندن/نوشتن در این بنچمارک چیست؟",
            "70% reads and 30% writes",
            "۷۰٪ خواندن و ۳۰٪ نوشتن",
        ),
        _qa(
            "cloud_txt_02",
            "text",
            CLOUD_PDF,
            1,
            "How many repeated runs are executed for each configuration?",
            "برای هر پیکربندی چند اجرای تکراری انجام می‌شود؟",
            "10 runs per configuration",
            "۱۰ اجرا برای هر پیکربندی",
        ),
        _qa(
            "cloud_txt_03",
            "text",
            CLOUD_PDF,
            5,
            "Which managed PostgreSQL engine is recommended for sustained 64-thread "
            "write workloads?",
            "برای بارهای نوشتن پیوسته با ۶۴ رشته، کدام موتور PostgreSQL مدیریت‌شده توصیه می‌شود؟",
            "Aurora PostgreSQL",
            "Aurora PostgreSQL",
        ),
        _qa(
            "cache_tbl_01",
            "table",
            CACHE_PDF,
            3,
            "In Table 1, what is the LFU hit ratio with a 256 GB cache?",
            "در جدول ۱، نسبت اصابت (hit ratio) سیاست LFU با کش ۲۵۶ گیگابایتی چقدر است؟",
            "96.3%",
            "۹۶٫۳٪",
        ),
        _qa(
            "cache_tbl_02",
            "table",
            CACHE_PDF,
            3,
            "What is the P99 latency of the TTL eviction policy?",
            "تاخیر P99 سیاست حذف TTL چقدر است؟",
            "280 microseconds",
            "۲۸۰ میکروثانیه",
        ),
        _qa(
            "cache_tbl_03",
            "table",
            CACHE_PDF,
            5,
            "What read throughput does the six-shard cluster support?",
            "خوشه شش‌شارده چه توان عملیاتی خواندن را پشتیبانی می‌کند؟",
            "360k ops/sec",
            "۳۶۰ هزار عملیات بر ثانیه",
        ),
        _qa(
            "cache_fig_01",
            "visual",
            CACHE_PDF,
            2,
            "How many cache shards are shown in the cluster topology?",
            "در دیاگرام توپولوژی خوشه، چند شارد کش نمایش داده شده است؟",
            "6 cache shards",
            "۶ شارد کش",
        ),
        _qa(
            "cache_fig_02",
            "visual",
            CACHE_PDF,
            2,
            "What replication factor does the cluster use across availability zones?",
            "خوشه در منطقه‌های دسترس‌پذیری از چه ضریب تکثیری استفاده می‌کند؟",
            "Replication factor 2 (RF=2) across 3 AZs",
            "ضریب تکثیر ۲ (RF=2) در ۳ منطقه دسترس‌پذیری",
        ),
        _qa(
            "cache_fig_03",
            "visual",
            CACHE_PDF,
            4,
            "Which eviction policy has the highest hit ratio at a 256 GB cache size?",
            "کدام سیاست حذف در اندازه کش ۲۵۶ گیگابایت بالاترین نسبت اصابت را دارد؟",
            "LFU, at 96.3%",
            "LFU، با ۹۶٫۳٪",
        ),
        _qa(
            "cache_fig_04",
            "visual",
            CACHE_PDF,
            4,
            "What is the LRU hit ratio at a 64 GB cache size?",
            "نسبت اصابت LRU در اندازه کش ۶۴ گیگابایت چقدر است؟",
            "88.4%",
            "۸۸٫۴٪",
        ),
        _qa(
            "cache_txt_01",
            "text",
            CACHE_PDF,
            1,
            "What consistency model does the cache cluster provide?",
            "خوشه کش چه مدل سازگاری ارائه می‌دهد؟",
            "Eventual consistency with CRDT counters",
            "سازگاری نهایی (eventual consistency) با شمارنده‌های CRDT",
        ),
        _qa(
            "cache_txt_02",
            "text",
            CACHE_PDF,
            1,
            "What is the default eviction policy?",
            "سیاست حذف پیش‌فرض چیست؟",
            "LFU (least frequently used)",
            "LFU (کم‌استفاده‌ترین)",
        ),
        _qa(
            "cache_txt_03",
            "text",
            CACHE_PDF,
            1,
            "Which client protocol does the cache expose?",
            "کش چه پروتکل کلاینتی را ارائه می‌دهد؟",
            "RESP3 over TLS/mTLS",
            "RESP3 روی TLS/mTLS",
        ),
        _qa(
            "gpu_tbl_01",
            "table",
            GPU_PDF,
            3,
            "In the batch-size matrix, what is AWQ P95 latency at batch size 4?",
            "در ماتریس اندازه batch، تاخیر P95 حالت AWQ در batch size ۴ چقدر است؟",
            "390 ms",
            "۳۹۰ میلی‌ثانیه",
        ),
        _qa(
            "gpu_tbl_02",
            "table",
            GPU_PDF,
            3,
            "What throughput does FP16 achieve at batch size 8?",
            "حالت FP16 در batch size ۸ چه توان عملیاتی دارد؟",
            "2,120 tokens/sec",
            "۲٬۱۲۰ توکن بر ثانیه",
        ),
        _qa(
            "gpu_tbl_03",
            "table",
            GPU_PDF,
            5,
            "For a 24 GB GPU, what is the maximum number of concurrent requests?",
            "برای GPU با ۲۴ گیگابایت حافظه، حداکثر تعداد درخواست‌های هم‌زمان چقدر است؟",
            "64 concurrent requests",
            "۶۴ درخواست هم‌زمان",
        ),
        _qa(
            "gpu_fig_01",
            "visual",
            GPU_PDF,
            2,
            "How much VRAM does AWQ use at batch size 8?",
            "حالت AWQ در batch size ۸ چقدر VRAM مصرف می‌کند؟",
            "9.4 GB",
            "۹٫۴ گیگابایت",
        ),
        _qa(
            "gpu_fig_02",
            "visual",
            GPU_PDF,
            2,
            "What is the projected FP16 VRAM footprint at batch size 16, and does it fit "
            "a 16 GB GPU?",
            "مصرف تخمینی VRAM در حالت FP16 برای batch size ۱۶ چقدر است و آیا در GPU با "
            "۱۶ گیگابایت جا می‌شود؟",
            "19.6 GB, which exceeds the 16 GB limit (OOM)",
            "۱۹٫۶ گیگابایت، که از محدودیت ۱۶ گیگابایت فراتر می‌رود (OOM)",
        ),
        _qa(
            "gpu_fig_03",
            "visual",
            GPU_PDF,
            4,
            "At which batch size does AWQ P95 latency first exceed 500 ms?",
            "در چه batch size‌ای تاخیر P95 حالت AWQ برای اولین بار از ۵۰۰ میلی‌ثانیه عبور می‌کند؟",
            "Batch size 8 (560 ms)",
            "batch size ۸ (۵۶۰ میلی‌ثانیه)",
        ),
        _qa(
            "gpu_txt_01",
            "text",
            GPU_PDF,
            1,
            "How many measured requests follow the 50 warm-up requests?",
            "پس از ۵۰ درخواست گرم‌کردن، چند درخواست اندازه‌گیری می‌شود؟",
            "500 measured requests",
            "۵۰۰ درخواست اندازه‌گیری‌شده",
        ),
        _qa(
            "gpu_txt_02",
            "text",
            GPU_PDF,
            1,
            "What are the maximum model length and image limit per request used in the benchmark?",
            "حداکثر طول مدل و محدودیت تصویر برای هر درخواست در این بنچمارک چقدر است؟",
            "8,192 tokens and at most 2 images per request",
            "۸٬۱۹۲ توکن و حداکثر ۲ تصویر برای هر درخواست",
        ),
        _qa(
            "gpu_txt_03",
            "text",
            GPU_PDF,
            5,
            "Which serving configuration is recommended for a 16 GB VRAM budget?",
            "برای بودجه ۱۶ گیگابایت VRAM کدام پیکربندی سرویس‌دهی توصیه می‌شود؟",
            "AWQ 4-bit with a recommended batch size of 8",
            "AWQ 4-bit با batch size توصیه‌شده ۸",
        ),
        _qa(
            "gpu_txt_04",
            "text",
            GPU_PDF,
            5,
            "What happens to FP16 at batch size 16 on a 16 GB GPU?",
            "در GPU با ۱۶ گیگابایت، حالت FP16 در batch size ۱۶ چه وضعیتی پیدا می‌کند؟",
            "It exceeds 16 GB and runs out of memory (OOM)",
            "از ۱۶ گیگابایت فراتر می‌رود و با کمبود حافظه (OOM) مواجه می‌شود",
        ),
    ]

    return {
        "version": "1.0",
        "description": "Bilingual golden QA dataset for the OmniDoc-Vision benchmark PDFs.",
        "documents": [
            {
                "doc_name": CLOUD_PDF,
                "page_count": TOTAL_PAGES,
                "topics": ["managed PostgreSQL", "throughput", "P99 latency", "connection pooling"],
            },
            {
                "doc_name": CACHE_PDF,
                "page_count": TOTAL_PAGES,
                "topics": ["cache topology", "eviction policies", "replication", "capacity"],
            },
            {
                "doc_name": GPU_PDF,
                "page_count": TOTAL_PAGES,
                "topics": ["AWQ", "FP16", "VRAM", "batch latency", "GPU capacity"],
            },
        ],
        "questions": questions,
    }


def validate_dataset(dataset: dict[str, Any]) -> None:
    documents = dataset.get("documents")
    questions = dataset.get("questions")
    if not isinstance(documents, list) or len(documents) != 3:
        raise ValueError("Dataset must describe exactly three documents")
    if not isinstance(questions, list) or len(questions) != 30:
        raise ValueError("Dataset must contain exactly 30 questions")

    page_counts: dict[str, int] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each document entry must be an object")
        doc_name = document.get("doc_name")
        page_count = document.get("page_count")
        if not isinstance(doc_name, str) or not doc_name:
            raise ValueError("Document entries require a non-empty doc_name")
        if page_count != TOTAL_PAGES:
            raise ValueError(f"{doc_name} must have {TOTAL_PAGES} pages")
        page_counts[doc_name] = page_count

    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object")
        qa_id = question.get("id")
        category = question.get("category")
        doc_name = question.get("doc_name")
        if not isinstance(qa_id, str) or not qa_id or qa_id in seen_ids:
            raise ValueError(f"Question IDs must be unique and non-empty: {qa_id!r}")
        seen_ids.add(qa_id)
        if category not in _CATEGORY_MODALITY:
            raise ValueError(f"Invalid category for {qa_id}: {category!r}")
        counts[category] += 1
        if doc_name not in page_counts:
            raise ValueError(f"Unknown document for {qa_id}: {doc_name!r}")
        expected_pages = question.get("expected_pages")
        if (
            not isinstance(expected_pages, list)
            or not expected_pages
            or not all(
                isinstance(page, int) and 1 <= page <= page_counts[str(doc_name)]
                for page in expected_pages
            )
        ):
            raise ValueError(f"Invalid expected_pages for {qa_id}")
        expected_modalities = question.get("expected_modalities")
        if expected_modalities != [_CATEGORY_MODALITY[str(category)]]:
            raise ValueError(f"Modality mismatch for {qa_id}")
        for field in ("question", "ground_truth"):
            text = question.get(field)
            if not isinstance(text, dict):
                raise ValueError(f"{qa_id} requires a bilingual {field} object")
            for language in ("en", "fa"):
                value = text.get(language)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{qa_id} has an empty {field}.{language}")

    if counts != Counter({"table": 10, "visual": 10, "text": 10}):
        raise ValueError(f"Unexpected category distribution: {dict(counts)}")

    answers = {
        question["id"]: question["ground_truth"]["en"]
        for question in questions
        if isinstance(question, dict)
    }
    expected_fragments = {
        "cloud_tbl_01": f"{CLOUD_PEAK_THROUGHPUT:,} ops/sec",
        "cloud_fig_03": f"{CLOUD_PEAK_THROUGHPUT:,} ops/sec",
        "cache_tbl_01": f"{CACHE_LFU_256:.1f}%",
        "gpu_fig_01": f"{GPU_AWQ_BATCH8_VRAM:.1f} GB",
        "gpu_fig_03": f"Batch size 8 ({GPU_AWQ_BATCH8_P95} ms)",
        "gpu_txt_03": "recommended batch size of 8",
    }
    for qa_id, fragment in expected_fragments.items():
        if fragment not in answers.get(qa_id, ""):
            raise ValueError(f"{qa_id} does not match benchmark data: {fragment!r}")
    if "maximum batch size" in answers.get("gpu_txt_03", ""):
        raise ValueError("gpu_txt_03 must describe the recommendation, not a hard maximum")


def write_dataset(dataset: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
