"""OmniDoc-Vision — Interactive Chainlit Multimodal Console (Task 6.1)."""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

import chainlit as cl

from src.common.config import PROJECT_ROOT, Settings, get_settings
from src.common.logger import get_logger
from src.graph.workflow import AdaptiveRAGWorkflow
from src.ingestion.docling_parser import DoclingParser
from src.ingestion.indexer import HybridIndexer
from src.ingestion.visual_captioner import VisualCaptioner
from src.schemas.api import (
    CRAGStatus,
    MultimodalQueryResponse,
    RoutingPath,
    VisualCitation,
)
from src.schemas.document import ChunkModality

logger = get_logger(__name__)


def build_telemetry_badge(response: MultimodalQueryResponse) -> str:
    """Format high-contrast telemetry badge and routing indicators in HTML."""
    is_visual = response.routing_path == RoutingPath.VISUAL_VLM_PATH
    crag_class = "confident" if response.crag_status == CRAGStatus.CONFIDENT else "low-confidence"

    if is_visual:
        pill_class = "route-pill visual-path"
        route_label = "👁️ Route: Visual VLM Reasoning (Qwen2.5-VL)"
        extra_metrics = (
            f'<span class="telemetry-metric">🖼️ {response.vlm_images_inspected} تصویر</span>'
        )
    else:
        pill_class = "route-pill fast-path"
        route_label = "⚡ Route: Fast Text/Table Path"
        extra_metrics = (
            f'<span class="telemetry-metric">🔍 بازیابی: '
            f"{response.latency_ms.retrieval_ms:.1f}ms</span>"
            f'<span class="telemetry-metric">📊 رتبه‌بندی: '
            f"{response.latency_ms.rerank_ms:.1f}ms</span>"
        )

    telemetry_html = f"""<div class="telemetry-container">
  <span class="{pill_class}">{route_label}</span>
  <span class="telemetry-metric">⏱️ کل: {response.latency_ms.total_ms:.1f}ms</span>
  {extra_metrics}
  <span class="crag-badge {crag_class}">🎯 CRAG: {response.crag_status.value}</span>
</div>"""
    return telemetry_html


def resolve_crop_paths(
    citations: Sequence[VisualCitation],
    crops_dir: Path,
) -> list[tuple[Path, str]]:
    """Locate existing crop files safely for figure citations."""
    resolved_crops: list[tuple[Path, str]] = []
    resolved_crops_dir = crops_dir.resolve()

    for citation in citations:
        if citation.modality != ChunkModality.FIGURE:
            continue

        raw_url = citation.image_url or ""
        filename = Path(raw_url).name if raw_url else f"{citation.chunk_id}.png"

        # Traversal guard
        if "/" in filename or "\\" in filename or ".." in filename:
            continue

        candidate = (resolved_crops_dir / filename).resolve()
        if candidate.is_file() and candidate.is_relative_to(resolved_crops_dir):
            label = f"شکل: صفحه {citation.page_number} ({citation.chunk_id})"
            resolved_crops.append((candidate, label))
        else:
            # Check relative to PROJECT_ROOT
            alt_candidate = (PROJECT_ROOT / "data" / "crops" / filename).resolve()
            if alt_candidate.is_file():
                label = f"شکل: صفحه {citation.page_number} ({citation.chunk_id})"
                resolved_crops.append((alt_candidate, label))

    return resolved_crops


def build_citations_markdown(citations: Sequence[VisualCitation]) -> str:
    """Format multimodal visual citations and reconstructed table cards into Markdown."""
    if not citations:
        return ""

    cards: list[str] = ["### 📚 مستندات و شواهد استنادشده (Grounding Evidence)"]

    for idx, cit in enumerate(citations, 1):
        bbox = cit.bounding_box
        bbox_str = f"[{bbox.x0:.3f}, {bbox.y0:.3f}, {bbox.x1:.3f}, {bbox.y1:.3f}]"
        modality_icon = "📊 جدول" if cit.modality == ChunkModality.TABLE else "🖼️ تصویر/نمودار"
        if cit.modality == ChunkModality.TEXT:
            modality_icon = "📄 متن"

        section = f" — *{html.escape(cit.section_title)}*" if cit.section_title else ""
        escaped_doc = html.escape(cit.doc_name)

        summary_header = (
            f"<summary><b>{idx}. {modality_icon}</b>: "
            f"<code>{escaped_doc}</code> | صفحه {cit.page_number}{section}</summary>"
        )

        card_lines = [
            "<details class='citation-card'>",
            summary_header,
            "",
            f"- **شناسه چانک:** `line_{cit.chunk_id}` | **امتیاز تطابق:** `{cit.rerank_score:.3f}`",
            f"- **مختصات Bounding Box:** `<span class='bbox-badge'>{bbox_str}</span>`",
            "",
            "> **محتوای استخراج‌شده:**",
            f"> {cit.content_preview}",
            "</details>",
        ]
        cards.append("\n".join(card_lines))

    return "\n\n".join(cards)


def get_starters() -> list[cl.Starter]:
    """Provide realistic sample benchmark questions on initial load."""
    return [
        cl.Starter(
            label="تحلیل نمودار P99 (مسیر بینایی)",
            message=(
                "بر اساس نمودار تاخیر P99 در صفحه ۳، از چه تعداد اتصالی خطای سیستم جهش می‌کند؟"
            ),
        ),
        cl.Starter(
            label="مقایسه توان عملیاتی دیتابیس (مسیر سریع جدول)",
            message=(
                "در جدول مقایسه دیتابیس‌ها، توان عملیاتی Qdrant در حالت "
                "کوانتیزاسیون Scalar چقدر است و چقدر رم مصرف می‌کند؟"
            ),
        ),
        cl.Starter(
            label="تحلیل معماری کش توزیع‌شده (مسیر دیاگرام)",
            message=(
                "نمودار معماری خوشه و توپولوژی کش توزیع‌شده را تحلیل کن و نقش هر نود را توضیح بده."
            ),
        ),
    ]


@cl.set_starters
async def set_starters(user: cl.User | None = None) -> list[cl.Starter]:
    """Register Chainlit starter cards callback."""
    del user
    return get_starters()


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize workflow and user session state on application startup."""
    settings = get_settings()
    workflow = AdaptiveRAGWorkflow(settings=settings)

    cl.user_session.set("settings", settings)
    cl.user_session.set("workflow", workflow)

    welcome_message = (
        "## 👁️ خوش آمدید به **OmniDoc-Vision**\n\n"
        "**سامانه هوشمند استخراج و بازیابی چندوجهی اسناد مهندسی و فنی**\n\n"
        "این سیستم پرسش‌های شما را با تحلیل ترکیبی **متون**، **جداول ساختاریافته** و "
        "**نمودارهای گرافیکی** پاسخ داده و مدارک را با دقت مختصات Bounding Box ارجاع می‌دهد:\n\n"
        "- ⚡ **مسیر سریع متن و جدول:** پاسخ‌دهی زیر ۳۰۰ میلی‌ثانیه برای پرسش‌های متنی و جداول.\n"
        "- 👁️ **مسیر استدلال بینایی:** ارسال کراپ نمودارها به مدل `Qwen2.5-VL` "
        "جهت تحلیل پیکسل‌ها.\n\n"
        "💡 *می‌توانید یکی از پرسش‌های پیشنهادی را انتخاب کنید یا سند PDF جدیدی ارسال نمایید.*"
    )
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user queries and uploaded PDF documents."""
    settings: Settings = cl.user_session.get("settings") or get_settings()
    workflow: AdaptiveRAGWorkflow = cl.user_session.get("workflow") or AdaptiveRAGWorkflow(
        settings=settings
    )

    # 1. Handle uploaded PDFs if present
    if message.elements:
        for element in message.elements:
            is_pdf = (
                isinstance(element, cl.File)
                and element.path is not None
                and element.path.lower().endswith(".pdf")
            )
            if is_pdf and element.path is not None:
                pdf_path = Path(element.path)
                status_msg = await cl.Message(
                    content=f"⏳ در حال پردازش و استخراج سند چندوجهی: `line_{pdf_path.name}`..."
                ).send()

                try:
                    parser = DoclingParser(settings=settings, crops_dir=settings.crops_dir)
                    captioner = VisualCaptioner(settings=settings)
                    indexer = HybridIndexer(settings=settings)

                    chunks = await parser.aparse_pdf(pdf_path)
                    chunks = await captioner.caption_chunks(chunks)
                    stats = await indexer.index_chunks(chunks)

                    await status_msg.update(
                        content=(
                            f"✅ سند `{pdf_path.name}` با موفقیت پردازش و ایندکس شد!\n"
                            f"- کل چانک‌های ایجادشده: **{stats.total_chunks}**\n"
                            f"- ایندکس در وکتور استور Qdrant و ایندکس محلی BM25 انجام گرفت."
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to process uploaded PDF %s: %s",
                        pdf_path,
                        exc,
                        exc_info=True,
                    )
                    await status_msg.update(
                        content=f"❌ خطا در پردازش سند PDF: `{exc}`"
                    )

    # 2. Process query text
    query_text = (message.content or "").strip()
    if not query_text:
        return

    async with cl.Step(name="Hybrid Retrieval & Adaptive Routing") as step:
        step.input = query_text
        response = await workflow.run_query(query_text)
        step.output = (
            f"مسیر: {response.routing_path.value} | "
            f"زمان کل: {response.latency_ms.total_ms:.1f}ms"
        )

    # 3. Format visual telemetry badge & citations
    telemetry_html = build_telemetry_badge(response)
    citations_md = build_citations_markdown(response.citations)

    full_response_text = f"""{telemetry_html}

<div class="persian-content">

{response.answer}

</div>

{citations_md}"""

    # 4. Resolve and attach inline crop images
    crop_entries = resolve_crop_paths(response.citations, settings.crops_dir)
    image_elements: list[cl.Image] = [
        cl.Image(
            path=str(crop_path),
            name=label,
            display="inline",
            size="large",
        )
        for crop_path, label in crop_entries
    ]

    await cl.Message(content=full_response_text, elements=image_elements).send()
