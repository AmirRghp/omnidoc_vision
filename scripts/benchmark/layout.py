"""ReportLab layout primitives shared by the benchmark documents."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Final

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Table, TableStyle

from scripts.benchmark.content import TOTAL_PAGES

PAGE_WIDTH: Final[float] = float(A4[0])
PAGE_HEIGHT: Final[float] = float(A4[1])
MARGIN: Final[float] = 50.0
CONTENT_WIDTH: Final[float] = PAGE_WIDTH - 2 * MARGIN

FONT_REGULAR: Final[str] = "Helvetica"
FONT_BOLD: Final[str] = "Helvetica-Bold"
FONT_ITALIC: Final[str] = "Helvetica-Oblique"

INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#64748B")
BORDER = colors.HexColor("#CBD5E1")
SURFACE = colors.HexColor("#F1F5F9")
VIOLET = colors.HexColor("#8B5CF6")
CYAN = colors.HexColor("#06B6D4")
EMERALD = colors.HexColor("#10B981")
AMBER = colors.HexColor("#F59E0B")
RED = colors.HexColor("#EF4444")


def _draw_page_frame(canvas: Canvas, doc_name: str, section_label: str, page: int) -> float:
    """Draw the report header, footer, and page number; return the content top."""
    canvas.setFillColor(INK)
    canvas.setFont(FONT_BOLD, 8)
    canvas.drawString(MARGIN, PAGE_HEIGHT - MARGIN, "OMNIDOC-VISION / BENCHMARK CORPUS")

    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN, doc_name)

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.7)
    canvas.line(
        MARGIN,
        PAGE_HEIGHT - MARGIN - 10,
        PAGE_WIDTH - MARGIN,
        PAGE_HEIGHT - MARGIN - 10,
    )

    if section_label:
        canvas.setFillColor(VIOLET)
        canvas.setFont(FONT_BOLD, 7.2)
        canvas.drawString(MARGIN, PAGE_HEIGHT - MARGIN - 25, section_label.upper())

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 39, PAGE_WIDTH - MARGIN, 39)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_REGULAR, 7)
    canvas.drawString(MARGIN, 27, "OmniDoc-Vision benchmark corpus")
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 27, f"Page {page} of {TOTAL_PAGES}")

    return PAGE_HEIGHT - MARGIN - 46


def _draw_title_block(
    canvas: Canvas,
    title: str,
    subtitle: str,
    meta: str,
    top_y: float,
) -> float:
    canvas.setFillColor(VIOLET)
    canvas.roundRect(MARGIN, top_y - 4, 52, 4, 1.5, fill=1, stroke=0)
    y = top_y - 25
    for line in simpleSplit(title, FONT_BOLD, 21, CONTENT_WIDTH):
        canvas.setFillColor(INK)
        canvas.setFont(FONT_BOLD, 21)
        canvas.drawString(MARGIN, y, line)
        y -= 25

    y -= 4
    y = _draw_wrapped_text(
        canvas,
        subtitle,
        MARGIN,
        y,
        CONTENT_WIDTH,
        font=FONT_REGULAR,
        size=10.2,
        leading=13.5,
        color=MUTED,
        after=5,
    )
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_REGULAR, 8)
    canvas.drawString(MARGIN, y, meta)
    return y - 18


def _draw_heading(canvas: Canvas, text: str, top_y: float, *, size: float = 13.0) -> float:
    canvas.setFillColor(INK)
    canvas.setFont(FONT_BOLD, size)
    canvas.drawString(MARGIN, top_y, text)
    return top_y - size - 9


def _draw_wrapped_text(
    canvas: Canvas,
    text: str,
    x: float,
    top_y: float,
    width: float,
    *,
    font: str = FONT_REGULAR,
    size: float = 9.5,
    leading: float = 13.0,
    color: colors.Color = INK,
    after: float = 6.0,
) -> float:
    lines = simpleSplit(text, font, size, width)
    canvas.setFillColor(color)
    canvas.setFont(font, size)
    y = top_y
    for line in lines:
        canvas.drawString(x, y, line)
        y -= leading
    return y - after


def _draw_bullets(
    canvas: Canvas,
    items: list[str],
    x: float,
    top_y: float,
    width: float,
    *,
    size: float = 9.2,
    leading: float = 12.5,
) -> float:
    y = top_y
    for item in items:
        canvas.setFillColor(CYAN)
        canvas.circle(x + 2.5, y + 3.0, 2.0, fill=1, stroke=0)
        y = _draw_wrapped_text(
            canvas,
            item,
            x + 11,
            y,
            width - 11,
            size=size,
            leading=leading,
            after=3.0,
        )
    return y


def _draw_caption(canvas: Canvas, text: str, top_y: float) -> float:
    return _draw_wrapped_text(
        canvas,
        text,
        MARGIN,
        top_y,
        CONTENT_WIDTH,
        font=FONT_ITALIC,
        size=8.2,
        leading=10.8,
        color=MUTED,
        after=8,
    )


def _draw_document_map(
    canvas: Canvas,
    top_y: float,
    rows: list[tuple[str, str]],
) -> None:
    row_height = 22.0
    height = 28.0 + row_height * len(rows)
    if top_y - height < 60:
        return

    canvas.setFillColor(SURFACE)
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.8)
    canvas.roundRect(MARGIN, top_y - height, CONTENT_WIDTH, height, 7, fill=1, stroke=1)
    canvas.setFillColor(VIOLET)
    canvas.setFont(FONT_BOLD, 7.5)
    canvas.drawString(MARGIN + 12, top_y - 16, "DOCUMENT MAP")

    y = top_y - 32
    for page_label, description in rows:
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_BOLD, 7.5)
        canvas.drawString(MARGIN + 12, y, page_label)
        canvas.setFillColor(INK)
        canvas.setFont(FONT_REGULAR, 8.2)
        canvas.drawString(MARGIN + 82, y, description)
        y -= row_height


def _draw_metric_panel(
    canvas: Canvas,
    x: float,
    top_y: float,
    width: float,
    title: str,
    items: list[tuple[str, str]],
) -> None:
    row_height = 31.0
    height = 26.0 + row_height * len(items)
    canvas.setFillColor(SURFACE)
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.8)
    canvas.roundRect(x, top_y - height, width, height, 7, fill=1, stroke=1)

    canvas.setFillColor(VIOLET)
    canvas.setFont(FONT_BOLD, 7.5)
    canvas.drawString(x + 12, top_y - 17, title.upper())

    y = top_y - 34
    for label, value in items:
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT_REGULAR, 6.8)
        canvas.drawString(x + 12, y, label.upper())
        canvas.setFillColor(INK)
        canvas.setFont(FONT_BOLD, 9.5)
        canvas.drawString(x + 12, y - 11, value)
        y -= row_height


Span = tuple[tuple[int, int], tuple[int, int]]


def _draw_table(
    canvas: Canvas,
    data: list[list[str]],
    col_widths: list[float],
    top_y: float,
    *,
    header_rows: int = 2,
    font_size: float = 6.8,
    spans: list[Span] | None = None,
    center_from: int = 2,
) -> float:
    if abs(sum(col_widths) - CONTENT_WIDTH) > 0.5:
        raise ValueError(f"Table columns must sum to {CONTENT_WIDTH}, got {sum(col_widths)}")

    table = Table(data, colWidths=col_widths)
    commands: list[tuple[Any, ...]] = [
        ("FONT", (0, 0), (-1, -1), FONT_REGULAR, font_size),
        ("FONT", (0, 0), (-1, header_rows - 1), FONT_BOLD, font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), SURFACE),
        ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (center_from, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 1.0, CYAN),
        (
            "ROWBACKGROUNDS",
            (0, header_rows),
            (-1, -1),
            [colors.white, colors.HexColor("#F8FAFC")],
        ),
    ]
    for start, end in spans or []:
        commands.append(("SPAN", start, end))

    table.setStyle(TableStyle(commands))
    _, raw_height = table.wrapOn(canvas, CONTENT_WIDTH, PAGE_HEIGHT)
    table_height = float(raw_height)
    table.drawOn(canvas, MARGIN, top_y - table_height)
    return top_y - table_height - 12


def _draw_figure(
    canvas: Canvas,
    image_buffer: io.BytesIO,
    top_y: float,
    *,
    max_height: float = 300.0,
) -> float:
    image = ImageReader(image_buffer)
    raw_width, raw_height = image.getSize()
    image_width = float(raw_width)
    image_height = float(raw_height)
    height = CONTENT_WIDTH * image_height / image_width
    width = CONTENT_WIDTH
    if height > max_height:
        height = max_height
        width = height * image_width / image_height
    x = MARGIN + (CONTENT_WIDTH - width) / 2
    canvas.drawImage(
        image,
        x,
        top_y - height,
        width=width,
        height=height,
        preserveAspectRatio=True,
        mask="auto",
    )
    return top_y - height - 8


def _draw_intro_page(
    canvas: Canvas,
    doc_name: str,
    title: str,
    subtitle: str,
    meta: str,
    summary: str,
    methodology: list[str],
    findings: list[str],
    metrics: list[tuple[str, str]],
    map_rows: list[tuple[str, str]],
) -> None:
    top = _draw_page_frame(canvas, doc_name, "BENCHMARK REPORT", 1)
    y = _draw_title_block(canvas, title, subtitle, meta, top)
    left_width = CONTENT_WIDTH * 0.60
    panel_x = MARGIN + CONTENT_WIDTH * 0.64
    panel_width = CONTENT_WIDTH * 0.36

    left_y = _draw_heading(canvas, "Executive Summary", y, size=12.5)
    left_y = _draw_wrapped_text(canvas, summary, MARGIN, left_y, left_width, size=9.0, leading=12.3)
    left_y = _draw_heading(canvas, "Methodology", left_y, size=12.5)
    left_y = _draw_bullets(canvas, methodology, MARGIN, left_y, left_width, size=8.8, leading=11.8)
    left_y = _draw_heading(canvas, "Key Findings", left_y, size=12.5)
    left_y = _draw_bullets(canvas, findings, MARGIN, left_y, left_width, size=8.8, leading=11.8)
    _draw_metric_panel(canvas, panel_x, y, panel_width, "AT A GLANCE", metrics)
    _draw_document_map(canvas, left_y - 4, map_rows)
    canvas.showPage()


def _new_canvas(path: Path, title: str) -> Canvas:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path), pagesize=A4, pageCompression=1)
    canvas.setTitle(title)
    canvas.setAuthor("OmniDoc-Vision Engineering Team")
    canvas.setSubject("Synthetic multimodal benchmark corpus")
    canvas.setCreator("OmniDoc-Vision benchmark generator")
    return canvas
