"""Generate the self-contained OmniDoc-Vision benchmark corpus.

The script creates three deterministic, five-page engineering PDFs and a
30-question bilingual golden dataset. It uses ReportLab for document layout
and Matplotlib for charts and topology diagrams; no network or GPU is needed.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Final

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Table, TableStyle

matplotlib.use("Agg", force=True)
rl_config.invariant = 1

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RAW_PDF_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw_pdfs"
DATASET_PATH: Final[Path] = PROJECT_ROOT / "data" / "golden_multimodal_dataset.json"

CLOUD_PDF: Final[str] = "cloud_database_benchmark_2026.pdf"
CACHE_PDF: Final[str] = "distributed_cache_architecture.pdf"
GPU_PDF: Final[str] = "gpu_inference_capacity_report.pdf"

PAGE_WIDTH: Final[float] = float(A4[0])
PAGE_HEIGHT: Final[float] = float(A4[1])
MARGIN: Final[float] = 50.0
CONTENT_WIDTH: Final[float] = PAGE_WIDTH - 2 * MARGIN
TOTAL_PAGES: Final[int] = 5

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

MPL_INK = "#0F172A"
MPL_MUTED = "#64748B"
MPL_GRID = "#CBD5E1"
MPL_VIOLET = "#8B5CF6"
MPL_CYAN = "#06B6D4"
MPL_EMERALD = "#10B981"
MPL_AMBER = "#F59E0B"
MPL_RED = "#EF4444"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#334155",
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


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


def _style_axes(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)


def _figure_bytes(fig: Figure) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "OmniDoc-Vision benchmark generator"},
    )
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _latency_figure() -> io.BytesIO:
    connections = [50, 100, 150, 200, 250, 300, 400, 500]
    aurora = [22, 23, 24, 26, 31, 35, 38, 44]
    rds = [38, 42, 46, 52, 620, 780, 910, 1040]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.plot(
        connections,
        aurora,
        color=MPL_CYAN,
        marker="o",
        linewidth=2.4,
        markeredgecolor="white",
        label="Aurora PostgreSQL",
    )
    ax.plot(
        connections,
        rds,
        color=MPL_RED,
        marker="s",
        linewidth=2.4,
        markeredgecolor="white",
        label="Standard RDS",
    )
    ax.axhline(500, color=MPL_AMBER, linewidth=1.4, linestyle="--")
    ax.text(55, 535, "500 ms P99 SLO", color="#B45309", fontsize=8.5, fontweight="bold")
    ax.annotate(
        "250 connections\n620 ms",
        xy=(250, 620),
        xytext=(300, 360),
        fontsize=8.5,
        color=MPL_RED,
        arrowprops={"arrowstyle": "->", "color": MPL_RED, "lw": 1.1},
    )
    ax.annotate(
        "500 connections\n44 ms",
        xy=(500, 44),
        xytext=(350, 120),
        fontsize=8.5,
        color=MPL_CYAN,
        arrowprops={"arrowstyle": "->", "color": MPL_CYAN, "lw": 1.1},
    )
    ax.set_title("Figure 1 - P99 Write Latency vs Concurrent Connections", loc="left")
    ax.set_xlabel("Concurrent connections")
    ax.set_ylabel("P99 write latency (ms)")
    ax.set_xticks(connections)
    ax.set_ylim(0, 1150)
    ax.legend(loc="upper left")
    _style_axes(ax)
    return _figure_bytes(fig)


def _throughput_figure() -> io.BytesIO:
    threads = [8, 16, 32, 64, 128]
    aurora = [5600, 9800, 14200, 18400, 18100]
    cloud_sql = [4800, 8200, 12100, 14900, 14750]
    rds = [3500, 6100, 8900, 10200, 10150]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.plot(threads, aurora, color=MPL_CYAN, marker="o", linewidth=2.3, label="Aurora PostgreSQL")
    ax.plot(threads, cloud_sql, color=MPL_VIOLET, marker="^", linewidth=2.3, label="Cloud SQL")
    ax.plot(threads, rds, color=MPL_RED, marker="s", linewidth=2.3, label="Standard RDS")
    ax.annotate(
        "Peak 18,400 ops/s\nat 64 threads",
        xy=(64, 18400),
        xytext=(72, 16000),
        fontsize=8.5,
        color=MPL_CYAN,
        arrowprops={"arrowstyle": "->", "color": MPL_CYAN, "lw": 1.1},
    )
    ax.set_title("Figure 2 - Read Throughput vs Thread Pool Size", loc="left")
    ax.set_xlabel("Thread pool size")
    ax.set_ylabel("Read throughput (ops/s)")
    ax.set_xticks(threads)
    ax.set_ylim(0, 20500)
    ax.legend(loc="lower right")
    _style_axes(ax)
    return _figure_bytes(fig)


def _topology_figure() -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(12.0, 6.6), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.set_title(
        "Figure 1 - Distributed Cache Cluster Topology (6 shards, RF=2, 3 AZs)",
        loc="left",
    )

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        subtitle: str,
        face: str,
        edge: str,
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                linewidth=1.3,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(
            x + width / 2,
            y + height * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=MPL_INK,
        )
        if subtitle:
            ax.text(
                x + width / 2,
                y + height * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=6.8,
                color=MPL_MUTED,
            )

    band_colors = ["#F5F3FF", "#ECFEFF", "#ECFDF5"]
    for index, label in enumerate(["AZ-A", "AZ-B", "AZ-C"]):
        band_x = 4.95 + index * 2.35
        ax.add_patch(
            FancyBboxPatch(
                (band_x, 0.85),
                2.25,
                4.65,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=0.9,
                edgecolor="#DDD6FE",
                facecolor=band_colors[index],
                linestyle="--",
            )
        )
        ax.text(
            band_x + 1.125,
            5.68,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=MPL_MUTED,
        )

    box(0.15, 2.55, 1.45, 0.90, "Client SDK", "RESP3 over TLS", "#F8FAFC", MPL_CYAN)
    box(1.80, 2.55, 1.45, 0.90, "Global LB", "L7 / least latency", "#F8FAFC", MPL_CYAN)
    box(3.45, 2.40, 1.30, 1.20, "Shard Routers", "3 instances", "#F5F3FF", MPL_VIOLET)

    for az in range(3):
        for row in range(2):
            shard_number = az * 2 + row + 1
            box(
                5.10 + az * 2.35,
                3.45 - row * 1.30,
                1.95,
                0.72,
                f"Shard {shard_number}",
                "primary + replica",
                "white",
                MPL_EMERALD if row == 0 else MPL_VIOLET,
            )

    box(
        4.95,
        0.12,
        4.35,
        0.50,
        "Sentinel / Control Plane",
        "health checks and failover",
        "white",
        MPL_AMBER,
    )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        color: str,
        *,
        dashed: bool = False,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color=color,
                linestyle="--" if dashed else "-",
            )
        )

    arrow((1.60, 3.00), (1.80, 3.00), MPL_CYAN)
    arrow((3.25, 3.00), (3.45, 3.00), MPL_CYAN)
    arrow((4.75, 3.60), (4.95, 4.60), MPL_VIOLET)
    ax.plot([4.95, 11.80], [4.60, 4.60], color=MPL_VIOLET, linewidth=1.0)
    for az in range(3):
        center = 6.075 + az * 2.35
        arrow((center, 4.60), (center, 4.20), MPL_VIOLET)
    ax.text(
        5.05,
        0.78,
        "RF=2: every shard has one replica in the next availability zone",
        fontsize=7.0,
        color=MPL_MUTED,
        ha="left",
    )
    return _figure_bytes(fig)


def _hit_ratio_figure() -> io.BytesIO:
    cache_sizes = [16, 32, 64, 128, 256, 512]
    lru = [79.5, 84.2, 88.4, 91.6, 94.1, 95.4]
    lfu = [82.1, 87.4, 91.2, 94.0, 96.3, 97.1]
    ttl = [71.8, 77.6, 82.7, 86.4, 89.5, 91.2]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax.plot(cache_sizes, lru, color=MPL_CYAN, marker="o", linewidth=2.3, label="LRU")
    ax.plot(cache_sizes, lfu, color=MPL_VIOLET, marker="^", linewidth=2.3, label="LFU")
    ax.plot(cache_sizes, ttl, color=MPL_AMBER, marker="s", linewidth=2.3, label="TTL")
    ax.annotate(
        "96.3% at 256 GB",
        xy=(256, 96.3),
        xytext=(300, 91.0),
        fontsize=8.5,
        color=MPL_VIOLET,
        arrowprops={"arrowstyle": "->", "color": MPL_VIOLET, "lw": 1.1},
    )
    ax.set_title("Figure 2 - Hit Ratio vs Cache Size by Eviction Policy", loc="left")
    ax.set_xlabel("Cache size (GB)")
    ax.set_ylabel("Hit ratio (%)")
    ax.set_xticks(cache_sizes)
    ax.set_ylim(65, 100)
    ax.legend(loc="lower right")
    _style_axes(ax)
    return _figure_bytes(fig)


def _vram_figure() -> io.BytesIO:
    batch_sizes = ["Batch 1", "Batch 4", "Batch 8", "Batch 16"]
    awq = [5.2, 6.8, 9.4, 13.1]
    fp16 = [14.2, 15.1, 15.6, 19.6]

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    positions = list(range(len(batch_sizes)))
    width = 0.36
    bars_awq = ax.bar(
        [p - width / 2 for p in positions],
        awq,
        width,
        color=MPL_VIOLET,
        label="AWQ 4-bit",
    )
    bars_fp16 = ax.bar(
        [p + width / 2 for p in positions],
        fp16,
        width,
        color=MPL_AMBER,
        label="FP16",
    )
    ax.axhline(16, color=MPL_RED, linewidth=1.5, linestyle="--")
    ax.text(2.55, 16.35, "16 GB GPU limit", color=MPL_RED, fontsize=8.5, fontweight="bold")
    ax.bar_label(bars_awq, fmt="%.1f", fontsize=7.5, padding=2)
    ax.bar_label(bars_fp16, fmt="%.1f", fontsize=7.5, padding=2)
    ax.annotate(
        "OOM",
        xy=(3 + width / 2, 19.0),
        xytext=(3 + width / 2, 21.0),
        ha="center",
        fontsize=8.5,
        color=MPL_RED,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": MPL_RED, "lw": 1.0},
    )
    ax.set_title("Figure 1 - VRAM Footprint: AWQ vs FP16", loc="left")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("VRAM (GB)")
    ax.set_xticks(positions)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylim(0, 23)
    ax.legend(loc="upper left")
    _style_axes(ax)
    return _figure_bytes(fig)


def _batch_scaling_figure() -> io.BytesIO:
    batches = [1, 2, 4, 8, 16]
    throughput = [820, 1540, 2680, 4100, 5200]
    p95 = [265, 310, 390, 560, 920]

    fig, ax_left = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    line_throughput = ax_left.plot(
        batches,
        throughput,
        color=MPL_VIOLET,
        marker="o",
        linewidth=2.4,
        label="AWQ throughput",
    )
    ax_left.set_xlabel("Batch size")
    ax_left.set_ylabel("Throughput (tokens/s)", color=MPL_VIOLET)
    ax_left.tick_params(axis="y", labelcolor=MPL_VIOLET)
    ax_left.set_ylim(0, 6000)
    ax_left.set_xticks(batches)

    ax_right = ax_left.twinx()
    line_latency = ax_right.plot(
        batches,
        p95,
        color=MPL_AMBER,
        marker="s",
        linewidth=2.4,
        label="AWQ P95 latency",
    )
    ax_right.set_ylabel("P95 latency (ms)", color=MPL_AMBER)
    ax_right.tick_params(axis="y", labelcolor=MPL_AMBER)
    ax_right.set_ylim(0, 1100)
    ax_right.axhline(500, color=MPL_RED, linewidth=1.3, linestyle="--")
    ax_right.annotate(
        "Batch 8: 560 ms",
        xy=(8, 560),
        xytext=(8.3, 700),
        fontsize=8.5,
        color=MPL_AMBER,
        arrowprops={"arrowstyle": "->", "color": MPL_AMBER, "lw": 1.0},
    )
    ax_left.set_title("Figure 2 - Throughput and P95 Latency vs Batch Size", loc="left")
    ax_left.legend(
        [line_throughput[0], line_latency[0]],
        ["AWQ throughput", "AWQ P95 latency"],
        loc="upper left",
    )
    _style_axes(ax_left)
    ax_right.grid(False)
    return _figure_bytes(fig)


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


CLOUD_TABLE_1: Final[list[list[str]]] = [
    [
        "Engine",
        "Instance Class",
        "Compute",
        "",
        "",
        "Throughput @ 64 threads (ops/s)",
        "",
        "P95 Latency (ms)",
        "",
    ],
    [
        "",
        "",
        "vCPU",
        "Mem. (GB)",
        "Cost/hr (USD)",
        "Cache On",
        "Cache Off",
        "Cache On",
        "Cache Off",
    ],
    [
        "Aurora PostgreSQL",
        "db.r7g.4xlarge",
        "16",
        "128",
        "2.32",
        "18,400",
        "11,200",
        "12.4",
        "28.1",
    ],
    [
        "",
        "db.r7g.2xlarge",
        "8",
        "64",
        "1.16",
        "11,200",
        "6,800",
        "11.8",
        "30.5",
    ],
    [
        "Cloud SQL Enterprise",
        "16 vCPU / 128 GB",
        "16",
        "128",
        "2.05",
        "14,900",
        "9,800",
        "18.2",
        "41.7",
    ],
    [
        "Standard RDS",
        "db.r7g.4xlarge",
        "16",
        "128",
        "1.94",
        "10,200",
        "7,100",
        "31.0",
        "88.4",
    ],
]

CLOUD_TABLE_2: Final[list[list[str]]] = [
    [
        "Engine",
        "Connection Pooling",
        "",
        "",
        "Storage",
        "",
        "Monthly Cost (USD)",
    ],
    [
        "",
        "Pool Mode",
        "Max Connections",
        "Pool Size / Node",
        "Storage Type",
        "Provisioned IOPS",
        "",
    ],
    [
        "Aurora PostgreSQL",
        "PgBouncer transaction",
        "5,000",
        "800",
        "Aurora I/O-Optimized",
        "64,000",
        "$1,680",
    ],
    [
        "Cloud SQL Enterprise",
        "Built-in connector",
        "4,000",
        "600",
        "Hyperdisk Balanced",
        "24,000",
        "$1,420",
    ],
    [
        "Standard RDS",
        "PgBouncer session",
        "2,500",
        "400",
        "gp3",
        "12,000",
        "$1,150",
    ],
]

CACHE_TABLE_1: Final[list[list[str]]] = [
    ["Policy", "Hit Ratio", "", "Memory Overhead", "P99 Latency (us)", "Recommended Workload"],
    ["", "64 GB cache", "256 GB cache", "", "", ""],
    ["LRU", "88.4%", "94.1%", "3.2%", "410", "Mixed read/write"],
    ["LFU", "91.2%", "96.3%", "4.1%", "430", "Stable hot keys"],
    ["TTL", "82.7%", "89.5%", "1.8%", "280", "Session data"],
    ["Random", "74.3%", "81.2%", "0.9%", "190", "Write-heavy ephemeral"],
]

CACHE_TABLE_2: Final[list[list[str]]] = [
    [
        "Shards",
        "RF",
        "vCPU / Node",
        "Memory/Node (GB)",
        "Throughput (k ops/s)",
        "",
        "Failover (s)",
        "",
    ],
    ["", "", "", "", "Reads", "Writes", "P50", "P99"],
    ["3", "2", "8", "64", "180", "72", "2.4", "8.1"],
    ["6", "2", "8", "128", "360", "144", "2.1", "7.4"],
    ["12", "2", "16", "256", "720", "288", "1.8", "6.2"],
]

GPU_TABLE_1: Final[list[list[str]]] = [
    ["Batch Size", "AWQ (4-bit)", "", "", "FP16", "", ""],
    ["", "P50 (ms)", "P95 (ms)", "Tokens/s", "P50 (ms)", "P95 (ms)", "Tokens/s"],
    ["1", "210", "265", "820", "340", "410", "510"],
    ["2", "245", "310", "1,540", "410", "500", "920"],
    ["4", "310", "390", "2,680", "560", "680", "1,480"],
    ["8", "455", "560", "4,100", "880", "1,050", "2,120"],
    ["16", "720", "920", "5,200", "OOM", "OOM", "OOM"],
]

GPU_TABLE_2: Final[list[list[str]]] = [
    [
        "GPU Memory (GB)",
        "Recommended Serving Envelope",
        "",
        "",
        "",
        "Exp. P95 (ms)",
        "Fit",
    ],
    [
        "",
        "Max AWQ Batch",
        "KV Cache Budget (GB)",
        "Max Concurrent Requests",
        "Max Model Len",
        "",
        "",
    ],
    ["16 GB", "8", "4.2", "32", "8,192", "560", "Yes (AWQ only)"],
    ["24 GB", "16", "9.8", "64", "8,192", "920", "Yes"],
    ["48 GB", "32", "22.0", "128", "16,384", "700", "Yes"],
]


def build_cloud_database_pdf(path: Path) -> None:
    canvas = _new_canvas(path, "Cloud Database Benchmark 2026")
    _draw_intro_page(
        canvas,
        CLOUD_PDF,
        "Cloud Database Benchmark 2026",
        "Managed PostgreSQL throughput, memory, and P99 latency under sustained "
        "connection concurrency.",
        "OmniDoc-Vision Benchmark Corpus | Engineering Report 2026 | 5 pages",
        "This report benchmarks three managed PostgreSQL deployments under a reproducible "
        "OLTP workload: Aurora PostgreSQL, Cloud SQL Enterprise, and Standard RDS. The study "
        "isolates cache state, connection pooling, and thread-pool scaling to quantify the "
        "latency cliff that appears when connection concurrency exceeds the provisioned envelope.",
        [
            "Workload: 70% reads and 30% writes, with 1 KB rows and 64 KB random I/O.",
            "Each configuration was executed 10 times after warm-up; reported values are medians.",
            "Cache On uses a 64 GB warm buffer pool; Cache Off starts from a flushed state.",
            "Client concurrency ranges from 50 to 500 connections across three availability zones.",
        ],
        [
            "Aurora PostgreSQL sustains 18,400 ops/sec at 64 threads with the cache enabled.",
            "Standard RDS jumps from 52 ms at 200 connections to 620 ms at 250 connections.",
            "Thread-pool scaling plateaus at 64 threads for every engine.",
        ],
        [
            ("BENCHMARK MIX", "70% reads / 30% writes"),
            ("REPEATED RUNS", "10 per configuration"),
            ("PEAK THROUGHPUT", "18,400 ops/sec"),
            ("LATENCY CLIFF", "250 connections"),
        ],
        [
            ("PAGE 2", "Table 1 - throughput and memory profile"),
            ("PAGE 3", "Figure 1 - P99 latency vs concurrency"),
            ("PAGE 4", "Figure 2 - read throughput vs thread count"),
            ("PAGE 5", "Table 2 - pooling and storage configuration"),
        ],
    )

    top = _draw_page_frame(canvas, CLOUD_PDF, "SECTION 02", 2)
    y = _draw_heading(canvas, "2. Throughput and Memory Profile", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Table 1 consolidates compute, cache, throughput, and P95 latency measurements at a "
        "fixed 64-thread pool. Grouped headers preserve the engine-level comparison while "
        "keeping numeric columns aligned for downstream table reconstruction.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_caption(
        canvas,
        "Table 1 - Managed PostgreSQL throughput and memory profile at 64 threads.",
        y,
    )
    y = _draw_table(
        canvas,
        CLOUD_TABLE_1,
        [88, 60, 36, 46, 56, 63, 63, 42, 41],
        y,
        header_rows=2,
        font_size=6.8,
        spans=[
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (4, 0)),
            ((5, 0), (6, 0)),
            ((7, 0), (8, 0)),
            ((0, 2), (0, 3)),
        ],
        center_from=2,
    )
    y = _draw_wrapped_text(
        canvas,
        "Cache On = warm 64 GB buffer pool; Cache Off = cold start after a full buffer flush. "
        "P95 latency is measured at the same 64-thread operating point.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        font=FONT_ITALIC,
        size=8.0,
        leading=10.5,
        color=MUTED,
    )
    y = _draw_heading(canvas, "2.1 Observations", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "Aurora 4xlarge delivers 80% more cache-on throughput than Standard RDS at the "
            "same vCPU and memory footprint.",
            "Cache-off throughput drops by 39% for Aurora and 34% for Cloud SQL, showing the "
            "cost of a cold buffer pool.",
            "Cloud SQL stays below 19 ms P95 with cache on, but rises to 41.7 ms after a flush.",
            "Aurora 2xlarge still reaches 11,200 ops/sec, making it a viable mid-size tier.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, CLOUD_PDF, "SECTION 03", 3)
    y = _draw_heading(canvas, "3. P99 Latency Under Concurrency", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Figure 1 plots P99 write latency against concurrent connections. The Standard RDS "
        "curve is stable through 200 connections, then enters a sharp saturation regime at "
        "250 connections where queueing dominates.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_figure(canvas, _latency_figure(), y, max_height=300)
    y = _draw_caption(
        canvas,
        "Figure 1 - P99 write latency (ms) versus concurrent connections for Aurora "
        "PostgreSQL and Standard RDS.",
        y,
    )
    y = _draw_heading(canvas, "3.1 What the Curve Shows", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "Standard RDS crosses the 500 ms SLO at 250 concurrent connections and reaches "
            "1,040 ms at 500 connections.",
            "Aurora stays below 45 ms through 500 connections, ending at 44 ms.",
            "The inflection is a capacity signal, not gradual degradation; provision to the "
            "200-connection knee.",
            "At 250 connections, Standard RDS is 20x slower than Aurora (620 ms versus 31 ms).",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, CLOUD_PDF, "SECTION 04", 4)
    y = _draw_heading(canvas, "4. Thread Scaling", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Figure 2 measures read throughput as the thread pool grows from 8 to 128 threads. "
        "All engines add throughput until 64 threads; beyond that point, contention offsets "
        "additional parallelism.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_figure(canvas, _throughput_figure(), y, max_height=300)
    y = _draw_caption(
        canvas,
        "Figure 2 - Read throughput (ops/s) versus thread-pool size for three managed "
        "PostgreSQL engines.",
        y,
    )
    y = _draw_heading(canvas, "4.1 Scaling Takeaways", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "Aurora peaks at 18,400 ops/sec with 64 threads and slightly regresses at 128 threads.",
            "Cloud SQL improves from 4,800 to 14,900 ops/sec between 8 and 64 threads.",
            "Standard RDS plateaus at roughly 10,200 ops/sec, the lowest ceiling of the three.",
            "A 64-thread pool is the recommended operating point for this workload.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, CLOUD_PDF, "SECTION 05", 5)
    y = _draw_heading(canvas, "5. Pooling and Storage Configuration", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Table 2 captures the connection-pooling and storage envelope used by each deployment. "
        "The matrix is designed to expose merged-cell decisions: pooling behavior, storage class, "
        "and monthly cost travel together for each engine.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_caption(
        canvas,
        "Table 2 - Connection pooling, storage configuration, and monthly cost.",
        y,
    )
    y = _draw_table(
        canvas,
        CLOUD_TABLE_2,
        [90, 68, 62, 62, 92, 56, 65],
        y,
        header_rows=2,
        font_size=6.8,
        spans=[
            ((0, 0), (0, 1)),
            ((1, 0), (3, 0)),
            ((4, 0), (5, 0)),
            ((6, 0), (6, 1)),
        ],
        center_from=2,
    )
    y = _draw_heading(canvas, "5.1 Recommendation", y, size=11.5)
    y = _draw_wrapped_text(
        canvas,
        "Aurora PostgreSQL with the transaction pooler is the recommended engine for sustained "
        "64-thread write workloads. It keeps the latency knee above 200 connections and "
        "preserves headroom for failover and burst traffic.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    _draw_bullets(
        canvas,
        [
            "Use PgBouncer in transaction mode with 5,000 max connections and 800 pooled "
            "connections per node.",
            "Reserve Aurora I/O-Optimized storage for predictable P99 latency at 64,000 IOPS.",
            "Cloud SQL Enterprise is about 15% cheaper per month but gives up 19% cache-on "
            "throughput.",
            "Standard RDS is not recommended above 200 connections without additional replicas.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()
    canvas.save()


def build_cache_pdf(path: Path) -> None:
    canvas = _new_canvas(path, "Distributed Cache Architecture 2026")
    _draw_intro_page(
        canvas,
        CACHE_PDF,
        "Distributed Cache Architecture 2026",
        "Multi-AZ cache topology, eviction policy trade-offs, and capacity sizing for a "
        "six-shard cluster.",
        "OmniDoc-Vision Benchmark Corpus | Architecture Report 2026 | 5 pages",
        "This architecture note describes a six-shard, multi-AZ cache tier designed for "
        "predictable microsecond reads at high request rates. The design uses RESP3 over "
        "mutual TLS, an LFU default eviction policy, and CRDT counters to reconcile writes "
        "across regions.",
        [
            "Target: sub-millisecond P99 reads at 360k read ops/sec across six shards.",
            "Availability: replication factor 2 with one replica in the next availability zone.",
            "Consistency: eventual consistency with CRDT counters for cross-region writes.",
            "Operations: a Sentinel control plane performs health checks and automated failover.",
        ],
        [
            "LFU preserves a 96.3% hit ratio with a 256 GB cache, the best result in the study.",
            "TTL has the lowest P99 latency (280 microseconds) but the weakest hit ratio.",
            "The six-shard cluster supports 360k read ops/sec and 144k write ops/sec with RF=2.",
        ],
        [
            ("SHARDS", "6 cache shards"),
            ("REPLICATION", "RF=2 across 3 AZs"),
            ("DEFAULT POLICY", "LFU"),
            ("PEAK READS", "360k ops/sec"),
        ],
        [
            ("PAGE 2", "Figure 1 - multi-AZ cluster topology"),
            ("PAGE 3", "Table 1 - eviction policy comparison"),
            ("PAGE 4", "Figure 2 - hit ratio vs cache size"),
            ("PAGE 5", "Table 2 - cluster sizing and replication"),
        ],
    )

    top = _draw_page_frame(canvas, CACHE_PDF, "SECTION 02", 2)
    y = _draw_heading(canvas, "2. Cluster Topology", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Figure 1 shows client traffic entering through a global load balancer and three "
        "stateless shard routers. Each of the six cache shards keeps a replica in the next "
        "availability zone; the control plane owns health checks and failover.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_figure(canvas, _topology_figure(), y, max_height=310)
    y = _draw_caption(
        canvas,
        "Figure 1 - Six-shard cache topology with replication factor 2 across three "
        "availability zones.",
        y,
    )
    y = _draw_heading(canvas, "2.1 Topology Rules", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "Clients speak RESP3 over mutual TLS to the global load balancer.",
            "Three stateless shard routers hash keys onto six cache shards across three AZs.",
            "RF=2 places one replica in the next AZ; a single AZ loss leaves every key readable.",
            "The Sentinel control plane monitors shard health and promotes replicas within the "
            "failover budget.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, CACHE_PDF, "SECTION 03", 3)
    y = _draw_heading(canvas, "3. Eviction Policy Comparison", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Table 1 compares four eviction policies at two cache sizes. Hit ratio, memory "
        "overhead, and P99 latency are measured on a mixed read/write trace that includes "
        "hot-key bursts and session-scoped records.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_caption(
        canvas,
        "Table 1 - Eviction policy comparison at 64 GB and 256 GB cache sizes.",
        y,
    )
    y = _draw_table(
        canvas,
        CACHE_TABLE_1,
        [70, 70, 70, 75, 75, 135],
        y,
        header_rows=2,
        font_size=6.8,
        spans=[
            ((0, 0), (0, 1)),
            ((1, 0), (2, 0)),
            ((3, 0), (3, 1)),
            ((4, 0), (4, 1)),
            ((5, 0), (5, 1)),
        ],
        center_from=1,
    )
    y = _draw_heading(canvas, "3.1 Policy Trade-offs", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "LFU wins on hit ratio at both cache sizes, at the cost of a 4.1% memory overhead.",
            "TTL has the lowest P99 latency and memory overhead, making it suitable for session "
            "data with bounded lifetimes.",
            "Random eviction is the cheapest policy but loses 15 points of hit ratio versus "
            "LFU at 256 GB.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, CACHE_PDF, "SECTION 04", 4)
    y = _draw_heading(canvas, "4. Hit Ratio vs Cache Size", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Figure 2 extends the comparison across cache sizes from 16 GB to 512 GB. The curves "
        "confirm that LFU's advantage widens as the cache grows, while TTL trades hit ratio "
        "for predictable expiry behavior.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_figure(canvas, _hit_ratio_figure(), y, max_height=300)
    y = _draw_caption(
        canvas,
        "Figure 2 - Hit ratio (%) versus cache size (GB) for LRU, LFU, and TTL policies.",
        y,
    )
    y = _draw_heading(canvas, "4.1 Sensitivity Findings", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "At 64 GB, LFU reaches 91.2% versus 88.4% for LRU and 82.7% for TTL.",
            "At 256 GB, LFU reaches 96.3%; LRU reaches 94.1%; TTL reaches 89.5%.",
            "LFU is the best fit for stable hot-key sets; TTL remains preferable for "
            "session-scoped data.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, CACHE_PDF, "SECTION 05", 5)
    y = _draw_heading(canvas, "5. Cluster Sizing and Replication", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Table 2 maps shard count to read/write throughput and failover budget. It is the "
        "capacity-planning view used by the control plane and the client retry policy.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_caption(
        canvas,
        "Table 2 - Cluster sizing, throughput, and failover matrix.",
        y,
    )
    y = _draw_table(
        canvas,
        CACHE_TABLE_2,
        [45, 35, 55, 70, 80, 80, 55, 75],
        y,
        header_rows=2,
        font_size=6.8,
        spans=[
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (2, 1)),
            ((3, 0), (3, 1)),
            ((4, 0), (5, 0)),
            ((6, 0), (7, 0)),
        ],
        center_from=0,
    )
    y = _draw_heading(canvas, "5.1 Operational Guidance", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "Start at six shards and RF=2; scale to 12 shards when sustained reads exceed "
            "300k ops/sec.",
            "Keep one replica in the next AZ and reserve the third AZ for read traffic and "
            "failover capacity.",
            "Use LFU by default; switch to TTL only for session data with explicit expiry "
            "semantics.",
            "Failover P99 is 7.4 seconds at six shards; budget client retries for at least "
            "8 seconds.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()
    canvas.save()


def build_gpu_pdf(path: Path) -> None:
    canvas = _new_canvas(path, "GPU Inference Capacity Report 2026")
    _draw_intro_page(
        canvas,
        GPU_PDF,
        "GPU Inference Capacity Report 2026",
        "AWQ versus FP16 serving envelope for Qwen2.5-VL-7B on 16 GB to 48 GB GPUs.",
        "OmniDoc-Vision Benchmark Corpus | Capacity Report 2026 | 5 pages",
        "This report measures the serving envelope of Qwen2.5-VL-7B-Instruct in 4-bit AWQ "
        "and FP16 precision on a single GPU. It focuses on the 16 GB VRAM ceiling and "
        "identifies the batch sizes that preserve interactive latency without OOM.",
        [
            "Engine: vLLM with max model length 8,192 tokens and at most 2 images per request.",
            "After 50 warm-up requests, 500 measured requests were collected per configuration.",
            "Metrics: P50/P95 end-to-end latency and output throughput in tokens/sec.",
            "GPU limits: 16 GB, 24 GB, and 48 GB; batch sizes 1, 2, 4, 8, and 16.",
        ],
        [
            "AWQ uses 9.4 GB at batch 8 and fits comfortably inside the 16 GB ceiling.",
            "FP16 uses 15.6 GB at batch 8, leaving almost no headroom for vision tokens.",
            "FP16 at batch 16 projects to 19.6 GB and runs out of memory on a 16 GB GPU.",
        ],
        [
            ("AWQ @ BATCH 8", "9.4 GB VRAM"),
            ("FP16 @ BATCH 8", "15.6 GB VRAM"),
            ("P95 @ BATCH 8", "560 ms"),
            ("IMAGE LIMIT", "2 per request"),
        ],
        [
            ("PAGE 2", "Figure 1 - AWQ vs FP16 VRAM footprint"),
            ("PAGE 3", "Table 1 - batch size latency matrix"),
            ("PAGE 4", "Figure 2 - throughput and P95 latency"),
            ("PAGE 5", "Table 2 - GPU capacity planning"),
        ],
    )

    top = _draw_page_frame(canvas, GPU_PDF, "SECTION 02", 2)
    y = _draw_heading(canvas, "2. VRAM Footprint", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Figure 1 compares the total VRAM footprint of AWQ 4-bit and FP16 across batch sizes. "
        "The 16 GB limit line marks the hard capacity boundary for the reference GPU.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_figure(canvas, _vram_figure(), y, max_height=300)
    y = _draw_caption(
        canvas,
        "Figure 1 - VRAM footprint (GB) by batch size for AWQ 4-bit and FP16, with the "
        "16 GB GPU limit.",
        y,
    )
    y = _draw_heading(canvas, "2.1 Capacity Findings", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "AWQ grows from 5.2 GB at batch 1 to 13.1 GB at batch 16.",
            "FP16 reaches 15.6 GB at batch 8 and 19.6 GB at batch 16.",
            "The 16 GB GPU can serve AWQ through batch 16; FP16 is limited to batch 8 or lower.",
            "KV-cache growth dominates beyond batch 8 in both precisions.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, GPU_PDF, "SECTION 03", 3)
    y = _draw_heading(canvas, "3. Batch Size Latency Matrix", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Table 1 reports P50/P95 latency and output throughput for both precisions. FP16 is "
        "unavailable at batch 16 because the projected allocation exceeds the 16 GB ceiling.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_caption(
        canvas,
        "Table 1 - Batch size latency and throughput matrix for AWQ 4-bit and FP16.",
        y,
    )
    y = _draw_table(
        canvas,
        GPU_TABLE_1,
        [65, 70, 70, 75, 70, 70, 75],
        y,
        header_rows=2,
        font_size=6.8,
        spans=[
            ((0, 0), (0, 1)),
            ((1, 0), (3, 0)),
            ((4, 0), (6, 0)),
        ],
        center_from=1,
    )
    y = _draw_heading(canvas, "3.1 Matrix Observations", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "AWQ P95 latency rises from 265 ms at batch 1 to 920 ms at batch 16.",
            "FP16 is slower at every batch size and becomes unavailable at batch 16.",
            "AWQ throughput at batch 8 is 4,100 tokens/sec, 93% higher than FP16.",
            "Batch 8 keeps AWQ P95 at 560 ms; batch 16 trades latency for throughput.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, GPU_PDF, "SECTION 04", 4)
    y = _draw_heading(canvas, "4. Throughput and Latency Trade-off", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Figure 2 combines output throughput and P95 latency on a shared batch-size axis. "
        "The 500 ms reference line marks the transition from latency-sensitive to "
        "throughput-oriented serving.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_figure(canvas, _batch_scaling_figure(), y, max_height=300)
    y = _draw_caption(
        canvas,
        "Figure 2 - AWQ throughput (tokens/s) and P95 latency (ms) versus batch size.",
        y,
    )
    y = _draw_heading(canvas, "4.1 Serving Guidance", y, size=11.5)
    _draw_bullets(
        canvas,
        [
            "AWQ throughput rises from 820 to 5,200 tokens/sec between batch 1 and 16.",
            "P95 latency crosses 500 ms at batch 8 (560 ms) and reaches 920 ms at batch 16.",
            "For interactive requests, batch 8 is the best balance; batch 16 suits offline "
            "or batch-tolerant workloads.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()

    top = _draw_page_frame(canvas, GPU_PDF, "SECTION 05", 5)
    y = _draw_heading(canvas, "5. Capacity Planning", top, size=14)
    y = _draw_wrapped_text(
        canvas,
        "Table 2 translates the measurements into a serving envelope for 16 GB, 24 GB, and "
        "48 GB GPUs. The envelope assumes AWQ 4-bit and the image limits from the benchmark.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    y = _draw_caption(
        canvas,
        "Table 2 - GPU capacity planning matrix for AWQ serving.",
        y,
    )
    y = _draw_table(
        canvas,
        GPU_TABLE_2,
        [60, 72, 82, 92, 82, 57, 50],
        y,
        header_rows=2,
        font_size=6.8,
        spans=[
            ((0, 0), (0, 1)),
            ((1, 0), (4, 0)),
            ((5, 0), (5, 1)),
            ((6, 0), (6, 1)),
        ],
        center_from=1,
    )
    y = _draw_heading(canvas, "5.1 Recommendation", y, size=11.5)
    y = _draw_wrapped_text(
        canvas,
        "Deploy AWQ 4-bit on 16 GB GPUs with a maximum batch size of 8. At 24 GB and above, "
        "batch 16 recovers throughput for batch-tolerant workloads while preserving the "
        "8,192-token model limit.",
        MARGIN,
        y,
        CONTENT_WIDTH,
        size=9.0,
        leading=12.2,
    )
    _draw_bullets(
        canvas,
        [
            "At 16 GB, cap AWQ at batch 8 and reserve the 4.2 GB KV-cache budget for vision "
            "tokens.",
            "At 24 GB, AWQ batch 16 supports 64 concurrent requests with a 920 ms P95.",
            "FP16 at batch 16 exceeds 16 GB and OOMs; it is not a valid 16 GB configuration.",
            "For 48 GB, batch 32 supports 128 concurrent requests at a 700 ms P95.",
        ],
        MARGIN,
        y,
        CONTENT_WIDTH,
    )
    canvas.showPage()
    canvas.save()


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
            "AWQ 4-bit with a maximum batch size of 8",
            "AWQ 4-bit با حداکثر batch size برابر ۸",
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


def _validate_dataset(dataset: dict[str, Any]) -> None:
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


def _write_dataset(dataset: dict[str, Any]) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    build_cloud_database_pdf(RAW_PDF_DIR / CLOUD_PDF)
    build_cache_pdf(RAW_PDF_DIR / CACHE_PDF)
    build_gpu_pdf(RAW_PDF_DIR / GPU_PDF)

    dataset = build_golden_dataset()
    _validate_dataset(dataset)
    _write_dataset(dataset)

    print(f"Generated 3 PDFs in {RAW_PDF_DIR}")
    print("Generated 30 QA pairs (table=10, visual=10, text=10)")
    print(f"Wrote dataset to {DATASET_PATH}")


if __name__ == "__main__":
    main()
