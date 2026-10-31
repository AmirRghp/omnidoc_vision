"""Benchmark document builders."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark.charts import _batch_scaling_figure, _vram_figure
from scripts.benchmark.content import GPU_PDF, GPU_TABLE_1, GPU_TABLE_2
from scripts.benchmark.layout import (
    CONTENT_WIDTH,
    MARGIN,
    _draw_bullets,
    _draw_caption,
    _draw_figure,
    _draw_heading,
    _draw_intro_page,
    _draw_page_frame,
    _draw_table,
    _draw_wrapped_text,
    _new_canvas,
)


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
            "Batch 16 fits the 16 GB ceiling, but the interactive recommendation remains batch 8 "
            "to reserve KV-cache headroom for vision tokens; FP16 is limited to batch 8.",
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
        "Deploy AWQ 4-bit on 16 GB GPUs with a recommended batch size of 8. At 24 GB and above, "
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
