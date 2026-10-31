"""Benchmark document builders."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark.charts import _latency_figure, _throughput_figure
from scripts.benchmark.content import CLOUD_PDF, CLOUD_TABLE_1, CLOUD_TABLE_2
from scripts.benchmark.layout import (
    CONTENT_WIDTH,
    FONT_ITALIC,
    MARGIN,
    MUTED,
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
