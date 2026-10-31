"""Matplotlib charts and architecture diagrams for the benchmark corpus."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.benchmark.content import (
    CACHE_LFU_HIT,
    CACHE_LRU_HIT,
    CACHE_SIZES,
    CACHE_TTL_HIT,
    CLOUD_AURORA_LATENCY,
    CLOUD_AURORA_THROUGHPUT,
    CLOUD_CLOUD_SQL_THROUGHPUT,
    CLOUD_CONNECTIONS,
    CLOUD_RDS_LATENCY,
    CLOUD_RDS_THROUGHPUT,
    CLOUD_THREADS,
    GPU_AWQ_P95,
    GPU_AWQ_THROUGHPUT,
    GPU_AWQ_VRAM,
    GPU_BATCHES,
    GPU_FP16_VRAM,
    GPU_VRAM_BATCHES,
)

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
    connections = CLOUD_CONNECTIONS
    aurora = CLOUD_AURORA_LATENCY
    rds = CLOUD_RDS_LATENCY

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
    threads = CLOUD_THREADS
    aurora = CLOUD_AURORA_THROUGHPUT
    cloud_sql = CLOUD_CLOUD_SQL_THROUGHPUT
    rds = CLOUD_RDS_THROUGHPUT

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
    cache_sizes = CACHE_SIZES
    lru = CACHE_LRU_HIT
    lfu = CACHE_LFU_HIT
    ttl = CACHE_TTL_HIT

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
    batch_sizes = [f"Batch {batch}" for batch in GPU_VRAM_BATCHES]
    awq = GPU_AWQ_VRAM
    fp16 = GPU_FP16_VRAM

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
    batches = GPU_BATCHES
    throughput = GPU_AWQ_THROUGHPUT
    p95 = GPU_AWQ_P95

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
