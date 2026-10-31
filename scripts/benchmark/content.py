"""Shared benchmark corpus metadata, tables, and chart series."""

from __future__ import annotations

from typing import Final

CLOUD_PDF: Final[str] = "cloud_database_benchmark_2026.pdf"
CACHE_PDF: Final[str] = "distributed_cache_architecture.pdf"
GPU_PDF: Final[str] = "gpu_inference_capacity_report.pdf"
TOTAL_PAGES: Final[int] = 5

CLOUD_CONNECTIONS: Final[list[int]] = [50, 100, 150, 200, 250, 300, 400, 500]
CLOUD_AURORA_LATENCY: Final[list[int]] = [22, 23, 24, 26, 31, 35, 38, 44]
CLOUD_RDS_LATENCY: Final[list[int]] = [38, 42, 46, 52, 620, 780, 910, 1040]
CLOUD_THREADS: Final[list[int]] = [8, 16, 32, 64, 128]
CLOUD_AURORA_THROUGHPUT: Final[list[int]] = [5600, 9800, 14200, 18400, 18100]
CLOUD_CLOUD_SQL_THROUGHPUT: Final[list[int]] = [4800, 8200, 12100, 14900, 14750]
CLOUD_RDS_THROUGHPUT: Final[list[int]] = [3500, 6100, 8900, 10200, 10150]

CACHE_SIZES: Final[list[int]] = [16, 32, 64, 128, 256, 512]
CACHE_LRU_HIT: Final[list[float]] = [79.5, 84.2, 88.4, 91.6, 94.1, 95.4]
CACHE_LFU_HIT: Final[list[float]] = [82.1, 87.4, 91.2, 94.0, 96.3, 97.1]
CACHE_TTL_HIT: Final[list[float]] = [71.8, 77.6, 82.7, 86.4, 89.5, 91.2]

GPU_BATCHES: Final[list[int]] = [1, 2, 4, 8, 16]
GPU_VRAM_BATCHES: Final[list[int]] = [1, 4, 8, 16]
GPU_AWQ_VRAM: Final[list[float]] = [5.2, 6.8, 9.4, 13.1]
GPU_FP16_VRAM: Final[list[float]] = [14.2, 15.1, 15.6, 19.6]
GPU_AWQ_THROUGHPUT: Final[list[int]] = [820, 1540, 2680, 4100, 5200]
GPU_AWQ_P95: Final[list[int]] = [265, 310, 390, 560, 920]

CLOUD_PEAK_THROUGHPUT: Final[int] = CLOUD_AURORA_THROUGHPUT[3]
CACHE_LFU_256: Final[float] = CACHE_LFU_HIT[4]
GPU_AWQ_BATCH8_VRAM: Final[float] = GPU_AWQ_VRAM[2]
GPU_FP16_BATCH16_VRAM: Final[float] = GPU_FP16_VRAM[3]
GPU_AWQ_BATCH8_P95: Final[int] = GPU_AWQ_P95[3]


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
        f"{CLOUD_PEAK_THROUGHPUT:,}",
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
    [
        "LRU",
        f"{CACHE_LRU_HIT[2]:.1f}%",
        f"{CACHE_LRU_HIT[4]:.1f}%",
        "3.2%",
        "410",
        "Mixed read/write",
    ],
    [
        "LFU",
        f"{CACHE_LFU_HIT[2]:.1f}%",
        f"{CACHE_LFU_HIT[4]:.1f}%",
        "4.1%",
        "430",
        "Stable hot keys",
    ],
    ["TTL", f"{CACHE_TTL_HIT[2]:.1f}%", f"{CACHE_TTL_HIT[4]:.1f}%", "1.8%", "280", "Session data"],
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
    ["1", "210", f"{GPU_AWQ_P95[0]}", f"{GPU_AWQ_THROUGHPUT[0]:,}", "340", "410", "510"],
    ["2", "245", f"{GPU_AWQ_P95[1]}", f"{GPU_AWQ_THROUGHPUT[1]:,}", "410", "500", "920"],
    ["4", "310", f"{GPU_AWQ_P95[2]}", f"{GPU_AWQ_THROUGHPUT[2]:,}", "560", "680", "1,480"],
    ["8", "455", f"{GPU_AWQ_P95[3]}", f"{GPU_AWQ_THROUGHPUT[3]:,}", "880", "1,050", "2,120"],
    ["16", "720", f"{GPU_AWQ_P95[4]}", f"{GPU_AWQ_THROUGHPUT[4]:,}", "OOM", "OOM", "OOM"],
]

GPU_TABLE_2: Final[list[list[str]]] = [
    [
        "GPU (GB)",
        "Recommended Serving Envelope",
        "",
        "",
        "",
        "Exp. P95 (ms)",
        "Fit",
    ],
    [
        "",
        "Rec. AWQ Batch",
        "KV Cache Budget (GB)",
        "Max Concurrent Requests",
        "Max Model Len",
        "",
        "",
    ],
    ["16 GB", "8", "4.2", "32", "8,192", f"{GPU_AWQ_BATCH8_P95}", "Yes (AWQ only)"],
    ["24 GB", "16", "9.8", "64", "8,192", f"{GPU_AWQ_P95[-1]}", "Yes"],
    ["48 GB", "32", "22.0", "128", "16,384", "700", "Yes"],
]

REFERENCE_TABLES: Final[dict[tuple[str, int], list[list[str]]]] = {
    (CLOUD_PDF, 2): CLOUD_TABLE_1,
    (CLOUD_PDF, 5): CLOUD_TABLE_2,
    (CACHE_PDF, 3): CACHE_TABLE_1,
    (CACHE_PDF, 5): CACHE_TABLE_2,
    (GPU_PDF, 3): GPU_TABLE_1,
    (GPU_PDF, 5): GPU_TABLE_2,
}

OFFLINE_APPROACH_A_FIGURE_CAPTIONS: Final[dict[tuple[str, int], str]] = {
    (CLOUD_PDF, 3): (
        "[FIGURE CAPTION & TECHNICAL SUMMARY] "
        "Title: Figure 1 - P99 Write Latency (ms) vs Concurrent Connections | "
        "Type: Line chart | Axes: Concurrent Connections (50 to 500) vs P99 Write Latency (ms) | "
        f"Connections: {CLOUD_CONNECTIONS} | "
        f"Aurora PostgreSQL P99 latency series (ms): {CLOUD_AURORA_LATENCY} "
        "(at 500 concurrent connections Aurora PostgreSQL P99 write latency is 44 ms) | "
        f"Standard RDS P99 latency series (ms): {CLOUD_RDS_LATENCY} "
        "(Standard RDS first exceeds 500 ms SLO threshold at 250 concurrent connections "
        "with 620 ms, reaching 1,040 ms at 500 connections)."
    ),
    (CLOUD_PDF, 4): (
        "[FIGURE CAPTION & TECHNICAL SUMMARY] "
        "Title: Figure 2 - Read Throughput (ops/s) vs Thread-Pool Size | "
        "Type: Line chart | Axes: Thread-Pool Size (8 to 128 threads) vs Read Throughput (ops/s) | "
        f"Threads: {CLOUD_THREADS} | "
        f"Aurora PostgreSQL read throughput series (ops/sec): {CLOUD_AURORA_THROUGHPUT} "
        "(reaches peak read throughput at 64 threads with 18,400 ops/sec) | "
        f"Cloud SQL Enterprise series (ops/sec): {CLOUD_CLOUD_SQL_THROUGHPUT} | "
        f"Standard RDS series (ops/sec): {CLOUD_RDS_THROUGHPUT}."
    ),
    (CACHE_PDF, 2): (
        "[FIGURE CAPTION & TECHNICAL SUMMARY] "
        "Title: Figure 1 - Multi-AZ Six-Shard Cache Topology | "
        "Type: Architecture topology diagram | "
        "Components: Global TLS/RESP3 Load Balancer, 3 Stateless Shard Routers, "
        "6 cache shards distributed across 3 Availability Zones (AZ-A, AZ-B, AZ-C), "
        "Sentinel Control Plane | "
        "Key Values & Inflection Points: 6 cache shards total, Replication factor 2 (RF=2) "
        "across 3 AZs with primary and replica placement."
    ),
    (CACHE_PDF, 4): (
        "[FIGURE CAPTION & TECHNICAL SUMMARY] "
        "Title: Figure 2 - Hit Ratio (%) vs Cache Size (GB) | "
        "Type: Line chart | Axes: Cache Size GB (16 to 512 GB) vs Hit Ratio (%) | "
        f"Cache Sizes (GB): {CACHE_SIZES} | "
        f"LFU hit ratio series (%): {CACHE_LFU_HIT} "
        "(LFU has the highest hit ratio at 256 GB cache size at 96.3%, and 91.2% at 64 GB) | "
        f"LRU hit ratio series (%): {CACHE_LRU_HIT} "
        "(LRU hit ratio at 64 GB cache size is 88.4%, and 94.1% at 256 GB) | "
        f"TTL hit ratio series (%): {CACHE_TTL_HIT}."
    ),
    (GPU_PDF, 2): (
        "[FIGURE CAPTION & TECHNICAL SUMMARY] "
        "Title: Figure 1 - AWQ 4-bit vs FP16 VRAM Footprint (GB) | "
        "Type: Grouped bar chart | Axes: Batch Size (1, 4, 8, 16) vs VRAM Usage (GB) | "
        f"Batch Sizes: {GPU_VRAM_BATCHES} | "
        f"AWQ 4-bit VRAM series (GB): {GPU_AWQ_VRAM} "
        "(AWQ uses 9.4 GB VRAM at batch size 8 and 13.1 GB at batch size 16, fitting 16 GB GPU) | "
        f"FP16 VRAM series (GB): {GPU_FP16_VRAM} "
        "(FP16 projected VRAM footprint at batch size 16 is 19.6 GB, which exceeds the 16 GB "
        "GPU limit and causes OOM)."
    ),
    (GPU_PDF, 4): (
        "[FIGURE CAPTION & TECHNICAL SUMMARY] "
        "Title: Figure 2 - AWQ Throughput (tokens/s) and P95 Latency (ms) vs Batch Size | "
        "Type: Dual-axis line/bar chart | Axes: Batch Size (1, 2, 4, 8, 16) | "
        f"Batch Sizes: {GPU_BATCHES} | "
        f"AWQ Throughput series (tokens/s): {GPU_AWQ_THROUGHPUT} | "
        f"AWQ P95 Latency series (ms): {GPU_AWQ_P95} "
        "(AWQ P95 latency is 390 ms at batch 4 and first exceeds 500 ms at Batch size 8 "
        "with 560 ms, reaching 920 ms at batch 16)."
    ),
}
