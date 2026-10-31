"""Benchmark document builders."""

from __future__ import annotations

from pathlib import Path

from scripts.benchmark.charts import _hit_ratio_figure, _topology_figure
from scripts.benchmark.content import CACHE_PDF, CACHE_TABLE_1, CACHE_TABLE_2
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
