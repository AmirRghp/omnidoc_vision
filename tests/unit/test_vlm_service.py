"""Unit tests for the VLM Service client (Fast Mode, Multimodal Mode, and security guards)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError
from PIL import Image

from src.common.config import get_settings
from src.common.exceptions import AssetAccessError
from src.retrieval.dense import ScoredChunk
from src.schemas.document import BoundingBox, ChunkModality, CoordOrigin, MultimodalChunk
from src.services.vlm_service import (
    VLMService,
    format_document_evidence,
    load_prompts,
    unwrap_chunk,
)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a sample 400x300 RGB test image."""
    return Image.new("RGB", (400, 300), color="blue")


@pytest.fixture
def large_pil_image() -> Image.Image:
    """Create a 2560x1440 image that exceeds the 1280px limit."""
    return Image.new("RGB", (2560, 1440), color="purple")


@pytest.fixture
def text_chunk() -> MultimodalChunk:
    bbox = BoundingBox(
        page=1,
        x0=0.1,
        y0=0.1,
        x1=0.9,
        y1=0.3,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    return MultimodalChunk(
        chunk_id="doc_t1_p01_txt01",
        doc_id="doc_t1",
        doc_name="benchmarks_2026.pdf",
        page_number=1,
        modality=ChunkModality.TEXT,
        section_title="Introduction",
        content="Qdrant demonstrates superior vector search throughput under high concurrency.",
        bounding_box=bbox,
    )


@pytest.fixture
def table_chunk() -> MultimodalChunk:
    bbox = BoundingBox(
        page=2,
        x0=0.1,
        y0=0.2,
        x1=0.85,
        y1=0.6,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    return MultimodalChunk(
        chunk_id="doc_t1_p02_tbl01",
        doc_id="doc_t1",
        doc_name="benchmarks_2026.pdf",
        page_number=2,
        modality=ChunkModality.TABLE,
        section_title="Latency Benchmarks",
        content="| Engine | QPS | P99 Latency |\n| Qdrant | 18400 | 14.2ms |",
        table_html=(
            "<table><tr><th>Engine</th><th>QPS</th></tr>"
            "<tr><td>Qdrant</td><td>18400</td></tr></table>"
        ),
        bounding_box=bbox,
    )


@pytest.fixture
def figure_chunks(tmp_path: Path) -> list[MultimodalChunk]:
    crops_dir = get_settings().crops_dir
    crops_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[MultimodalChunk] = []
    for idx in range(1, 4):
        img_name = f"unit_test_vlm_fig{idx}.png"
        img_file = crops_dir / img_name
        img = Image.new("RGB", (300, 200), color="red")
        img.save(img_file, format="PNG")

        bbox = BoundingBox(
            page=idx,
            x0=0.1,
            y0=0.2,
            x1=0.8,
            y1=0.7,
            coord_origin=CoordOrigin.TOPLEFT,
        )
        chunk = MultimodalChunk(
            chunk_id=f"doc_t1_p0{idx}_fig01",
            doc_id="doc_t1",
            doc_name="benchmarks_2026.pdf",
            page_number=idx,
            modality=ChunkModality.FIGURE,
            section_title=f"Chart Section {idx}",
            content=f"[CHART] P99 Latency vs Concurrency curve #{idx}",
            image_path=f"data/crops/{img_name}",
            bounding_box=bbox,
        )
        chunks.append(chunk)

    return chunks


def test_load_prompts_defaults_and_yaml() -> None:
    """Prompt loader loads fallback prompts and overrides from yaml if present."""
    prompts = load_prompts()
    assert "fast_text" in prompts
    assert "visual_reasoning" in prompts
    assert "visual_caption" in prompts
    assert "crag_rewrite" in prompts
    assert "passive data" in prompts["fast_text"]["system"]
    assert "passive data" in prompts["visual_reasoning"]["system"]


def test_unwrap_chunk_and_format_document_evidence(
    text_chunk: MultimodalChunk,
    table_chunk: MultimodalChunk,
) -> None:
    """Evidence formatting wraps text and tables in XML tags and extracts ScoredChunk."""
    scored_table = ScoredChunk(chunk=table_chunk, score=0.92, rerank_score=0.95)

    assert unwrap_chunk(text_chunk) == text_chunk
    assert unwrap_chunk(scored_table) == table_chunk

    formatted = format_document_evidence([text_chunk, scored_table])
    assert '<document_evidence doc="benchmarks_2026.pdf" page="1" modality="text"' in formatted
    assert '<document_evidence doc="benchmarks_2026.pdf" page="2" modality="table"' in formatted
    assert "[HTML_TABLE]" in formatted
    assert "</document_evidence>" in formatted


@pytest.mark.asyncio
async def test_encode_image_clamping(large_pil_image: Image.Image) -> None:
    """Image resolution exceeding 1280px is scaled down preserving aspect ratio."""
    service = VLMService(offline_mode=True)
    b64 = await service.encode_image(large_pil_image, max_dim=1280)
    assert isinstance(b64, str)
    assert len(b64) > 0

    import base64

    raw = base64.b64decode(b64)
    with Image.open(io.BytesIO(raw)) as img:
        assert img.width == 1280
        assert img.height == 720


@pytest.mark.asyncio
async def test_encode_image_path_traversal_guards() -> None:
    """Path traversal in image_paths is blocked with AssetAccessError."""
    service = VLMService(offline_mode=True)
    with pytest.raises(AssetAccessError, match="Path traversal detected"):
        await service.encode_image("../../../etc/passwd")

    with pytest.raises(AssetAccessError, match="does not exist"):
        await service.encode_image("data/crops/non_existent_crop_9999.png")


@pytest.mark.asyncio
async def test_generate_fast_text_offline(
    text_chunk: MultimodalChunk,
    table_chunk: MultimodalChunk,
) -> None:
    """Fast text generation produces deterministic grounded output in offline mode."""
    service = VLMService(offline_mode=True)
    res = await service.generate_fast_text(
        query="What is Qdrant throughput?",
        chunks=[text_chunk, table_chunk],
    )
    assert "benchmarks_2026.pdf" in res
    assert "Deterministic offline mode" in res
    assert "Qdrant" in res


@pytest.mark.asyncio
async def test_generate_fast_text_online_mock(
    text_chunk: MultimodalChunk,
    table_chunk: MultimodalChunk,
) -> None:
    """Fast text sends text-only messages to OpenAI client with no image blocks."""
    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Qdrant delivers 18,400 QPS with 14.2ms P99 latency."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = VLMService(client=mock_client, offline_mode=False)
    res = await service.generate_fast_text(
        query="What is the throughput?",
        chunks=[text_chunk, table_chunk],
    )

    assert "18,400 QPS" in res
    mock_client.chat.completions.create.assert_awaited_once()

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "passive data" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "<document_evidence" in messages[1]["content"]
    # Verify no image_url in fast path
    assert isinstance(messages[1]["content"], str)


@pytest.mark.asyncio
async def test_generate_fast_text_connection_error_fallback(
    text_chunk: MultimodalChunk,
) -> None:
    """Network connection errors fall back to deterministic offline summary."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    service = VLMService(client=mock_client, offline_mode=False)
    res = await service.generate_fast_text(
        query="Check latency",
        chunks=[text_chunk],
    )
    assert "Deterministic offline mode" in res


@pytest.mark.asyncio
async def test_generate_visual_reasoning_offline(
    figure_chunks: list[MultimodalChunk],
) -> None:
    """Visual reasoning produces deterministic summary in offline mode."""
    service = VLMService(offline_mode=True)
    res = await service.generate_visual_reasoning(
        query="What is the latency inflection point?",
        chunks=figure_chunks,
    )
    assert "Deterministic offline visual reasoning mode" in res
    assert "benchmarks_2026.pdf" in res


@pytest.mark.asyncio
async def test_generate_visual_reasoning_online_mock(
    figure_chunks: list[MultimodalChunk],
) -> None:
    """Visual reasoning attaches Base64 images capped at max_images_per_query (2)."""
    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "The latency curve inflects sharply at 250 concurrency."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = VLMService(client=mock_client, offline_mode=False)
    # We pass 3 figure chunks, but service must clamp to 2
    res = await service.generate_visual_reasoning(
        query="Find inflection point",
        chunks=figure_chunks,
    )

    assert "inflects sharply at 250" in res
    mock_client.chat.completions.create.assert_awaited_once()

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"

    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    text_parts = [p for p in user_content if p["type"] == "text"]
    image_parts = [p for p in user_content if p["type"] == "image_url"]

    assert len(text_parts) == 1
    assert "<document_evidence" in text_parts[0]["text"]
    # Strict 16GB VRAM guard: max 2 images
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_generate_visual_reasoning_no_images_falls_back_to_fast_text(
    text_chunk: MultimodalChunk,
) -> None:
    """When chunks contain no figure crops, it cleanly routes to fast text generation."""
    service = VLMService(offline_mode=True)
    res = await service.generate_visual_reasoning(
        query="What is Qdrant throughput?",
        chunks=[text_chunk],
    )
    assert "Deterministic offline mode" in res


@pytest.mark.asyncio
async def test_rewrite_query_offline_and_online() -> None:
    """Query rewriting works in offline expansion and online completion modes."""
    service = VLMService(offline_mode=True)
    offline_rewrite = await service.rewrite_query("latency graph")
    assert "latency graph" in offline_rewrite
    assert "benchmark" in offline_rewrite

    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "database p99 latency concurrency curve benchmark"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    online_service = VLMService(client=mock_client, offline_mode=False)
    online_rewrite = await online_service.rewrite_query("latency graph")
    assert online_rewrite == "database p99 latency concurrency curve benchmark"
