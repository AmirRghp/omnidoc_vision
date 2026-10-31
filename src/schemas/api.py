from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from src.schemas.document import BoundingBox, ChunkModality

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RoutingPath(StrEnum):
    FAST_TEXT_TABLE_PATH = "FAST_TEXT_TABLE_PATH"
    VISUAL_VLM_PATH = "VISUAL_VLM_PATH"


class CRAGStatus(StrEnum):
    CONFIDENT = "CONFIDENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pdf_path: NonEmptyStr
    generate_vlm_captions: bool = True

    @field_validator("pdf_path")
    @classmethod
    def _validate_pdf_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or path.anchor or ".." in path.parts:
            raise ValueError("pdf_path must be relative to the configured PDF directory")
        return value


class ChunkCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: int = Field(ge=0)
    table: int = Field(ge=0)
    figure: int = Field(ge=0)
    total: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_total(self) -> Self:
        expected = self.text + self.table + self.figure
        if self.total != expected:
            raise ValueError(f"total must equal text + table + figure ({expected})")
        return self


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["indexed"] = "indexed"
    doc_id: str = Field(min_length=1)
    doc_name: str = Field(min_length=1)
    pages_processed: int = Field(ge=0)
    chunks_created: ChunkCounts
    elapsed_ms: float = Field(ge=0)


class MultimodalQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: NonEmptyStr
    top_k: int = Field(default=5, ge=1, le=20)
    modality_filter: ChunkModality | None = None
    force_visual_route: bool = False


class LatencyBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_ms: float = Field(ge=0)
    rerank_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class VisualCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    doc_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    modality: ChunkModality
    section_title: str | None = None
    rerank_score: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox
    image_url: str | None = Field(default=None, min_length=1)
    content_preview: str = Field(min_length=1)

    @field_validator("image_url")
    @classmethod
    def _validate_image_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("/api/v1/crops/"):
            raise ValueError("image_url must point at the crop endpoint")
        return value

    @model_validator(mode="after")
    def _validate_citation(self) -> Self:
        if self.bounding_box.page != self.page_number:
            raise ValueError("bounding_box.page must equal page_number")
        if self.modality is ChunkModality.FIGURE and self.image_url is None:
            raise ValueError("figure citations require image_url")
        return self


class MultimodalQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    routing_path: RoutingPath
    crag_status: CRAGStatus
    rewritten_query: str | None = Field(default=None, min_length=1)
    vlm_images_inspected: int = Field(default=0, ge=0, le=2)
    latency_ms: LatencyBreakdown
    citations: list[VisualCitation] = Field(default_factory=list)
