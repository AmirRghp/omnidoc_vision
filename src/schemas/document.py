from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class ChunkModality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class CoordOrigin(StrEnum):
    TOPLEFT = "TOPLEFT"
    BOTTOMLEFT = "BOTTOMLEFT"


NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class BoundingBox(BaseModel):
    """Normalized page-relative rectangle in [0, 1], as [x0, y0, x1, y1]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    page: int = Field(ge=1)
    x0: NormalizedCoordinate
    y0: NormalizedCoordinate
    x1: NormalizedCoordinate
    y1: NormalizedCoordinate
    coord_origin: CoordOrigin = CoordOrigin.TOPLEFT

    @model_validator(mode="after")
    def _validate_geometry(self) -> Self:
        if self.x1 <= self.x0:
            raise ValueError("x1 must be greater than x0")
        if self.y1 <= self.y0:
            raise ValueError("y1 must be greater than y0")
        return self


class MultimodalChunk(BaseModel):
    """Unified text, table, or figure chunk stored in Qdrant and BM25."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    doc_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    modality: ChunkModality
    section_title: str | None = None
    content: str = Field(min_length=1)
    table_html: str | None = Field(default=None, min_length=1)
    image_path: str | None = Field(default=None, min_length=1)
    bounding_box: BoundingBox
    ingestion_timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() or path.anchor or ".." in path.parts:
            raise ValueError("image_path must be a relative path without traversal")
        return value

    @field_validator("table_html")
    @classmethod
    def _reject_script_html(cls, value: str | None) -> str | None:
        if value is not None and "<script" in value.lower():
            raise ValueError("table_html must not contain script tags")
        return value

    @model_validator(mode="after")
    def _validate_payload(self) -> Self:
        if self.bounding_box.page != self.page_number:
            raise ValueError("bounding_box.page must equal page_number")

        if self.modality is ChunkModality.TEXT:
            if self.table_html is not None or self.image_path is not None:
                raise ValueError("text chunks must not have table_html or image_path")
        elif self.modality is ChunkModality.TABLE:
            if self.table_html is None:
                raise ValueError("table chunks require table_html")
        elif self.modality is ChunkModality.FIGURE:
            if self.image_path is None:
                raise ValueError("figure chunks require image_path")
            if self.table_html is not None:
                raise ValueError("figure chunks must not have table_html")

        return self
