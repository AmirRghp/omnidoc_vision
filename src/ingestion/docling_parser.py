"""IBM Docling layout analysis, table reconstruction, and figure cropping engine."""

from __future__ import annotations

import asyncio
import html
import io
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pdfplumber
from PIL import Image
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.common.config import Settings, get_settings
from src.common.exceptions import DocumentProcessingError, PdfValidationError
from src.common.logger import get_logger
from src.schemas.document import BoundingBox, ChunkModality, CoordOrigin, MultimodalChunk

# Enforce Decompression Bomb Defense (25 Megapixels max) per docs/SECURITY.md
Image.MAX_IMAGE_PIXELS = 25_000_000

logger = get_logger(__name__)

_HEADER_FOOTER_TOP_PT: Final[float] = 48.0
_HEADER_FOOTER_BOTTOM_PT: Final[float] = 45.0
_MIN_BBOX_SPAN: Final[float] = 1e-4
_SECTION_HEADING_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+\S+|Executive Summary|Methodology|Key Findings)$"
)
_FIGURE_CAPTION_RE: Final[re.Pattern[str]] = re.compile(r"^Figure\s+\d+\b", re.IGNORECASE)
_TABLE_CAPTION_RE: Final[re.Pattern[str]] = re.compile(r"^Table\s+\d+\b", re.IGNORECASE)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _make_doc_id(pdf_path: Path) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", pdf_path.stem).strip("_").lower()
    if not slug:
        slug = "document"
    return slug if slug.startswith("doc_") else f"doc_{slug}"


def clamp_image_dimensions(image: Image.Image, max_dim: int = 1280) -> Image.Image:
    """Resize image preserving aspect ratio so neither width nor height exceeds max_dim."""
    if max_dim <= 0:
        raise ValueError("max_dim must be positive")
    width, height = image.size
    if width <= max_dim and height <= max_dim:
        return image.copy()

    scale = min(max_dim / float(width), max_dim / float(height))
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    rgb_image = image.convert("RGB") if image.mode not in ("RGB", "RGBA", "L") else image
    return rgb_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def normalize_bbox(
    raw_bbox: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
    page_number: int,
    origin: CoordOrigin = CoordOrigin.TOPLEFT,
) -> BoundingBox:
    """Normalize page coordinates into [0.0, 1.0] with TOPLEFT origin."""
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page_width and page_height must be positive")

    raw_x0, raw_y0, raw_x1, raw_y1 = raw_bbox
    x_min = min(raw_x0, raw_x1)
    x_max = max(raw_x0, raw_x1)

    if origin is CoordOrigin.BOTTOMLEFT:
        y_min = page_height - max(raw_y0, raw_y1)
        y_max = page_height - min(raw_y0, raw_y1)
    else:
        y_min = min(raw_y0, raw_y1)
        y_max = max(raw_y0, raw_y1)

    x0 = max(0.0, min(1.0 - _MIN_BBOX_SPAN, x_min / page_width))
    y0 = max(0.0, min(1.0 - _MIN_BBOX_SPAN, y_min / page_height))
    x1 = max(x0 + _MIN_BBOX_SPAN, min(1.0, x_max / page_width))
    y1 = max(y0 + _MIN_BBOX_SPAN, min(1.0, y_max / page_height))

    return BoundingBox(
        page=page_number,
        x0=round(x0, 6),
        y0=round(y0, 6),
        x1=round(x1, 6),
        y1=round(y1, 6),
        coord_origin=CoordOrigin.TOPLEFT,
    )


def _has_multi_row_header(cleaned_rows: list[list[str]]) -> bool:
    if len(cleaned_rows) < 3:
        return False
    row0, row1 = cleaned_rows[0], cleaned_rows[1]
    if not any(row0) or not any(row1):
        return False
    # Multi-row header occurs when row 0 has horizontal spans (empty cells after a label)
    # and row 1 supplies sub-column labels while leaving spanned lead columns blank.
    row0_has_blanks = any(cell == "" for cell in row0[1:])
    row1_starts_blank = row1[0] == ""
    return row0_has_blanks and row1_starts_blank


def reconstruct_table_artifacts(
    raw_rows: Sequence[Sequence[str | None]],
    *,
    caption: str | None = None,
    section_title: str | None = None,
) -> tuple[str, str]:
    """Reconstruct both searchable Markdown and lossless HTML from extracted table rows."""
    cleaned_rows: list[list[str]] = [
        [_clean_cell(cell) for cell in row]
        for row in raw_rows
        if any(_clean_cell(cell) for cell in row)
    ]
    if not cleaned_rows:
        raise DocumentProcessingError("Cannot reconstruct an empty table")

    col_count = max(len(row) for row in cleaned_rows)
    normalized_rows = [row + [""] * (col_count - len(row)) for row in cleaned_rows]

    multi_header = _has_multi_row_header(normalized_rows)
    header_row_count = 2 if multi_header else 1

    # Propagate horizontal spans across row 0
    top_headers: list[str] = []
    carry_header = ""
    for cell in normalized_rows[0]:
        if cell:
            carry_header = cell
        top_headers.append(carry_header)

    if multi_header:
        sub_headers = normalized_rows[1]
        composite_headers: list[str] = []
        for idx in range(col_count):
            top_val = top_headers[idx]
            sub_val = sub_headers[idx]
            if top_val and sub_val and top_val != sub_val:
                composite_headers.append(f"{top_val} / {sub_val}")
            else:
                composite_headers.append(top_val or sub_val or f"Col {idx + 1}")
    else:
        composite_headers = [
            cell if cell else f"Col {idx + 1}" for idx, cell in enumerate(normalized_rows[0])
        ]

    # Propagate vertical spans on the primary entity column (column 0) across data rows
    data_rows: list[list[str]] = []
    carry_first_col = ""
    for row in normalized_rows[header_row_count:]:
        resolved_row = list(row)
        if resolved_row[0]:
            carry_first_col = resolved_row[0]
        elif carry_first_col:
            resolved_row[0] = carry_first_col
        data_rows.append(resolved_row)

    # Build Markdown representation
    md_lines: list[str] = []
    if section_title:
        md_lines.append(f"### {section_title}")
    if caption:
        md_lines.append(caption)

    header_line = "| " + " | ".join(composite_headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in composite_headers) + " |"
    md_lines.extend([header_line, separator_line])
    for data_row in data_rows:
        md_lines.append("| " + " | ".join(cell or "-" for cell in data_row) + " |")

    markdown = "\n".join(md_lines)

    # Build lossless HTML representation
    html_parts: list[str] = ["<table>", "<thead>"]
    for h_row in normalized_rows[:header_row_count]:
        cells_html = "".join(f"<th>{html.escape(cell)}</th>" for cell in h_row)
        html_parts.append(f"<tr>{cells_html}</tr>")
    html_parts.extend(["</thead>", "<tbody>"])
    for d_row in data_rows:
        cells_html = "".join(f"<td>{html.escape(cell)}</td>" for cell in d_row)
        html_parts.append(f"<tr>{cells_html}</tr>")
    html_parts.extend(["</tbody>", "</table>"])

    return markdown, "".join(html_parts)


class DoclingParser:
    """Extracts reading-order text, reconstructed tables, and cropped figures from PDFs."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        crops_dir: Path | None = None,
        use_docling: bool = False,
        converter: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.crops_dir = crops_dir if crops_dir is not None else self.settings.crops_dir
        self.use_docling = use_docling
        self._converter = converter

    def _validate_pdf(self, pdf_path: Path) -> int:
        if not pdf_path.exists() or not pdf_path.is_file():
            raise PdfValidationError(f"PDF file does not exist: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise PdfValidationError(f"Expected a .pdf file, got: {pdf_path.name}")

        size_bytes = pdf_path.stat().st_size
        if size_bytes <= 0:
            raise PdfValidationError(f"PDF file is empty: {pdf_path.name}")
        max_bytes = self.settings.max_pdf_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise PdfValidationError(
                f"PDF file {pdf_path.name} ({size_bytes} bytes) exceeds maximum file size "
                f"of {self.settings.max_pdf_size_mb} MB"
            )

        try:
            reader = PdfReader(str(pdf_path))
            page_count = len(reader.pages)
        except (PdfReadError, Exception) as exc:
            raise PdfValidationError(f"Invalid or unreadable PDF {pdf_path.name}: {exc}") from exc

        if page_count <= 0:
            raise PdfValidationError(f"PDF {pdf_path.name} has no pages")
        if page_count > self.settings.max_pdf_pages:
            raise PdfValidationError(
                f"PDF {pdf_path.name} ({page_count} pages) exceeds maximum page limit "
                f"of {self.settings.max_pdf_pages}"
            )
        return page_count

    def _save_crop_image(self, image: Image.Image, chunk_id: str) -> str:
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{chunk_id}.png"
        target_path = self.crops_dir / filename
        clamped = clamp_image_dimensions(image, max_dim=self.settings.max_image_dim)
        clamped.save(target_path, format="PNG")
        return f"data/crops/{filename}"

    def parse_pdf(self, pdf_path: Path | str) -> list[MultimodalChunk]:
        """Synchronously parse a PDF into validated MultimodalChunk objects."""
        resolved_path = Path(pdf_path)
        page_count = self._validate_pdf(resolved_path)
        doc_id = _make_doc_id(resolved_path)
        doc_name = resolved_path.name

        if self.use_docling or self._converter is not None:
            try:
                chunks = self._parse_with_docling(resolved_path, doc_id=doc_id, doc_name=doc_name)
                if chunks:
                    logger.info(
                        "Parsed PDF with Docling engine",
                        extra={
                            "doc_name": doc_name,
                            "pages": page_count,
                            "chunks": len(chunks),
                        },
                    )
                    return chunks
            except Exception as exc:
                logger.warning(
                    "Docling engine unavailable or failed; falling back to deterministic parser",
                    extra={"doc_name": doc_name, "error": str(exc)},
                )

        chunks = self._parse_with_fallback(resolved_path, doc_id=doc_id, doc_name=doc_name)
        logger.info(
            "Parsed PDF with deterministic fallback engine",
            extra={
                "doc_name": doc_name,
                "pages": page_count,
                "chunks": len(chunks),
            },
        )
        return chunks

    async def aparse_pdf(self, pdf_path: Path | str) -> list[MultimodalChunk]:
        """Asynchronously parse a PDF in a worker thread without blocking the event loop."""
        return await asyncio.to_thread(self.parse_pdf, pdf_path)

    def _get_or_create_docling_converter(self) -> Any:
        if self._converter is not None:
            return self._converter

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            generate_picture_images=True,
            generate_page_images=False,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        return self._converter

    def _parse_with_docling(
        self,
        pdf_path: Path,
        *,
        doc_id: str,
        doc_name: str,
    ) -> list[MultimodalChunk]:
        converter = self._get_or_create_docling_converter()
        conv_result = converter.convert(str(pdf_path))
        doc = conv_result.document

        pages_dict: dict[int, Any] = getattr(doc, "pages", {}) or {}
        chunks: list[MultimodalChunk] = []
        current_section: str | None = None
        counters: dict[tuple[int, ChunkModality], int] = {}

        for entry in doc.iterate_items():
            item = entry[0] if isinstance(entry, tuple) else entry
            label = str(getattr(item, "label", "")).lower()

            prov_list = getattr(item, "prov", None) or []
            page_no = int(getattr(prov_list[0], "page_no", 1)) if prov_list else 1
            page_obj = pages_dict.get(page_no)
            page_size = getattr(page_obj, "size", None)
            page_width = float(getattr(page_size, "width", 595.27))
            page_height = float(getattr(page_size, "height", 841.89))

            bbox = self._extract_docling_bbox(
                prov_list,
                page_width=page_width,
                page_height=page_height,
                page_number=page_no,
            )

            if "header" in label or "title" in label:
                heading_text = _clean_cell(getattr(item, "text", ""))
                if heading_text:
                    current_section = heading_text
                continue

            if "table" in label:
                idx = counters.get((page_no, ChunkModality.TABLE), 0) + 1
                counters[(page_no, ChunkModality.TABLE)] = idx
                chunk_id = f"{doc_id}_p{page_no:02d}_tbl{idx:02d}"

                md_text = self._call_docling_export(item, "export_to_markdown", doc)
                html_text = self._call_docling_export(item, "export_to_html", doc)
                if not md_text:
                    md_text = "[TABLE]"
                if not html_text or "<table" not in html_text.lower():
                    html_text = f"<table><tr><td>{html.escape(md_text)}</td></tr></table>"

                chunks.append(
                    MultimodalChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_number=page_no,
                        modality=ChunkModality.TABLE,
                        section_title=current_section,
                        content=md_text,
                        table_html=html_text,
                        bounding_box=bbox,
                    )
                )
            elif "picture" in label or "figure" in label:
                pil_image = self._extract_docling_image(item, doc)
                if pil_image is None:
                    continue
                idx = counters.get((page_no, ChunkModality.FIGURE), 0) + 1
                counters[(page_no, ChunkModality.FIGURE)] = idx
                chunk_id = f"{doc_id}_p{page_no:02d}_fig{idx:02d}"
                image_path = self._save_crop_image(pil_image, chunk_id)

                caption = _clean_cell(getattr(item, "caption_text", ""))
                if not caption:
                    caption_fn = getattr(item, "caption_text", None)
                    if callable(caption_fn):
                        caption = _clean_cell(caption_fn(doc))
                fallback_label = current_section or doc_name
                content = (
                    f"[FIGURE] {caption}"
                    if caption
                    else f"[FIGURE] Visual figure on page {page_no} ({fallback_label})"
                )
                chunks.append(
                    MultimodalChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_number=page_no,
                        modality=ChunkModality.FIGURE,
                        section_title=current_section,
                        content=content,
                        image_path=image_path,
                        bounding_box=bbox,
                    )
                )
            else:
                text_val = _clean_cell(getattr(item, "text", ""))
                if not text_val:
                    continue
                idx = counters.get((page_no, ChunkModality.TEXT), 0) + 1
                counters[(page_no, ChunkModality.TEXT)] = idx
                chunk_id = f"{doc_id}_p{page_no:02d}_txt{idx:02d}"
                formatted_text = (
                    f"{current_section}\n{text_val}" if current_section else text_val
                )
                chunks.append(
                    MultimodalChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_number=page_no,
                        modality=ChunkModality.TEXT,
                        section_title=current_section,
                        content=formatted_text,
                        bounding_box=bbox,
                    )
                )

        return chunks

    @staticmethod
    def _call_docling_export(item: Any, method_name: str, doc: Any) -> str:
        method = getattr(item, method_name, None)
        if not callable(method):
            return ""
        try:
            return str(method(doc=doc)).strip()
        except TypeError:
            return str(method()).strip()

    @staticmethod
    def _extract_docling_image(item: Any, doc: Any) -> Image.Image | None:
        get_image = getattr(item, "get_image", None)
        if callable(get_image):
            try:
                img = get_image(doc)
            except TypeError:
                img = get_image()
            if isinstance(img, Image.Image):
                return img
        image_attr = getattr(item, "image", None)
        pil_img = getattr(image_attr, "pil_image", None)
        if isinstance(pil_img, Image.Image):
            return pil_img
        return None

    @staticmethod
    def _extract_docling_bbox(
        prov_list: Sequence[Any],
        *,
        page_width: float,
        page_height: float,
        page_number: int,
    ) -> BoundingBox:
        if not prov_list:
            return BoundingBox(page=page_number, x0=0.05, y0=0.05, x1=0.95, y1=0.95)
        bbox_obj = getattr(prov_list[0], "bbox", None)
        if bbox_obj is None:
            return BoundingBox(page=page_number, x0=0.05, y0=0.05, x1=0.95, y1=0.95)

        left = float(getattr(bbox_obj, "l", 0.0))
        top = float(getattr(bbox_obj, "t", 0.0))
        right = float(getattr(bbox_obj, "r", page_width))
        bottom = float(getattr(bbox_obj, "b", page_height))
        origin_attr = getattr(bbox_obj, "coord_origin", None)
        origin_name = str(getattr(origin_attr, "name", origin_attr or "TOPLEFT")).upper()
        origin = (
            CoordOrigin.BOTTOMLEFT
            if "BOTTOM" in origin_name
            else CoordOrigin.TOPLEFT
        )
        return normalize_bbox(
            (left, top, right, bottom),
            page_width=page_width,
            page_height=page_height,
            page_number=page_number,
            origin=origin,
        )

    def _parse_with_fallback(
        self,
        pdf_path: Path,
        *,
        doc_id: str,
        doc_name: str,
    ) -> list[MultimodalChunk]:
        reader = PdfReader(str(pdf_path))
        chunks: list[MultimodalChunk] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_index, plumber_page in enumerate(pdf.pages):
                page_number = page_index + 1
                pypdf_page = reader.pages[page_index]
                page_width = float(plumber_page.width or 595.27)
                page_height = float(plumber_page.height or 841.89)

                detected_tables = plumber_page.find_tables()
                table_bboxes: list[tuple[float, float, float, float]] = [
                    (
                        float(tbl.bbox[0]),
                        float(tbl.bbox[1]),
                        float(tbl.bbox[2]),
                        float(tbl.bbox[3]),
                    )
                    for tbl in detected_tables
                ]

                lines = self._extract_non_table_lines(
                    plumber_page,
                    table_bboxes=table_bboxes,
                    page_height=page_height,
                )

                section_title = self._detect_section_title(lines)
                figure_captions = [
                    line["text"]
                    for line in lines
                    if _FIGURE_CAPTION_RE.match(line["text"])
                ]
                table_captions = [
                    line["text"]
                    for line in lines
                    if _TABLE_CAPTION_RE.match(line["text"])
                ]

                # 1. Reading-order text chunk(s) on this page
                prose_lines = [
                    line
                    for line in lines
                    if not _FIGURE_CAPTION_RE.match(line["text"])
                    and not _TABLE_CAPTION_RE.match(line["text"])
                ]
                if prose_lines:
                    prose_text = "\n".join(line["text"] for line in prose_lines)
                    text_bbox = normalize_bbox(
                        (
                            min(line["x0"] for line in prose_lines),
                            min(line["top"] for line in prose_lines),
                            max(line["x1"] for line in prose_lines),
                            max(line["bottom"] for line in prose_lines),
                        ),
                        page_width=page_width,
                        page_height=page_height,
                        page_number=page_number,
                    )
                    chunks.append(
                        MultimodalChunk(
                            chunk_id=f"{doc_id}_p{page_number:02d}_txt01",
                            doc_id=doc_id,
                            doc_name=doc_name,
                            page_number=page_number,
                            modality=ChunkModality.TEXT,
                            section_title=section_title,
                            content=prose_text,
                            bounding_box=text_bbox,
                        )
                    )

                # 2. Reconstructed Table chunks on this page
                for tbl_idx, tbl in enumerate(detected_tables, start=1):
                    raw_rows = tbl.extract()
                    if not raw_rows or not any(
                        any(_clean_cell(c) for c in row) for row in raw_rows if row
                    ):
                        continue
                    caption = (
                        table_captions[tbl_idx - 1]
                        if tbl_idx - 1 < len(table_captions)
                        else None
                    )
                    md_table, html_table = reconstruct_table_artifacts(
                        raw_rows,
                        caption=caption,
                        section_title=section_title,
                    )
                    tbl_bbox = normalize_bbox(
                        (
                            float(tbl.bbox[0]),
                            float(tbl.bbox[1]),
                            float(tbl.bbox[2]),
                            float(tbl.bbox[3]),
                        ),
                        page_width=page_width,
                        page_height=page_height,
                        page_number=page_number,
                    )
                    chunks.append(
                        MultimodalChunk(
                            chunk_id=f"{doc_id}_p{page_number:02d}_tbl{tbl_idx:02d}",
                            doc_id=doc_id,
                            doc_name=doc_name,
                            page_number=page_number,
                            modality=ChunkModality.TABLE,
                            section_title=section_title,
                            content=md_table,
                            table_html=html_table,
                            bounding_box=tbl_bbox,
                        )
                    )

                # 3. Cropped Figure chunks on this page
                plumber_images = list(getattr(plumber_page, "images", []) or [])
                for fig_idx, image_file in enumerate(pypdf_page.images, start=1):
                    try:
                        with Image.open(io.BytesIO(image_file.data)) as raw_img:
                            pil_img = raw_img.copy()
                    except Exception as exc:
                        raise DocumentProcessingError(
                            f"Failed to decode embedded image on page {page_number} of {doc_name}"
                        ) from exc

                    chunk_id = f"{doc_id}_p{page_number:02d}_fig{fig_idx:02d}"
                    image_path = self._save_crop_image(pil_img, chunk_id)

                    fig_bbox = self._resolve_figure_bbox(
                        plumber_images,
                        fig_index=fig_idx - 1,
                        page_width=page_width,
                        page_height=page_height,
                        page_number=page_number,
                    )
                    fig_caption = (
                        figure_captions[fig_idx - 1]
                        if fig_idx - 1 < len(figure_captions)
                        else (figure_captions[0] if figure_captions else None)
                    )
                    summary_parts = [f"[FIGURE] {fig_caption}" if fig_caption else "[FIGURE]"]
                    if section_title:
                        summary_parts.append(f"Section: {section_title}")
                    if prose_lines:
                        context_snippet = " ".join(line["text"] for line in prose_lines[:4])
                        summary_parts.append(f"Context: {context_snippet}")

                    chunks.append(
                        MultimodalChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            doc_name=doc_name,
                            page_number=page_number,
                            modality=ChunkModality.FIGURE,
                            section_title=section_title,
                            content=" | ".join(summary_parts),
                            image_path=image_path,
                            bounding_box=fig_bbox,
                        )
                    )

        return chunks

    @staticmethod
    def _extract_non_table_lines(
        plumber_page: Any,
        *,
        table_bboxes: list[tuple[float, float, float, float]],
        page_height: float,
    ) -> list[dict[str, Any]]:
        words = plumber_page.extract_words(keep_blank_chars=False, use_text_flow=True) or []
        filtered_words: list[dict[str, Any]] = []
        for word in words:
            x0 = float(word["x0"])
            x1 = float(word["x1"])
            top = float(word["top"])
            bottom = float(word["bottom"])

            # Skip page frame header/footer banners
            if top < _HEADER_FOOTER_TOP_PT or bottom > page_height - _HEADER_FOOTER_BOTTOM_PT:
                continue

            cx = (x0 + x1) / 2.0
            cy = (top + bottom) / 2.0
            in_table = any(
                (tb_x0 - 2.0 <= cx <= tb_x1 + 2.0) and (tb_top - 2.0 <= cy <= tb_bottom + 2.0)
                for tb_x0, tb_top, tb_x1, tb_bottom in table_bboxes
            )
            if not in_table:
                filtered_words.append(word)

        if not filtered_words:
            return []

        # Group words into horizontal reading lines by vertical proximity
        filtered_words.sort(key=lambda w: (round(float(w["top"]) / 3.0), float(w["x0"])))
        lines: list[dict[str, Any]] = []
        for word in filtered_words:
            w_top = float(word["top"])
            w_bottom = float(word["bottom"])
            w_x0 = float(word["x0"])
            w_x1 = float(word["x1"])
            w_text = str(word["text"]).strip()
            if not w_text:
                continue

            if lines and abs(w_top - float(lines[-1]["top"])) <= 3.5:
                current = lines[-1]
                current["words"].append((w_x0, w_text))
                current["x0"] = min(float(current["x0"]), w_x0)
                current["x1"] = max(float(current["x1"]), w_x1)
                current["top"] = min(float(current["top"]), w_top)
                current["bottom"] = max(float(current["bottom"]), w_bottom)
            else:
                lines.append(
                    {
                        "words": [(w_x0, w_text)],
                        "x0": w_x0,
                        "x1": w_x1,
                        "top": w_top,
                        "bottom": w_bottom,
                        "text": "",
                    }
                )

        for line in lines:
            sorted_words = [t for _, t in sorted(line["words"], key=lambda item: item[0])]
            line["text"] = " ".join(sorted_words)
            del line["words"]

        return [line for line in lines if line["text"]]

    @staticmethod
    def _detect_section_title(lines: Sequence[dict[str, Any]]) -> str | None:
        for line in lines:
            text = str(line["text"]).strip()
            if _SECTION_HEADING_RE.match(text):
                return text
        return str(lines[0]["text"]).strip() if lines else None

    @staticmethod
    def _resolve_figure_bbox(
        plumber_images: Sequence[dict[str, Any]],
        *,
        fig_index: int,
        page_width: float,
        page_height: float,
        page_number: int,
    ) -> BoundingBox:
        if fig_index < len(plumber_images):
            img_meta = plumber_images[fig_index]
            x0 = float(img_meta.get("x0", page_width * 0.08))
            top = float(img_meta.get("top", page_height * 0.22))
            x1 = float(img_meta.get("x1", page_width * 0.92))
            bottom = float(img_meta.get("bottom", page_height * 0.72))
            return normalize_bbox(
                (x0, top, x1, bottom),
                page_width=page_width,
                page_height=page_height,
                page_number=page_number,
                origin=CoordOrigin.TOPLEFT,
            )
        return BoundingBox(
            page=page_number,
            x0=0.08,
            y0=0.22,
            x1=0.92,
            y1=0.72,
            coord_origin=CoordOrigin.TOPLEFT,
        )
