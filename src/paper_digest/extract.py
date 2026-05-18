"""PDF text extraction and weak-page detection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paper_digest.schema import ExtractedPaper, PageText, PaperMetadata, SourceInfo


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be parsed."""


def extract_pdf(source_info: SourceInfo) -> ExtractedPaper:
    """Extract page-aware text and metadata from a PDF with PyMuPDF."""

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise ExtractionError("PyMuPDF is required for PDF extraction. Install pymupdf.") from exc

    document = fitz.open(source_info.source_pdf)
    raw_metadata: dict[str, Any] = document.metadata or {}
    source_url = (
        source_info.input_ref if source_info.input_ref.startswith(("http://", "https://")) else None
    )
    metadata = PaperMetadata(
        title=_clean_metadata(raw_metadata.get("title")),
        authors=_split_authors(raw_metadata.get("author")),
        subject=_clean_metadata(raw_metadata.get("subject")),
        keywords=_split_keywords(raw_metadata.get("keywords")),
        page_count=document.page_count,
        source_url=source_url,
        sha256=source_info.sha256,
    )

    pages: list[PageText] = []
    for index, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        image_count = len(page.get_images(full=True))
        weak_text, notes = score_page_quality(text=text, image_count=image_count)
        pages.append(
            PageText(
                page_number=index,
                text=text,
                image_count=image_count,
                character_count=len(text),
                weak_text=weak_text,
                quality_notes=notes,
            )
        )
    document.close()
    return ExtractedPaper(metadata=metadata, pages=pages)


def score_page_quality(text: str, image_count: int) -> tuple[bool, list[str]]:
    """Flag pages that likely need extra vision-model parsing."""

    notes: list[str] = []
    stripped = text.strip()
    if len(stripped) < 250:
        notes.append("very_low_text")
    if image_count > 0 and len(stripped) < 800:
        notes.append("image_heavy_low_text")
    if _garbled_ratio(stripped) > 0.18:
        notes.append("garbled_text")
    if image_count >= 3:
        notes.append("many_embedded_images")
    return bool(notes), notes


def render_pages_to_images(pdf_path: Path, page_numbers: list[int], image_dir: Path) -> list[Path]:
    """Render selected one-indexed PDF pages as PNGs for vision models."""

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("PyMuPDF is required to render pages for vision parsing.") from exc

    image_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    image_paths: list[Path] = []
    for page_number in page_numbers:
        if page_number < 1 or page_number > document.page_count:
            continue
        page = document[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image_path = image_dir / f"page-{page_number:04d}.png"
        pixmap.save(image_path)
        image_paths.append(image_path)
    document.close()
    return image_paths


def write_extracted_text(extracted: ExtractedPaper, destination: Path) -> None:
    lines: list[str] = []
    for page in extracted.pages:
        notes = f" ({', '.join(page.quality_notes)})" if page.quality_notes else ""
        lines.append(f"## Page {page.page_number}{notes}\n")
        lines.append(page.text or "[No selectable text extracted]")
        lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_extracted_paper(extracted: ExtractedPaper, destination: Path) -> None:
    destination.write_text(extracted.model_dump_json(indent=2), encoding="utf-8")


def read_cached_extracted_paper(path: Path, expected_sha256: str) -> ExtractedPaper | None:
    if not path.exists():
        return None
    try:
        extracted = ExtractedPaper.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if extracted.metadata.sha256 != expected_sha256:
        return None
    if not extracted.pages:
        return None
    return extracted


def _clean_metadata(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def _split_authors(value: str | None) -> list[str]:
    cleaned = _clean_metadata(value)
    if not cleaned:
        return []
    return [item.strip() for item in re.split(r";|, and | and ", cleaned) if item.strip()]


def _split_keywords(value: str | None) -> list[str]:
    cleaned = _clean_metadata(value)
    if not cleaned:
        return []
    return [item.strip() for item in re.split(r";|,", cleaned) if item.strip()]


def _garbled_ratio(text: str) -> float:
    if not text:
        return 0.0
    garbled = sum(1 for character in text if character == "\ufffd" or ord(character) < 9)
    return garbled / len(text)
