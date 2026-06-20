"""Parser plus vision-model page parsing for PDFs."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from paper_digest.extract import render_pages_to_images
from paper_digest.llm import LLMHTTPError, OpenRouterClient
from paper_digest.schema import ExtractedPaper, SourceInfo, VisualExtraction
from paper_digest.tracing import TraceWriter

VISION_SYSTEM_PROMPT = """You parse rendered research-paper PDF pages.
Extract visible text, equations, tables, figures, captions, and layout-relevant notes.
Use the rendered page to recover anything the text parser might miss.
Do not infer beyond the image. Mark uncertainty explicitly."""


class VisionPageParse(VisualExtraction):
    """Schema alias used for one page at a time."""


def run_vision_parsing(
    *,
    source_info: SourceInfo,
    extracted: ExtractedPaper,
    llm: OpenRouterClient,
    trace: TraceWriter,
    max_pages: int,
    mode: Literal["auto", "all"] = "auto",
    concurrency: int = 5,
) -> list[VisualExtraction]:
    page_numbers = _select_pages(extracted, max_pages=max_pages, mode=mode)
    parse_dir = source_info.parse_root()
    if not page_numbers:
        _write_visual_extractions(parse_dir, [])
        return []

    cache_path = parse_dir / "visual_extractions.json"
    cached = _read_cached_visual_extractions(cache_path, page_numbers)
    if cached is not None and not llm.settings.force_parse:
        if llm.settings.verbose:
            print(
                f"[vision-parse] Reusing cached vision parse for pages {page_numbers}",
                flush=True,
            )
        return cached

    if llm.settings.verbose:
        print(
            f"[vision-parse] Rendering {len(page_numbers)} page(s) for vision parsing: "
            f"{page_numbers}",
            flush=True,
        )
    image_paths = render_pages_to_images(
        source_info.source_pdf,
        page_numbers,
        parse_dir / "page_images",
    )
    results = _parse_pages_in_parallel(
        image_paths=image_paths,
        page_numbers=page_numbers,
        llm=llm,
        trace=trace,
        concurrency=concurrency,
    )

    _write_visual_extractions(parse_dir, results)
    return results


def _parse_pages_in_parallel(
    *,
    image_paths: list[Path],
    page_numbers: list[int],
    llm: OpenRouterClient,
    trace: TraceWriter,
    concurrency: int,
) -> list[VisualExtraction]:
    pairs = list(zip(image_paths, page_numbers, strict=False))
    max_workers = max(1, min(concurrency, len(pairs)))
    if llm.settings.verbose:
        print(f"[vision-parse] Processing up to {max_workers} page(s) in parallel", flush=True)

    results_by_page: dict[int, VisualExtraction] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _parse_single_page,
                image_path=image_path,
                page_number=page_number,
                llm=llm,
                trace=trace,
            ): page_number
            for image_path, page_number in pairs
        }
        for future in as_completed(futures):
            page_number = futures[future]
            visual = future.result()
            results_by_page[page_number] = visual
            provider_rejected = any("vision_parse_http_403" in note for note in visual.notes)
            if provider_rejected:
                trace.record(
                    "vision_parse_provider_rejected",
                    output={"page_number": page_number},
                    notes=[
                        "A vision request was rejected; submitted pages may finish."
                    ],
                )

    return [
        results_by_page[page_number]
        for page_number in page_numbers
        if page_number in results_by_page
    ]


def _parse_single_page(
    *,
    image_path: Path,
    page_number: int,
    llm: OpenRouterClient,
    trace: TraceWriter,
) -> VisualExtraction:
    if llm.settings.verbose:
        print(f"[vision-parse] Page {page_number}: calling vision model", flush=True)
    try:
        result = llm.vision_json(
            step=f"vision_parse_page_{page_number}",
            system_prompt=VISION_SYSTEM_PROMPT,
            user_prompt=(
                f"Parse page {page_number}. Extract visible text the parser may miss, "
                "equations, table contents, figure meanings, captions, and uncertainty notes."
            ),
            image_paths=[image_path],
            response_model=VisionPageParse,
            trace=trace,
            prompt_version="vision-parse-page-v1",
        )
        if llm.settings.verbose:
            print(
                f"[vision-parse] Page {page_number}: complete "
                f"(confidence={result.confidence:.2f})",
                flush=True,
            )
        return VisualExtraction(
            page_number=page_number,
            image_path=str(image_path),
            extracted_text=result.extracted_text,
            visual_summary=result.visual_summary,
            equations=result.equations,
            tables=result.tables,
            figures=result.figures,
            confidence=result.confidence,
            notes=result.notes,
        )
    except LLMHTTPError as exc:
        visual = VisualExtraction(
            page_number=page_number,
            image_path=str(image_path),
            visual_summary="Vision model parsing failed for this page.",
            confidence=0.0,
            notes=[f"vision_parse_http_{exc.status_code}", exc.message[:500]],
        )
        trace.record(
            f"vision_parse_page_{page_number}_failed",
            input_summary={"image_path": image_path},
            output=visual,
            model=llm.settings.vision_model,
            validation_errors=[exc.message],
            notes=["Vision parsing was attempted but OpenRouter rejected the request."],
        )
        if llm.settings.verbose:
            print(
                f"[vision-parse] Page {page_number}: failed (HTTP {exc.status_code})",
                flush=True,
            )
        return visual


def write_empty_visual_extractions(run_dir: Path) -> None:
    _write_visual_extractions(run_dir, [])


def _select_pages(
    extracted: ExtractedPaper,
    *,
    max_pages: int,
    mode: Literal["auto", "all"],
) -> list[int]:
    if mode == "all":
        page_numbers = [page.page_number for page in extracted.pages]
    else:
        page_numbers = extracted.weak_pages
    if max_pages <= 0:
        return page_numbers
    return page_numbers[:max_pages]


def _write_visual_extractions(run_dir: Path, extractions: list[VisualExtraction]) -> None:
    path = run_dir / "visual_extractions.json"
    payload = [item.model_dump(mode="json") for item in extractions]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_cached_visual_extractions(
    path: Path,
    expected_pages: list[int],
) -> list[VisualExtraction] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        extractions = [VisualExtraction.model_validate(item) for item in raw]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    by_page = {item.page_number: item for item in extractions}
    if not all(page_number in by_page for page_number in expected_pages):
        return None
    return [by_page[page_number] for page_number in expected_pages]
