"""PDF source resolution and caching."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from paper_digest.schema import SourceInfo


class IngestError(RuntimeError):
    """Raised when a PDF source cannot be resolved."""


def resolve_source(input_ref: str, output_dir: Path) -> SourceInfo:
    """Copy or download a PDF into its run directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _source_filename(input_ref)
    slug = slugify(Path(filename).stem or "paper")
    run_dir = output_dir / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    source_pdf = run_dir / "source.pdf"

    if _is_url(input_ref):
        _download_pdf(input_ref, source_pdf)
    else:
        path = Path(input_ref).expanduser()
        if not path.exists():
            raise IngestError(f"PDF path does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise IngestError(f"Expected a .pdf file, got: {path}")
        shutil.copyfile(path, source_pdf)

    return SourceInfo(
        input_ref=input_ref,
        source_pdf=source_pdf,
        paper_slug=slug,
        sha256=sha256_file(source_pdf),
        run_dir=run_dir,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "paper"


def _is_url(input_ref: str) -> bool:
    parsed = urlparse(input_ref)
    return parsed.scheme in {"http", "https"}


def _source_filename(input_ref: str) -> str:
    if _is_url(input_ref):
        parsed = urlparse(input_ref)
        name = Path(unquote(parsed.path)).name
        return name if name.lower().endswith(".pdf") else "paper.pdf"
    return Path(input_ref).name


def _download_pdf(url: str, destination: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        response_path = urlparse(str(response.url)).path
        if "pdf" not in content_type.lower() and not response_path.endswith(".pdf"):
            raise IngestError(f"URL did not look like a PDF: {response.url}")
        with destination.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
