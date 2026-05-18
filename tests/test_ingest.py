from pathlib import Path

from paper_digest.ingest import sha256_file, slugify


def test_slugify_keeps_readable_ascii_slug() -> None:
    assert slugify("My Paper: A New Method!") == "my-paper-a-new-method"


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"paper")
    assert sha256_file(path) == "382635c9325bf3273d195ff1b8a44e5b11afd7d97addeb8863ea35feb98c1a07"
