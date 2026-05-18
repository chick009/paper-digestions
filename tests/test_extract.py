from paper_digest.extract import score_page_quality


def test_score_page_quality_flags_low_text_image_page() -> None:
    weak, notes = score_page_quality(text="short", image_count=2)
    assert weak is True
    assert "very_low_text" in notes
    assert "image_heavy_low_text" in notes


def test_score_page_quality_accepts_text_rich_page() -> None:
    weak, notes = score_page_quality(text="research text " * 200, image_count=0)
    assert weak is False
    assert notes == []
