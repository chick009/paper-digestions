from paper_digest.render_markdown import render_digest_markdown
from tests.factories import sample_report


def test_render_digest_markdown_includes_major_sections() -> None:
    rendered = render_digest_markdown(sample_report())
    assert "# Sample Paper" in rendered
    assert "## Core Methodology" in rendered
    assert "## Concept And Formula Explanations" in rendered
    assert "## Critique" in rendered
    assert "`core_methodology`" in rendered
