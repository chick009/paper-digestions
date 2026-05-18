from paper_digest.render_latex import escape_latex, render_digest_latex
from tests.factories import sample_report


def test_escape_latex_special_characters() -> None:
    assert escape_latex("a_b & 50%") == r"a\_b \& 50\%"


def test_render_digest_latex_includes_document() -> None:
    rendered = render_digest_latex(sample_report())
    assert r"\documentclass" in rendered
    assert r"\section{Executive Summary}" in rendered
    assert r"Sample Paper" in rendered
