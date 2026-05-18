"""Markdown rendering for digest reports."""

from __future__ import annotations

from paper_digest.schema import CritiqueSet, DigestReport


def render_digest_markdown(report: DigestReport) -> str:
    if report.article_markdown.strip():
        return _render_article_markdown(report)

    lines = [
        f"# {report.title}",
        "",
    ]
    if report.subtitle:
        lines.extend([f"*{report.subtitle}*", ""])
    lines.extend(
        [
            "## Executive Summary",
            report.executive_summary,
            "",
            "## Classification",
            f"- Type: `{report.classification.kind.value}`",
            f"- Confidence: {report.classification.confidence:.2f}",
            f"- Rationale: {report.classification.rationale}",
            "",
            "## Core Methodology",
            report.methodology.overview,
            "",
            f"**Core method:** {report.methodology.core_algorithm_or_method}",
            "",
        ]
    )
    if report.methodology.steps:
        lines.extend(["### Method Steps", *[f"- {step}" for step in report.methodology.steps], ""])
    if report.methodology.important_formulas:
        lines.extend(
            [
                "### Important Formulas",
                *[f"- `{formula}`" for formula in report.methodology.important_formulas],
                "",
            ]
        )

    lines.extend(["## Important Findings", ""])
    lines.extend(_bullet_list(report.findings.important_findings, empty="No findings extracted."))
    if report.findings.reported_results:
        lines.extend(["", "### Reported Results", *_bullet_list(report.findings.reported_results)])
    if report.findings.limitations:
        lines.extend(["", "### Limitations", *_bullet_list(report.findings.limitations)])

    if report.explanations:
        lines.extend(["", "## Concept And Formula Explanations", ""])
        for explanation in report.explanations:
            pages = _pages(explanation.source_pages)
            lines.extend(
                [
                    f"### {explanation.concept}",
                    f"{explanation.explanation}",
                    "",
                    f"**Why it matters:** {explanation.why_it_matters}",
                    f"**Evidence pages:** {pages}",
                    "",
                ]
            )

    lines.extend(["## Critique", ""])
    for critique in report.critiques:
        lines.extend(
            [
                f"### {critique.lens}",
                f"**Verdict:** {critique.verdict}",
                "",
                f"**SOTA assessment:** {critique.sota_assessment}",
                "",
                "**Strengths:**",
                *_bullet_list(critique.strengths, empty="None listed."),
                "",
                "**Weaknesses:**",
                *_bullet_list(critique.weaknesses, empty="None listed."),
                "",
                f"**Evidence pages:** {_pages(critique.evidence_pages)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Final Assessment",
            report.final_assessment,
            "",
            "## Practical Takeaways",
            *_bullet_list(report.practical_takeaways, empty="No takeaways extracted."),
            "",
            "## Open Questions",
            *_bullet_list(report.open_questions, empty="No open questions extracted."),
            "",
        ]
    )
    if report.references:
        lines.extend(["## References", *_bullet_list(report.references), ""])
    return "\n".join(lines).strip() + "\n"


def render_critiques_markdown(critique_set: CritiqueSet) -> str:
    lines = [
        "# Critiques And SOTA Assessment",
        "",
        f"Final stance: `{critique_set.final_sota_stance}`",
        "",
        "## Synthesis",
        critique_set.synthesis,
        "",
    ]
    for critique in critique_set.critiques:
        lines.extend(
            [
                f"## {critique.lens}",
                f"Verdict: {critique.verdict}",
                "",
                f"SOTA assessment: {critique.sota_assessment}",
                "",
                "Strengths:",
                *_bullet_list(critique.strengths),
                "",
                "Weaknesses:",
                *_bullet_list(critique.weaknesses),
                "",
                "Missing evidence:",
                *_bullet_list(critique.missing_evidence, empty="None listed."),
                "",
            ]
        )
    if critique_set.uncertainty:
        lines.extend(["## Uncertainty", *_bullet_list(critique_set.uncertainty), ""])
    return "\n".join(lines).strip() + "\n"


def _render_article_markdown(report: DigestReport) -> str:
    article = report.article_markdown.strip()
    if article.startswith("# "):
        return article + "\n"

    lines = [f"# {report.title}", ""]
    if report.subtitle:
        lines.extend([f"*{report.subtitle}*", ""])
    lines.append(article)
    return "\n".join(lines).strip() + "\n"


def _bullet_list(items: list[str], *, empty: str | None = None) -> list[str]:
    if not items and empty:
        return [f"- {empty}"]
    return [f"- {item}" for item in items]


def _pages(pages: list[int]) -> str:
    return ", ".join(str(page) for page in pages) if pages else "not specified"
