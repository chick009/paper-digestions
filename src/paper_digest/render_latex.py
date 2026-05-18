"""LaTeX rendering and PDF compilation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from paper_digest.schema import DigestReport

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def render_digest_latex(report: DigestReport) -> str:
    if report.article_markdown.strip():
        return render_article_markdown_latex(report)

    title = escape_latex(report.title)
    subtitle = escape_latex(report.subtitle or "")
    body = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{enumitem}",
        r"\setlist{nosep}",
        r"\title{" + title + r"}",
        r"\author{Paper Digest}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
    ]
    if subtitle:
        body.extend([r"\begin{center}\emph{" + subtitle + r"}\end{center}", ""])

    body.extend(
        [
            section("Executive Summary", report.executive_summary),
            section(
                "Classification",
                (
                    f"Type: {report.classification.kind.value}. "
                    f"Confidence: {report.classification.confidence:.2f}. "
                    f"{report.classification.rationale}"
                ),
            ),
            section("Core Methodology", report.methodology.overview),
            paragraph("Core method", report.methodology.core_algorithm_or_method),
            itemize("Method Steps", report.methodology.steps),
            itemize("Important Formulas", report.methodology.important_formulas),
            itemize("Important Findings", report.findings.important_findings),
            itemize("Reported Results", report.findings.reported_results),
            itemize("Limitations", report.findings.limitations),
            r"\section{Concept And Formula Explanations}",
        ]
    )
    for explanation in report.explanations:
        body.extend(
            [
                r"\subsection{" + escape_latex(explanation.concept) + r"}",
                escape_latex(explanation.explanation),
                "",
                paragraph("Why it matters", explanation.why_it_matters),
                paragraph("Evidence pages", _pages(explanation.source_pages)),
            ]
        )

    body.append(r"\section{Critique}")
    for critique in report.critiques:
        body.extend(
            [
                r"\subsection{" + escape_latex(critique.lens) + r"}",
                paragraph("Verdict", critique.verdict),
                paragraph("SOTA assessment", critique.sota_assessment),
                itemize("Strengths", critique.strengths),
                itemize("Weaknesses", critique.weaknesses),
                paragraph("Evidence pages", _pages(critique.evidence_pages)),
            ]
        )

    body.extend(
        [
            section("Final Assessment", report.final_assessment),
            itemize("Practical Takeaways", report.practical_takeaways),
            itemize("Open Questions", report.open_questions),
            itemize("References", report.references),
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(part for part in body if part is not None)


def render_article_markdown_latex(report: DigestReport) -> str:
    title = escape_latex(report.title)
    subtitle = escape_latex(report.subtitle or "")
    body = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{enumitem}",
        r"\setlist{nosep}",
        r"\title{" + title + r"}",
        r"\author{Paper Digest}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
    ]
    if subtitle:
        body.extend([r"\begin{center}\emph{" + subtitle + r"}\end{center}", ""])
    body.extend(_markdown_blocks_to_latex(report.article_markdown))
    body.extend([r"\end{document}", ""])
    return "\n".join(part for part in body if part is not None)


def compile_latex(tex_path: Path) -> Path | None:
    """Compile LaTeX if a supported compiler is available."""

    if compiler := shutil.which("tectonic"):
        return _run_compiler([compiler, tex_path.name], tex_path)
    if compiler := shutil.which("latexmk"):
        return _run_compiler(
            [compiler, "-pdf", "-interaction=nonstopmode", tex_path.name],
            tex_path,
        )
    if compiler := shutil.which("pdflatex"):
        return _run_compiler([compiler, "-interaction=nonstopmode", tex_path.name], tex_path)

    marker = tex_path.with_suffix(".pdf.skipped.txt")
    marker.write_text(
        "No LaTeX compiler found. Install tectonic, latexmk, or pdflatex to build digest.pdf.\n",
        encoding="utf-8",
    )
    return None


def _run_compiler(command: list[str], tex_path: Path) -> Path | None:
    try:
        subprocess.run(command, cwd=tex_path.parent, check=True)
    except subprocess.CalledProcessError as exc:
        marker = tex_path.with_suffix(".pdf.failed.txt")
        marker.write_text(f"LaTeX compilation failed: {exc}\n", encoding="utf-8")
        return None
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def escape_latex(value: str) -> str:
    return "".join(LATEX_SPECIALS.get(character, character) for character in value)


def section(title: str, content: str) -> str:
    return rf"\section{{{escape_latex(title)}}}" + "\n" + escape_latex(content) + "\n"


def paragraph(label: str, content: str) -> str:
    return rf"\paragraph{{{escape_latex(label)}}} {escape_latex(content)}" + "\n"


def itemize(title: str, items: list[str]) -> str:
    if not items:
        return ""
    lines = [rf"\subsection{{{escape_latex(title)}}}", r"\begin{itemize}"]
    lines.extend(r"\item " + escape_latex(item) for item in items)
    lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n"


def _markdown_blocks_to_latex(markdown: str) -> list[str]:
    blocks: list[str] = []
    list_items: list[str] = []

    def flush_list() -> None:
        if not list_items:
            return
        blocks.append(r"\begin{itemize}")
        blocks.extend(r"\item " + escape_latex(item) for item in list_items)
        blocks.append(r"\end{itemize}")
        list_items.clear()

    for raw_line in markdown.strip().splitlines():
        line = raw_line.strip()
        if not line:
            flush_list()
            blocks.append("")
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            flush_list()
            blocks.append(r"\section{" + escape_latex(line.removeprefix("## ").strip()) + r"}")
            continue
        if line.startswith("### "):
            flush_list()
            blocks.append(r"\subsection{" + escape_latex(line.removeprefix("### ").strip()) + r"}")
            continue
        if line.startswith("- "):
            list_items.append(line.removeprefix("- ").strip())
            continue
        if line.startswith("|"):
            flush_list()
            blocks.append(r"\texttt{" + escape_latex(line) + r"}")
            continue
        flush_list()
        blocks.append(escape_latex(line) + "\n")

    flush_list()
    return blocks


def _pages(pages: list[int]) -> str:
    return ", ".join(str(page) for page in pages) if pages else "not specified"
