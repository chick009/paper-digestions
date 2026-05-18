"""Command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from paper_digest.config import Settings
from paper_digest.evaluate import run_evaluation_suite
from paper_digest.graph import PaperDigestWorkflow
from paper_digest.llm import LLMError, OpenRouterClient

app = typer.Typer(help="Digest research papers into Markdown, LaTeX, PDF, and trace artifacts.")


@app.callback()
def main() -> None:
    """Paper Digest command group."""


@app.command()
def digest(
    source: Annotated[str, typer.Argument(help="Local PDF path or direct PDF URL.")],
    model: Annotated[str | None, typer.Option(help="OpenRouter text model.")] = None,
    vision_model: Annotated[str | None, typer.Option(help="OpenRouter vision model.")] = None,
    output_dir: Annotated[Path, typer.Option(help="Artifact output directory.")] = Path("output"),
    vision_parse: Annotated[
        bool,
        typer.Option("--vision-parse/--no-vision-parse", help="Enable vision-model page parsing."),
    ] = True,
    vision_parse_mode: Annotated[
        str,
        typer.Option(help="Vision parsing mode: auto or all."),
    ] = "auto",
    max_vision_pages: Annotated[
        int,
        typer.Option(help="Maximum pages to send to vision model. Use 0 for no limit."),
    ] = 8,
    vision_parallelism: Annotated[
        int,
        typer.Option(help="Number of vision pages to parse concurrently."),
    ] = 5,
    force_parse: Annotated[
        bool,
        typer.Option("--force-parse/--reuse-parse", help="Refresh PDF and vision parse caches."),
    ] = False,
    reasoning: Annotated[
        bool,
        typer.Option("--reasoning/--no-reasoning", help="Request OpenRouter reasoning mode."),
    ] = True,
    trace: Annotated[
        bool,
        typer.Option("--trace/--no-trace", help="Write trace.jsonl records."),
    ] = True,
    compile_pdf: Annotated[
        bool,
        typer.Option("--compile-pdf/--no-compile-pdf", help="Compile digest.tex into digest.pdf."),
    ] = True,
) -> None:
    """Run the full paper digest workflow."""

    settings = Settings.from_env(
        model=model,
        vision_model=vision_model,
        output_dir=output_dir,
        vision_parse_enabled=vision_parse,
        vision_parse_mode=vision_parse_mode,
        max_vision_pages=max_vision_pages,
        vision_parse_concurrency=vision_parallelism,
        force_parse=force_parse,
        reasoning_enabled=reasoning,
    )
    try:
        llm = OpenRouterClient(settings)
        workflow = PaperDigestWorkflow(
            settings=settings,
            llm=llm,
            save_trace=trace,
            compile_pdf=compile_pdf,
        )
        final_state = workflow.run(source)
    except LLMError as exc:
        raise typer.BadParameter(str(exc)) from exc

    artifacts = final_state["artifacts"]
    typer.echo(f"Digest written to: {artifacts.run_dir}")
    typer.echo(f"Markdown: {artifacts.digest_md}")
    typer.echo(f"LaTeX: {artifacts.digest_tex}")
    if artifacts.digest_pdf:
        typer.echo(f"PDF: {artifacts.digest_pdf}")
    else:
        typer.echo("PDF compilation skipped or no compiler was found.")
    typer.echo(f"Trace: {artifacts.trace_jsonl}")


@app.command()
def evaluate(
    manifest: Annotated[Path, typer.Argument(help="Evaluation manifest JSON path.")],
    output_dir: Annotated[Path, typer.Option(help="Evaluation artifact output directory.")] = Path(
        "test/evaluations"
    ),
    model: Annotated[
        str,
        typer.Option(
            help="OpenRouter model for both digest and judge unless manifest overrides it."
        ),
    ] = "x-ai/grok-4.3",
    vision_model: Annotated[
        str,
        typer.Option(help="OpenRouter model for vision page parsing."),
    ] = "openai/gpt-5.4-mini",
    vision_parse: Annotated[
        bool,
        typer.Option("--vision-parse/--no-vision-parse", help="Enable vision-model page parsing."),
    ] = True,
    vision_parse_mode: Annotated[
        str,
        typer.Option(help="Vision parsing mode: auto or all."),
    ] = "all",
    max_vision_pages: Annotated[
        int,
        typer.Option(help="Maximum pages to send to vision model. Use 0 for no limit."),
    ] = 8,
    vision_parallelism: Annotated[
        int,
        typer.Option(help="Number of vision pages to parse concurrently."),
    ] = 5,
    force_parse: Annotated[
        bool,
        typer.Option("--force-parse/--reuse-parse", help="Refresh PDF and vision parse caches."),
    ] = False,
    reasoning: Annotated[
        bool,
        typer.Option("--reasoning/--no-reasoning", help="Request OpenRouter reasoning mode."),
    ] = True,
    compile_pdf: Annotated[
        bool,
        typer.Option("--compile-pdf/--no-compile-pdf", help="Compile digest.tex into digest.pdf."),
    ] = False,
) -> None:
    """Run a manifest-based digest suite and judge the generated summaries."""

    settings = Settings.from_env(
        model=model,
        vision_model=vision_model,
        output_dir=output_dir,
        vision_parse_enabled=vision_parse,
        vision_parse_mode=vision_parse_mode,
        max_vision_pages=max_vision_pages,
        vision_parse_concurrency=vision_parallelism,
        force_parse=force_parse,
        reasoning_enabled=reasoning,
    )
    try:
        llm = OpenRouterClient(settings)
        evaluations = run_evaluation_suite(
            manifest_path=manifest,
            settings=settings,
            llm=llm,
            output_dir=output_dir,
            compile_pdf=compile_pdf,
        )
    except LLMError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Evaluated {len(evaluations)} papers.")
    typer.echo(f"Judge artifacts: {output_dir / 'judge'}")


if __name__ == "__main__":
    app()
