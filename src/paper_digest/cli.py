"""Command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from paper_digest.compare import DEFAULT_BASELINE_MODELS, run_fusion_comparison
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
    multi_agent: Annotated[
        bool,
        typer.Option(
            "--multi-agent/--single-agent",
            help="Run the full analysis pipeline across multiple text models before synthesis.",
        ),
    ] = False,
    agent_models: Annotated[
        str | None,
        typer.Option(help="Comma-separated OpenRouter text models for multi-agent pipelines."),
    ] = None,
    agent_concurrency: Annotated[
        int,
        typer.Option(
            help="Parallel multi-agent candidate pipelines. Use 0 for one worker per model.",
        ),
    ] = 0,
    synthesizer_model: Annotated[
        str | None,
        typer.Option(help="OpenRouter text model that synthesizes multi-agent drafts."),
    ] = None,
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
        multi_agent_enabled=multi_agent,
        agent_models=agent_models,
        agent_concurrency=agent_concurrency,
        synthesizer_model=synthesizer_model,
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
    if settings.multi_agent_enabled:
        typer.echo(f"Multi-agent models: {', '.join(settings.agent_models)}")
        typer.echo(f"Synthesizer model: {settings.synthesizer_model}")


@app.command("compare-fusion")
def compare_fusion(
    source: Annotated[str, typer.Argument(help="Local PDF path or direct PDF URL.")],
    baseline_models: Annotated[
        str,
        typer.Option(help="Comma-separated pair of baseline OpenRouter text models."),
    ] = ",".join(DEFAULT_BASELINE_MODELS),
    fusion_outer_model: Annotated[
        str | None,
        typer.Option(help="Outer OpenRouter model used for the Fusion run."),
    ] = None,
    fusion_analysis_models: Annotated[
        str | None,
        typer.Option(help="Comma-separated OpenRouter models for the Fusion analysis panel."),
    ] = None,
    fusion_judge_model: Annotated[
        str | None,
        typer.Option(help="OpenRouter model used as the Fusion judge."),
    ] = None,
    fusion_max_tool_calls: Annotated[
        int,
        typer.Option(help="Maximum Fusion web search/fetch tool loops per inner model."),
    ] = 8,
    fusion_temperature: Annotated[
        float,
        typer.Option(help="Sampling temperature forwarded to Fusion inner calls."),
    ] = 0.2,
    force_fusion: Annotated[
        bool,
        typer.Option(
            "--force-fusion/--auto-fusion",
            help="Force the OpenRouter Fusion tool on every Fusion text step.",
        ),
    ] = True,
    comparison_model: Annotated[
        str | None,
        typer.Option(help="OpenRouter model used to judge the three generated runs."),
    ] = None,
    vision_model: Annotated[str | None, typer.Option(help="OpenRouter vision model.")] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Comparison artifact output directory."),
    ] = Path("output/comparisons"),
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
    ] = False,
) -> None:
    """Compare two single-model digests against an OpenRouter Fusion digest."""

    settings = Settings.from_env(
        model=comparison_model,
        vision_model=vision_model,
        output_dir=output_dir,
        fusion_enabled=False,
        fusion_analysis_models=fusion_analysis_models,
        fusion_judge_model=fusion_judge_model or fusion_outer_model,
        fusion_max_tool_calls=fusion_max_tool_calls,
        fusion_temperature=fusion_temperature,
        fusion_force=force_fusion,
        vision_parse_enabled=vision_parse,
        vision_parse_mode=vision_parse_mode,
        max_vision_pages=max_vision_pages,
        vision_parse_concurrency=vision_parallelism,
        force_parse=force_parse,
        reasoning_enabled=reasoning,
    )
    try:
        result = run_fusion_comparison(
            source=source,
            settings=settings,
            output_dir=output_dir,
            baseline_models=_model_csv(baseline_models, default=DEFAULT_BASELINE_MODELS),
            fusion_outer_model=fusion_outer_model or settings.fusion_judge_model,
            fusion_analysis_models=_model_csv(
                fusion_analysis_models,
                default=settings.fusion_analysis_models,
            ),
            fusion_judge_model=(
                fusion_judge_model or fusion_outer_model or settings.fusion_judge_model
            ),
            save_trace=trace,
            compile_pdf=compile_pdf,
            comparison_model=comparison_model,
        )
    except (LLMError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Compared paper: {result.paper_slug}")
    for snapshot in result.snapshots:
        typer.echo(f"{snapshot.label}: {snapshot.run_dir}")
    typer.echo(f"Comparison JSON: {result.comparison_json}")
    typer.echo(f"Comparison Markdown: {result.comparison_md}")


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
    ] = "x-ai/grok-4.3",
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


def _model_csv(value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


if __name__ == "__main__":
    app()
