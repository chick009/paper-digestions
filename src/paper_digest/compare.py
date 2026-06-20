"""Run and compare baseline and OpenRouter Fusion paper digests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from paper_digest.config import Settings
from paper_digest.graph import PaperDigestWorkflow
from paper_digest.llm import OpenRouterClient
from paper_digest.prompts import load_prompt
from paper_digest.schema import RunComparisonReport, RunSnapshot
from paper_digest.tracing import TraceWriter, to_jsonable

DEFAULT_BASELINE_MODELS = ("x-ai/grok-4.3", "deepseek/deepseek-v4-pro")


def _log(settings: Settings, step: str, message: str) -> None:
    if settings.verbose:
        print(f"[compare-fusion:{step}] {message}", flush=True)


@dataclass(frozen=True)
class FusionComparisonResult:
    paper_slug: str
    report_dir: Path
    comparison_json: Path
    comparison_md: Path
    snapshots: list[RunSnapshot]
    report: RunComparisonReport


LLMFactory = Callable[[Settings], OpenRouterClient]
WorkflowFactory = Callable[..., PaperDigestWorkflow]


def run_fusion_comparison(
    *,
    source: str,
    settings: Settings,
    output_dir: Path,
    baseline_models: tuple[str, ...] = DEFAULT_BASELINE_MODELS,
    fusion_outer_model: str | None = None,
    fusion_analysis_models: tuple[str, ...] | None = None,
    fusion_judge_model: str | None = None,
    save_trace: bool = True,
    compile_pdf: bool = False,
    comparison_model: str | None = None,
    llm_factory: LLMFactory = OpenRouterClient,
    workflow_factory: WorkflowFactory = PaperDigestWorkflow,
) -> FusionComparisonResult:
    """Run two single-model baselines and one Fusion variant, then compare them."""

    if len(baseline_models) != 2:
        raise ValueError("Fusion comparison requires exactly two baseline models.")

    _log(
        settings,
        "start",
        (
            f"Comparing {source} with baselines={', '.join(baseline_models)} "
            f"and fusion outer model={fusion_outer_model or settings.model}"
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [
        ("baseline-a", "single", baseline_models[0], False),
        ("baseline-b", "single", baseline_models[1], False),
        ("fusion", "fusion", fusion_outer_model or settings.model, True),
    ]
    snapshots: list[RunSnapshot] = []
    analyses: list[dict[str, Any]] = []

    paper_settings = replace(settings, output_dir=output_dir)
    prepare_workflow = workflow_factory(
        settings=paper_settings,
        llm=llm_factory(paper_settings),
        save_trace=save_trace,
        compile_pdf=compile_pdf,
    )
    _log(settings, "prepare", f"Parsing PDF once into shared directory under {output_dir}")
    prepared = prepare_workflow.prepare_paper(source)
    _log(
        settings,
        "prepare",
        (
            f"Shared artifacts at {prepared.paper_dir} "
            f"({prepared.extracted_paper.metadata.page_count} pages, "
            f"{len(prepared.extracted_paper.visual_extractions)} vision page(s))"
        ),
    )

    for label, mode, model, fusion_enabled in variants:
        variant_dir = prepared.paper_dir / label
        _log(
            settings,
            label,
            (
                f"Starting {mode} run with model={model}, fusion={fusion_enabled}, "
                f"variant_dir={variant_dir}"
            ),
        )
        variant_settings = _variant_settings(
            settings,
            output_dir=variant_dir,
            model=model,
            fusion_enabled=fusion_enabled,
            fusion_analysis_models=fusion_analysis_models,
            fusion_judge_model=fusion_judge_model,
        )
        workflow = workflow_factory(
            settings=variant_settings,
            llm=llm_factory(variant_settings),
            save_trace=save_trace,
            compile_pdf=compile_pdf,
        )
        final_state = workflow.run_variant(prepared, variant_dir=variant_dir)
        snapshot, analysis = _snapshot_from_state(
            label=label,
            fallback_mode=mode,
            fallback_model=model,
            final_state=final_state,
        )
        snapshots.append(snapshot)
        analyses.append(analysis)
        quality_status = (
            "passed"
            if snapshot.quality_check and snapshot.quality_check.get("passed")
            else "failed/unknown"
        )
        _log(
            settings,
            label,
            f"Completed run at {snapshot.run_dir} (quality_check={quality_status})",
        )

    paper_slug = _paper_slug(snapshots) or prepared.paper_dir.name
    report_dir = output_dir / "_reports" / paper_slug
    report_dir.mkdir(parents=True, exist_ok=True)
    judge_model = comparison_model or settings.model
    _log(settings, "judge", f"Comparing {len(snapshots)} runs with model={judge_model}")
    trace = TraceWriter(report_dir, enabled=save_trace)
    judge_settings = replace(settings, output_dir=report_dir, fusion_enabled=False)
    report = llm_factory(judge_settings).complete_json(
        step=f"compare_fusion_{paper_slug}",
        system_prompt=load_prompt("compare_runs.md"),
        user_prompt=_comparison_context(
            source=source,
            paper_slug=paper_slug,
            snapshots=snapshots,
            analyses=analyses,
        ),
        response_model=RunComparisonReport,
        trace=trace,
        model=comparison_model or settings.model,
        prompt_version="compare-runs-v1",
    )
    _log(
        settings,
        "judge",
        f"Comparison complete; recommended_run={report.recommended_run!r}",
    )

    comparison_json = report_dir / "comparison.json"
    comparison_md = report_dir / "comparison.md"
    comparison_payload = {
        "source": source,
        "paper_slug": paper_slug,
        "snapshots": to_jsonable(snapshots),
        "report": to_jsonable(report),
    }
    comparison_json.write_text(
        json.dumps(comparison_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    comparison_md.write_text(render_comparison_markdown(report, snapshots), encoding="utf-8")
    trace.set_analysis("comparison", comparison_payload)
    trace.write_analysis()
    _log(settings, "done", f"Wrote {comparison_json} and {comparison_md}")

    return FusionComparisonResult(
        paper_slug=paper_slug,
        report_dir=report_dir,
        comparison_json=comparison_json,
        comparison_md=comparison_md,
        snapshots=snapshots,
        report=report,
    )


def render_comparison_markdown(
    report: RunComparisonReport,
    snapshots: list[RunSnapshot],
) -> str:
    lines = [
        f"# Fusion Comparison: {report.paper_id}",
        "",
        "## Compared Runs",
    ]
    for snapshot in snapshots:
        lines.extend(
            [
                f"- `{snapshot.label}`: `{snapshot.mode}` using `{snapshot.model}`",
                f"  - run_dir: `{snapshot.run_dir}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            f"Recommended run: `{report.recommended_run}`",
            "",
            report.recommendation_rationale,
            "",
            "## Fusion Assessment",
            report.fusion_assessment,
            "",
            "## Consensus",
            *_bullet_lines(report.consensus),
            "",
            "## Disagreements",
            *_bullet_lines(report.disagreements),
            "",
            "## Criteria",
        ]
    )
    for criterion in report.criteria:
        best_run_line = (
            f"Best run: `{criterion.best_run}`" if criterion.best_run else "Best run: not specified"
        )
        lines.extend(
            [
                f"### {criterion.criterion}",
                best_run_line,
                "",
                f"Scores: {_format_scores(criterion.scores)}",
                "",
                criterion.rationale,
                "",
            ]
        )
        if criterion.concerns:
            lines.extend(["Concerns:", *_bullet_lines(criterion.concerns), ""])

    lines.extend(["## Unique Strengths"])
    for label, strengths in report.unique_strengths.items():
        lines.extend([f"### {label}", *_bullet_lines(strengths), ""])

    lines.extend(["## Blind Spots"])
    for label, blind_spots in report.blind_spots.items():
        lines.extend([f"### {label}", *_bullet_lines(blind_spots), ""])

    lines.extend(["## Revision Suggestions", *_bullet_lines(report.revision_suggestions), ""])
    return "\n".join(lines).strip() + "\n"


def _variant_settings(
    settings: Settings,
    *,
    output_dir: Path,
    model: str,
    fusion_enabled: bool,
    fusion_analysis_models: tuple[str, ...] | None,
    fusion_judge_model: str | None,
) -> Settings:
    return replace(
        settings,
        model=model,
        output_dir=output_dir,
        multi_agent_enabled=False,
        fusion_enabled=fusion_enabled,
        fusion_analysis_models=fusion_analysis_models or settings.fusion_analysis_models,
        fusion_judge_model=fusion_judge_model or settings.fusion_judge_model or model,
    )


def _snapshot_from_state(
    *,
    label: str,
    fallback_mode: str,
    fallback_model: str,
    final_state: dict[str, Any],
) -> tuple[RunSnapshot, dict[str, Any]]:
    artifacts = final_state["artifacts"]
    analysis_path = artifacts.analysis_json
    analysis = _read_json(analysis_path)
    run_config = analysis.get("run_config") if isinstance(analysis.get("run_config"), dict) else {}
    digest_text = (
        artifacts.digest_md.read_text(encoding="utf-8") if artifacts.digest_md.exists() else ""
    )
    snapshot = RunSnapshot(
        label=label,
        mode=str(run_config.get("mode", fallback_mode)),
        model=str(run_config.get("model", fallback_model)),
        run_dir=artifacts.run_dir,
        digest_md=artifacts.digest_md,
        analysis_json=analysis_path,
        digest_excerpt=_excerpt(digest_text, 12000),
        run_config=run_config,
        quality_check=analysis.get("quality_check")
        if isinstance(analysis.get("quality_check"), dict)
        else None,
    )
    return snapshot, analysis


def _comparison_context(
    *,
    source: str,
    paper_slug: str,
    snapshots: list[RunSnapshot],
    analyses: list[dict[str, Any]],
) -> str:
    blocks = []
    for snapshot, analysis in zip(snapshots, analyses, strict=True):
        analysis_excerpt = json.dumps(_analysis_brief(analysis), indent=2, ensure_ascii=False)
        blocks.append(
            "\n".join(
                [
                    f"Run label: {snapshot.label}",
                    f"Mode: {snapshot.mode}",
                    f"Model: {snapshot.model}",
                    f"Run config: {json.dumps(snapshot.run_config, ensure_ascii=False)}",
                    "",
                    "Digest markdown:",
                    snapshot.digest_excerpt,
                    "",
                    "Structured analysis excerpt:",
                    _excerpt(analysis_excerpt, 30000),
                ]
            )
        )
    return (
        f"Source: {source}\n"
        f"Paper ID: {paper_slug}\n"
        f"Compared runs: {', '.join(snapshot.label for snapshot in snapshots)}\n\n"
        "Compare these runs as outputs for the same paper. Use run labels exactly as provided.\n\n"
        + "\n\n---\n\n".join(blocks)
    )


def _analysis_brief(analysis: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "metadata",
        "classification",
        "methodology",
        "findings",
        "explanations",
        "critiques",
        "report",
        "quality_check",
        "run_config",
    ]
    return {key: analysis[key] for key in keys if key in analysis}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _paper_slug(snapshots: list[RunSnapshot]) -> str:
    for snapshot in snapshots:
        if snapshot.run_config.get("source_info", {}).get("paper_slug"):
            return str(snapshot.run_config["source_info"]["paper_slug"])
        if snapshot.run_dir.name:
            return snapshot.run_dir.name
    return "paper"


def _excerpt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return (
        f"{text[:head].rstrip()}\n\n"
        "[MIDDLE TRUNCATED FOR COMPARISON]\n\n"
        f"{text[-tail:].lstrip()}"
    )


def _bullet_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- None noted."]


def _format_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "not provided"
    return ", ".join(f"{label}={score:.1f}/5" for label, score in scores.items())
