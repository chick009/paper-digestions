"""Manifest-driven digest evaluation with an LLM judge."""

from __future__ import annotations

import json
from pathlib import Path

from paper_digest.config import Settings
from paper_digest.graph import PaperDigestWorkflow
from paper_digest.llm import OpenRouterClient
from paper_digest.prompts import load_prompt
from paper_digest.schema import EvaluationCase, EvaluationManifest, JudgeEvaluation
from paper_digest.tracing import TraceWriter


def load_manifest(path: Path) -> EvaluationManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    manifest = EvaluationManifest.model_validate(raw)
    base_dir = path.parent.parent.parent if path.parent.name == "expectations" else path.parent
    resolved_cases: list[EvaluationCase] = []
    for case in manifest.cases:
        pdf_path = case.pdf_path
        if not pdf_path.is_absolute():
            pdf_path = (base_dir / pdf_path).resolve()
        resolved_cases.append(case.model_copy(update={"pdf_path": pdf_path}))
    return manifest.model_copy(update={"cases": resolved_cases})


def run_evaluation_suite(
    *,
    manifest_path: Path,
    settings: Settings,
    llm: OpenRouterClient,
    output_dir: Path,
    compile_pdf: bool = False,
) -> list[JudgeEvaluation]:
    manifest = load_manifest(manifest_path)
    suite_dir = output_dir / "judge"
    suite_dir.mkdir(parents=True, exist_ok=True)

    digest_settings = Settings.from_env(
        model=manifest.model,
        vision_model=manifest.vision_model,
        output_dir=output_dir / "digests",
        vision_parse_enabled=settings.vision_parse_enabled,
        vision_parse_mode=settings.vision_parse_mode,
        max_vision_pages=settings.max_vision_pages,
        vision_parse_concurrency=settings.vision_parse_concurrency,
        force_parse=settings.force_parse,
        reasoning_enabled=settings.reasoning_enabled,
        reasoning_effort=settings.reasoning_effort,
        verbose=settings.verbose,
    )
    digest_llm = OpenRouterClient(digest_settings)
    workflow = PaperDigestWorkflow(
        settings=digest_settings,
        llm=digest_llm,
        save_trace=True,
        compile_pdf=compile_pdf,
    )

    evaluations: list[JudgeEvaluation] = []
    for case in manifest.cases:
        if settings.verbose:
            print(f"[evaluate] Running digest for {case.id}: {case.pdf_path}", flush=True)
        final_state = workflow.run(str(case.pdf_path))
        digest_md = final_state["artifacts"].digest_md.read_text(encoding="utf-8")
        analysis_json = final_state["artifacts"].analysis_json.read_text(encoding="utf-8")
        case_dir = suite_dir / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        trace = TraceWriter(case_dir, enabled=True)
        if settings.verbose:
            print(f"[evaluate] Judging digest for {case.id}", flush=True)
        judge = judge_digest(
            case=case,
            digest_markdown=digest_md,
            analysis_json=analysis_json,
            llm=llm,
            trace=trace,
            model=manifest.judge_model,
        )
        evaluations.append(judge)
        (case_dir / "judge.json").write_text(
            judge.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (case_dir / "judge.md").write_text(render_judge_markdown(judge), encoding="utf-8")
        trace.write_analysis()

    (suite_dir / "summary.md").write_text(render_suite_summary(evaluations), encoding="utf-8")
    (suite_dir / "summary.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in evaluations], indent=2),
        encoding="utf-8",
    )
    return evaluations


def judge_digest(
    *,
    case: EvaluationCase,
    digest_markdown: str,
    analysis_json: str,
    llm: OpenRouterClient,
    trace: TraceWriter,
    model: str,
) -> JudgeEvaluation:
    expected_claims = json.dumps(
        [item.model_dump(mode="json") for item in case.expected_claims],
        indent=2,
        ensure_ascii=False,
    )
    questions = json.dumps(case.questions, indent=2, ensure_ascii=False)
    expected_sections = json.dumps(case.expected_sections, indent=2, ensure_ascii=False)
    user_prompt = (
        f"Paper ID: {case.id}\n"
        f"Title: {case.title}\n"
        f"Reference blog URL: {case.final_blog_url or 'not provided'}\n\n"
        f"Reference blog markdown:\n{case.final_blog_markdown or 'not provided'}\n\n"
        f"User questions:\n{questions}\n\n"
        "Expected claims and result requirements:\n"
        f"{expected_claims}\n\n"
        f"Expected blog sections:\n{expected_sections}\n\n"
        f"Generated digest markdown:\n{digest_markdown}\n\n"
        f"Structured analysis JSON:\n{analysis_json[:50000]}"
    )
    return llm.complete_json(
        step=f"judge_{case.id}",
        system_prompt=load_prompt("judge.md"),
        user_prompt=user_prompt,
        response_model=JudgeEvaluation,
        trace=trace,
        model=model,
        prompt_version="judge-v1",
    )


def render_judge_markdown(evaluation: JudgeEvaluation) -> str:
    lines = [
        f"# Judge Evaluation: {evaluation.paper_id}",
        "",
        f"- Overall score: {evaluation.overall_score:.1f}/5",
        f"- Concise summary score: {evaluation.concise_summary_score:.1f}/5",
        f"- Claim coverage score: {evaluation.claim_coverage_score:.1f}/5",
        f"- Evidence quality score: {evaluation.evidence_quality_score:.1f}/5",
        f"- Critique quality score: {evaluation.critique_quality_score:.1f}/5",
        f"- Surpasses baseline expectation: {evaluation.surpasses_baseline_expectation}",
        "",
        "## Criteria",
    ]
    for criterion in evaluation.criteria:
        lines.extend(
            [
                f"### {criterion.criterion}",
                f"Score: {criterion.score:.1f}/5",
                "",
                criterion.rationale,
                "",
            ]
        )
        if criterion.missing_or_weak_points:
            lines.extend(["Missing or weak points:", *[
                f"- {item}" for item in criterion.missing_or_weak_points
            ], ""])
    lines.extend(["## Strengths", *[f"- {item}" for item in evaluation.strengths], ""])
    lines.extend(["## Weaknesses", *[f"- {item}" for item in evaluation.weaknesses], ""])
    lines.extend(
        [
            "## Recommended Revisions",
            *[f"- {item}" for item in evaluation.recommended_revisions],
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_suite_summary(evaluations: list[JudgeEvaluation]) -> str:
    if not evaluations:
        return "# Judge Summary\n\nNo evaluations were produced.\n"
    average = sum(item.overall_score for item in evaluations) / len(evaluations)
    lines = ["# Judge Summary", "", f"Average overall score: {average:.1f}/5", ""]
    for evaluation in evaluations:
        lines.extend(
            [
                f"## {evaluation.paper_id}",
                f"- Overall score: {evaluation.overall_score:.1f}/5",
                f"- Surpasses baseline expectation: {evaluation.surpasses_baseline_expectation}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"
