"""Shared data models for the paper digest workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class DocumentKind(StrEnum):
    empirical_findings = "empirical_findings"
    core_methodology = "core_methodology"
    blog_reference = "blog_reference"


class SourceInfo(BaseModel):
    input_ref: str
    source_pdf: Path
    paper_slug: str
    sha256: str
    run_dir: Path
    parse_dir: Path | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def parse_root(self) -> Path:
        """Directory holding shared PDF parse and vision artifacts."""

        return self.parse_dir if self.parse_dir is not None else self.run_dir


class PageText(BaseModel):
    page_number: int
    text: str
    image_count: int = 0
    character_count: int = 0
    weak_text: bool = False
    quality_notes: list[str] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    subject: str | None = None
    keywords: list[str] = Field(default_factory=list)
    page_count: int = 0
    source_url: str | None = None
    sha256: str | None = None


class VisualExtraction(BaseModel):
    page_number: int
    image_path: str | None = None
    extracted_text: str = ""
    visual_summary: str = ""
    equations: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class ExtractedPaper(BaseModel):
    metadata: PaperMetadata
    pages: list[PageText]
    visual_extractions: list[VisualExtraction] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[Page {page.page_number}]\n{page.text}" for page in self.pages)

    @property
    def weak_pages(self) -> list[int]:
        return [page.page_number for page in self.pages if page.weak_text]


class EvidenceRef(BaseModel):
    page: int | None = None
    quote: str | None = None
    note: str


class PaperClassification(BaseModel):
    kind: DocumentKind
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_pages: list[int] = Field(default_factory=list)
    suggested_rubric: list[str] = Field(default_factory=list)


class MethodologyAnalysis(BaseModel):
    overview: str
    core_algorithm_or_method: str
    steps: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    important_formulas: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class FindingsAnalysis(BaseModel):
    important_findings: list[str] = Field(default_factory=list)
    datasets_or_inputs: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    reported_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ConceptExplanation(BaseModel):
    concept: str
    explanation: str
    why_it_matters: str
    source_pages: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    follow_up_questions: list[str] = Field(default_factory=list)


class ConceptExplanationSet(BaseModel):
    explanations: list[ConceptExplanation] = Field(default_factory=list)


class Critique(BaseModel):
    lens: str
    verdict: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    evidence_pages: list[int] = Field(default_factory=list)
    sota_assessment: str
    missing_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CritiqueSet(BaseModel):
    critiques: list[Critique] = Field(default_factory=list)
    synthesis: str
    final_sota_stance: Literal[
        "likely_sota",
        "strong_but_not_clearly_sota",
        "incremental",
        "insufficient_evidence",
        "reference_only",
    ]
    uncertainty: list[str] = Field(default_factory=list)


class BlogSynthesis(BaseModel):
    title: str
    subtitle: str | None = None
    article_markdown: str
    executive_summary: str
    final_assessment: str
    practical_takeaways: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class ModelPipelineDraft(BaseModel):
    model: str
    classification: PaperClassification
    methodology: MethodologyAnalysis
    findings: FindingsAnalysis
    explanations: ConceptExplanationSet
    critiques: CritiqueSet
    blog: BlogSynthesis


class DigestReport(BaseModel):
    title: str
    subtitle: str | None = None
    article_markdown: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: PaperMetadata
    classification: PaperClassification
    executive_summary: str
    methodology: MethodologyAnalysis
    findings: FindingsAnalysis
    explanations: list[ConceptExplanation] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    final_assessment: str
    practical_takeaways: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class QualityCheck(BaseModel):
    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    weak_sota_claims: list[str] = Field(default_factory=list)
    missing_baselines: list[str] = Field(default_factory=list)
    unclear_formulas: list[str] = Field(default_factory=list)
    revision_notes: list[str] = Field(default_factory=list)


class ExpectedClaim(BaseModel):
    claim: str
    required_results: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    id: str
    title: str
    pdf_path: Path
    final_blog_url: str | None = None
    final_blog_markdown: str | None = None
    questions: list[str] = Field(default_factory=list)
    expected_claims: list[ExpectedClaim] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)


class EvaluationManifest(BaseModel):
    model: str = "x-ai/grok-4.3"
    vision_model: str = "x-ai/grok-4.3"
    judge_model: str = "x-ai/grok-4.3"
    cases: list[EvaluationCase]


class JudgeCriterionScore(BaseModel):
    criterion: str
    score: float = Field(ge=0.0, le=5.0)
    rationale: str
    missing_or_weak_points: list[str] = Field(default_factory=list)


class JudgeEvaluation(BaseModel):
    paper_id: str
    overall_score: float = Field(ge=0.0, le=5.0)
    concise_summary_score: float = Field(ge=0.0, le=5.0)
    claim_coverage_score: float = Field(ge=0.0, le=5.0)
    evidence_quality_score: float = Field(ge=0.0, le=5.0)
    critique_quality_score: float = Field(ge=0.0, le=5.0)
    criteria: list[JudgeCriterionScore] = Field(default_factory=list)
    surpasses_baseline_expectation: bool
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommended_revisions: list[str] = Field(default_factory=list)


class RunSnapshot(BaseModel):
    label: str
    mode: str
    model: str
    run_dir: Path
    digest_md: Path | None = None
    analysis_json: Path | None = None
    digest_excerpt: str = ""
    run_config: dict[str, Any] = Field(default_factory=dict)
    quality_check: dict[str, Any] | None = None


class RunComparisonCriterion(BaseModel):
    criterion: str
    scores: dict[str, float] = Field(default_factory=dict)
    best_run: str | None = None
    rationale: str
    concerns: list[str] = Field(default_factory=list)


class RunComparisonReport(BaseModel):
    paper_id: str
    compared_runs: list[str] = Field(default_factory=list)
    consensus: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    unique_strengths: dict[str, list[str]] = Field(default_factory=dict)
    blind_spots: dict[str, list[str]] = Field(default_factory=dict)
    criteria: list[RunComparisonCriterion] = Field(default_factory=list)
    recommended_run: str
    recommendation_rationale: str
    fusion_assessment: str
    revision_suggestions: list[str] = Field(default_factory=list)


class ArtifactPaths(BaseModel):
    run_dir: Path
    digest_md: Path
    digest_tex: Path
    digest_pdf: Path | None = None
    trace_jsonl: Path
    analysis_json: Path
    visual_extractions_json: Path
    critiques_md: Path


class PaperDigestState(TypedDict, total=False):
    input_ref: str
    source_info: SourceInfo
    extracted_paper: ExtractedPaper
    classification: PaperClassification
    methodology: MethodologyAnalysis
    findings: FindingsAnalysis
    explanations: ConceptExplanationSet
    critiques: CritiqueSet
    model_pipeline_drafts: list[ModelPipelineDraft]
    report: DigestReport
    quality_check: QualityCheck
    artifacts: ArtifactPaths
    errors: list[str]
    analysis: dict[str, Any]
