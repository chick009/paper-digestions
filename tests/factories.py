from paper_digest.schema import (
    ConceptExplanation,
    Critique,
    DigestReport,
    DocumentKind,
    EvidenceRef,
    FindingsAnalysis,
    MethodologyAnalysis,
    PaperClassification,
    PaperMetadata,
)


def sample_report() -> DigestReport:
    classification = PaperClassification(
        kind=DocumentKind.core_methodology,
        confidence=0.82,
        rationale="The source introduces a new algorithm.",
        evidence_pages=[1, 2],
        suggested_rubric=["method", "ablations"],
    )
    methodology = MethodologyAnalysis(
        overview="The method combines retrieval with generation.",
        core_algorithm_or_method="Retrieve, rerank, then generate.",
        steps=["Retrieve candidates", "Rerank evidence", "Generate answer"],
        assumptions=["The corpus contains relevant evidence."],
        important_formulas=["s(q, d) = q^T d"],
        evidence=[EvidenceRef(page=2, quote="retrieve", note="Method description")],
    )
    findings = FindingsAnalysis(
        important_findings=["Improves accuracy on the benchmark."],
        datasets_or_inputs=["Benchmark A"],
        metrics=["Accuracy"],
        baselines=["Baseline B"],
        reported_results=["+3 accuracy points"],
        limitations=["Small evaluation set"],
        evidence=[EvidenceRef(page=5, quote="+3", note="Main result")],
    )
    explanation = ConceptExplanation(
        concept="Dense retrieval",
        explanation="Documents are represented as vectors and compared by similarity.",
        why_it_matters="It controls what evidence the generator sees.",
        source_pages=[2],
        confidence=0.9,
    )
    critique = Critique(
        lens="Novelty",
        verdict="Useful but not clearly SOTA.",
        strengths=["Clear pipeline"],
        weaknesses=["Limited baselines"],
        evidence_pages=[5],
        sota_assessment="Strong but not clearly state-of-the-art.",
        missing_evidence=["Larger benchmark"],
        confidence=0.7,
    )
    return DigestReport(
        title="Sample Paper",
        subtitle="A traceable digest",
        metadata=PaperMetadata(title="Sample Paper", page_count=6, sha256="abc"),
        classification=classification,
        executive_summary="This paper proposes a retrieval-generation method.",
        methodology=methodology,
        findings=findings,
        explanations=[explanation],
        critiques=[critique],
        final_assessment="A practical incremental contribution.",
        practical_takeaways=["Use stronger baselines."],
        open_questions=["Does it scale?"],
        references=["Sample et al. 2026"],
    )
