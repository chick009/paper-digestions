from pathlib import Path

import pytest

from paper_digest.config import Settings
from paper_digest.schema import (
    BlogSynthesis,
    ConceptExplanation,
    ConceptExplanationSet,
    Critique,
    CritiqueSet,
    DocumentKind,
    ExtractedPaper,
    FindingsAnalysis,
    MethodologyAnalysis,
    PageText,
    PaperClassification,
    PaperMetadata,
    SourceInfo,
)

pytest.importorskip("langgraph")

from paper_digest.graph import PaperDigestWorkflow  # noqa: E402


class FakeLLM:
    def complete_json(self, *, response_model: type, **_: object) -> object:
        if response_model is PaperClassification:
            return PaperClassification(
                kind=DocumentKind.core_methodology,
                confidence=0.9,
                rationale="It introduces a method.",
                evidence_pages=[1],
            )
        if response_model is MethodologyAnalysis:
            return MethodologyAnalysis(
                overview="A staged method.",
                core_algorithm_or_method="Stage A then Stage B.",
                steps=["A", "B"],
            )
        if response_model is FindingsAnalysis:
            return FindingsAnalysis(important_findings=["Works well"], baselines=["Baseline"])
        if response_model is ConceptExplanationSet:
            return ConceptExplanationSet(
                explanations=[
                    ConceptExplanation(
                        concept="Stage A",
                        explanation="First processing stage.",
                        why_it_matters="It prepares evidence.",
                        source_pages=[1],
                    )
                ]
            )
        if response_model is CritiqueSet:
            return CritiqueSet(
                critiques=[
                    Critique(
                        lens="Novelty",
                        verdict="Incremental.",
                        sota_assessment="Not clearly SOTA.",
                    )
                ],
                synthesis="Useful but incremental.",
                final_sota_stance="incremental",
            )
        if response_model is BlogSynthesis:
            return BlogSynthesis(
                title="Mock Paper",
                article_markdown="# Mock Paper\n\nA mocked integrated digest.",
                executive_summary="A mocked digest.",
                final_assessment="Incremental but useful.",
                practical_takeaways=["Check baselines"],
                open_questions=["What about scale?"],
            )
        raise AssertionError(f"Unexpected response model: {response_model}")


def test_mocked_workflow_writes_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "mock-paper"
    source_info = SourceInfo(
        input_ref="mock.pdf",
        source_pdf=run_dir / "source.pdf",
        paper_slug="mock-paper",
        sha256="abc",
        run_dir=run_dir,
    )

    def fake_resolve_source(_: str, output_dir: Path) -> SourceInfo:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        source_info.source_pdf.write_bytes(b"%PDF")
        return source_info

    def fake_extract_pdf(_: SourceInfo) -> ExtractedPaper:
        return ExtractedPaper(
            metadata=PaperMetadata(title="Mock Paper", page_count=1, sha256="abc"),
            pages=[PageText(page_number=1, text="A method paper.", character_count=15)],
        )

    monkeypatch.setattr("paper_digest.graph.resolve_source", fake_resolve_source)
    monkeypatch.setattr("paper_digest.graph.extract_pdf", fake_extract_pdf)
    monkeypatch.setattr("paper_digest.graph.compile_latex", lambda _: None)

    workflow = PaperDigestWorkflow(
        settings=Settings(openrouter_api_key="test", output_dir=tmp_path),
        llm=FakeLLM(),  # type: ignore[arg-type]
        save_trace=True,
        compile_pdf=True,
    )
    final_state = workflow.run("mock.pdf")

    artifacts = final_state["artifacts"]
    assert artifacts.digest_md.exists()
    assert artifacts.digest_tex.exists()
    assert artifacts.critiques_md.exists()
    assert artifacts.visual_extractions_json.exists()
    assert artifacts.analysis_json.exists()
