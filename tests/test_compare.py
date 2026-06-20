from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_digest.compare import run_fusion_comparison
from paper_digest.config import Settings
from paper_digest.graph import PreparedPaper
from paper_digest.schema import ArtifactPaths, ExtractedPaper, RunComparisonReport, SourceInfo


def test_run_fusion_comparison_writes_labeled_runs_and_report(tmp_path: Path) -> None:
    workflow_settings: list[Settings] = []
    prepare_calls = 0
    variant_runs: list[str] = []

    class FakeWorkflow:
        def __init__(
            self,
            *,
            settings: Settings,
            llm: object,
            save_trace: bool,
            compile_pdf: bool,
        ) -> None:
            self.settings = settings
            workflow_settings.append(settings)

        def prepare_paper(self, source: str) -> PreparedPaper:
            nonlocal prepare_calls
            prepare_calls += 1
            paper_dir = self.settings.output_dir / "mock-paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            (paper_dir / "source.pdf").write_bytes(b"%PDF")
            (paper_dir / "extracted_paper.json").write_text("{}", encoding="utf-8")
            (paper_dir / "visual_extractions.json").write_text("[]", encoding="utf-8")
            source_info = SourceInfo(
                input_ref=source,
                source_pdf=paper_dir / "source.pdf",
                paper_slug="mock-paper",
                sha256="abc",
                run_dir=paper_dir,
            )
            return PreparedPaper(
                paper_dir=paper_dir,
                input_ref=source,
                source_info=source_info,
                extracted_paper=ExtractedPaper.model_validate(
                    {
                        "metadata": {
                            "title": "Mock",
                            "page_count": 1,
                            "sha256": "abc",
                        },
                        "pages": [{"page_number": 1, "text": "text", "character_count": 4}],
                    }
                ),
            )

        def run_variant(self, prepared: PreparedPaper, *, variant_dir: Path) -> dict[str, Any]:
            variant_runs.append(variant_dir.name)
            run_dir = variant_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            digest_md = run_dir / "digest.md"
            digest_tex = run_dir / "digest.tex"
            critiques_md = run_dir / "critiques.md"
            analysis_json = run_dir / "analysis.json"
            trace_jsonl = run_dir / "trace.jsonl"
            digest_md.write_text(
                f"# {self.settings.model}\n\nDigest for {prepared.input_ref}.",
                encoding="utf-8",
            )
            digest_tex.write_text("", encoding="utf-8")
            critiques_md.write_text("", encoding="utf-8")
            trace_jsonl.write_text("", encoding="utf-8")
            mode = "fusion" if self.settings.fusion_enabled else "single"
            analysis_json.write_text(
                json.dumps(
                    {
                        "run_config": {
                            "mode": mode,
                            "model": self.settings.model,
                            "source_info": {"paper_slug": "mock-paper"},
                            "fusion": {
                                "analysis_models": list(
                                    self.settings.fusion_analysis_models
                                )
                            }
                            if self.settings.fusion_enabled
                            else None,
                        },
                        "quality_check": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            return {
                "artifacts": ArtifactPaths(
                    run_dir=run_dir,
                    digest_md=digest_md,
                    digest_tex=digest_tex,
                    digest_pdf=None,
                    trace_jsonl=trace_jsonl,
                    analysis_json=analysis_json,
                    visual_extractions_json=prepared.paper_dir / "visual_extractions.json",
                    critiques_md=critiques_md,
                )
            }

    class FakeLLM:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def complete_json(self, *, response_model: type, **kwargs: object) -> object:
            assert response_model is RunComparisonReport
            return RunComparisonReport(
                paper_id="mock-paper",
                compared_runs=["baseline-a", "baseline-b", "fusion"],
                consensus=["All runs identify the paper."],
                disagreements=["Fusion adds a caveat."],
                unique_strengths={"fusion": ["Better caveats."]},
                blind_spots={"baseline-a": ["Less detail."]},
                criteria=[],
                recommended_run="fusion",
                recommendation_rationale="It has the best caveats.",
                fusion_assessment="Fusion improves synthesis without being assumed correct.",
                revision_suggestions=["Merge caveats into the final digest."],
            )

    result = run_fusion_comparison(
        source="mock.pdf",
        settings=Settings(openrouter_api_key="test", output_dir=tmp_path, verbose=False),
        output_dir=tmp_path,
        baseline_models=("model-a", "model-b"),
        fusion_outer_model="fusion-outer",
        fusion_analysis_models=("panel-a", "panel-b"),
        fusion_judge_model="fusion-judge",
        llm_factory=FakeLLM,  # type: ignore[arg-type]
        workflow_factory=FakeWorkflow,  # type: ignore[arg-type]
    )

    assert prepare_calls == 1
    assert variant_runs == ["baseline-a", "baseline-b", "fusion"]
    assert workflow_settings[0].output_dir == tmp_path
    assert workflow_settings[1].output_dir == tmp_path / "mock-paper" / "baseline-a"
    assert workflow_settings[2].output_dir == tmp_path / "mock-paper" / "baseline-b"
    assert workflow_settings[3].output_dir == tmp_path / "mock-paper" / "fusion"
    assert workflow_settings[1].fusion_enabled is False
    assert workflow_settings[2].fusion_enabled is False
    assert workflow_settings[3].fusion_enabled is True
    assert workflow_settings[3].model == "fusion-outer"
    assert workflow_settings[3].fusion_analysis_models == ("panel-a", "panel-b")
    assert result.comparison_json.exists()
    assert result.comparison_md.exists()

    payload = json.loads(result.comparison_json.read_text(encoding="utf-8"))
    assert [item["label"] for item in payload["snapshots"]] == [
        "baseline-a",
        "baseline-b",
        "fusion",
    ]
    assert payload["report"]["recommended_run"] == "fusion"
