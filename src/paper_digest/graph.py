"""LangGraph workflow for paper digestion."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from paper_digest.config import Settings
from paper_digest.extract import (
    extract_pdf,
    read_cached_extracted_paper,
    write_extracted_paper,
    write_extracted_text,
)
from paper_digest.ingest import resolve_source
from paper_digest.llm import OpenRouterClient
from paper_digest.prompts import load_prompt
from paper_digest.quality import local_quality_check
from paper_digest.render_latex import compile_latex, render_digest_latex
from paper_digest.render_markdown import render_critiques_markdown, render_digest_markdown
from paper_digest.schema import (
    ArtifactPaths,
    BlogSynthesis,
    ConceptExplanationSet,
    CritiqueSet,
    DigestReport,
    ExtractedPaper,
    FindingsAnalysis,
    MethodologyAnalysis,
    ModelPipelineDraft,
    PaperClassification,
    PaperDigestState,
    SourceInfo,
)
from paper_digest.tracing import TraceWriter, to_jsonable
from paper_digest.vision_parse import run_vision_parsing, write_empty_visual_extractions


@dataclass(frozen=True)
class PreparedPaper:
    """Shared PDF text extraction and vision parse artifacts for one paper."""

    paper_dir: Path
    input_ref: str
    source_info: SourceInfo
    extracted_paper: ExtractedPaper


class PaperDigestWorkflow:
    """Build and run the LangGraph-backed digest pipeline."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm: OpenRouterClient,
        save_trace: bool = True,
        compile_pdf: bool = True,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.save_trace = save_trace
        self.compile_pdf = compile_pdf

    def run(self, input_ref: str) -> PaperDigestState:
        self._progress("workflow", f"Starting digest pipeline for: {input_ref}")
        graph = self._build_graph().compile()
        state = graph.invoke({"input_ref": input_ref})
        self._progress(
            "workflow",
            "Digest pipeline complete; running post-pipeline quality evaluation",
        )
        result = self._run_quality_check(state)
        source_info = result["source_info"]
        self._progress("workflow", f"Finished digest run: {source_info.run_dir}")
        return result

    def prepare_paper(self, input_ref: str) -> PreparedPaper:
        """Ingest, extract, and vision-parse a paper once into a shared directory."""

        self._progress("prepare", f"Preparing shared parse artifacts for: {input_ref}")
        state: PaperDigestState = {"input_ref": input_ref}
        state.update(self._ingest(state))
        state.update(self._extract(state))
        if self.settings.vision_parse_enabled:
            state.update(self._vision_parse(state))
        else:
            write_empty_visual_extractions(state["source_info"].parse_root())
        paper_dir = state["source_info"].run_dir
        self._progress("prepare", f"Shared parse artifacts ready at {paper_dir}")
        return PreparedPaper(
            paper_dir=paper_dir,
            input_ref=input_ref,
            source_info=state["source_info"],
            extracted_paper=state["extracted_paper"],
        )

    def run_variant(self, prepared: PreparedPaper, *, variant_dir: Path) -> PaperDigestState:
        """Run the analysis pipeline using pre-parsed paper artifacts."""

        variant_dir.mkdir(parents=True, exist_ok=True)
        self._progress(
            "workflow",
            (
                f"Starting variant pipeline {variant_dir.name} "
                f"using shared parse at {prepared.paper_dir}"
            ),
        )
        source_info = prepared.source_info.model_copy(
            update={"run_dir": variant_dir, "parse_dir": prepared.paper_dir},
        )
        run_config = self._run_config()
        trace = TraceWriter(variant_dir, enabled=self.save_trace)
        trace.set_analysis("source_info", source_info)
        trace.set_analysis("run_config", run_config)
        trace.write_analysis()

        initial_state: PaperDigestState = {
            "input_ref": prepared.input_ref,
            "source_info": source_info,
            "extracted_paper": prepared.extracted_paper,
            "analysis": {
                "source_info": to_jsonable(source_info),
                "run_config": run_config,
                "metadata": to_jsonable(prepared.extracted_paper.metadata),
                "weak_pages": prepared.extracted_paper.weak_pages,
                "visual_extractions": to_jsonable(prepared.extracted_paper.visual_extractions),
            },
        }
        graph = self._build_analysis_graph().compile()
        state = graph.invoke(initial_state)
        self._progress(
            "workflow",
            "Variant pipeline complete; running post-pipeline quality evaluation",
        )
        result = self._run_quality_check(state)
        self._progress("workflow", f"Finished variant run: {variant_dir}")
        return result

    def _build_analysis_graph(self) -> StateGraph:
        workflow = StateGraph(PaperDigestState)
        workflow.add_node("classify", self._classify)
        workflow.add_node("analyze_methodology", self._analyze_methodology)
        workflow.add_node("analyze_findings", self._analyze_findings)
        workflow.add_node("explain_concepts", self._explain_concepts)
        workflow.add_node("critique", self._critique)
        workflow.add_node("draft_report", self._draft_report)
        workflow.add_node("multi_agent_synthesize", self._multi_agent_synthesize)
        workflow.add_node("render_artifacts", self._render_artifacts)

        workflow.add_conditional_edges(
            START,
            self._route_analysis_entry,
            {"single": "classify", "multi_agent": "multi_agent_synthesize"},
        )
        workflow.add_edge("classify", "analyze_methodology")
        workflow.add_edge("analyze_methodology", "analyze_findings")
        workflow.add_edge("explain_concepts", "critique")
        workflow.add_edge("analyze_findings", "explain_concepts")
        workflow.add_edge("critique", "draft_report")
        workflow.add_edge("draft_report", "render_artifacts")
        workflow.add_edge("multi_agent_synthesize", "render_artifacts")
        workflow.add_edge("render_artifacts", END)
        return workflow

    def _route_analysis_entry(self, _: PaperDigestState) -> str:
        route = "multi_agent" if self.settings.multi_agent_enabled else "single"
        self._progress("route", f"Routing to {route} analysis path")
        return route

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(PaperDigestState)
        workflow.add_node("ingest", self._ingest)
        workflow.add_node("extract", self._extract)
        workflow.add_node("vision_parse", self._vision_parse)
        workflow.add_node("classify", self._classify)
        workflow.add_node("analyze_methodology", self._analyze_methodology)
        workflow.add_node("analyze_findings", self._analyze_findings)
        workflow.add_node("explain_concepts", self._explain_concepts)
        workflow.add_node("critique", self._critique)
        workflow.add_node("draft_report", self._draft_report)
        workflow.add_node("multi_agent_synthesize", self._multi_agent_synthesize)
        workflow.add_node("render_artifacts", self._render_artifacts)

        workflow.add_edge(START, "ingest")
        workflow.add_edge("ingest", "extract")
        workflow.add_conditional_edges(
            "extract",
            self._route_after_extract,
            {
                "vision": "vision_parse",
                "single": "classify",
                "multi_agent": "multi_agent_synthesize",
            },
        )
        workflow.add_conditional_edges(
            "vision_parse",
            self._route_after_parse,
            {"single": "classify", "multi_agent": "multi_agent_synthesize"},
        )
        workflow.add_edge("classify", "analyze_methodology")
        workflow.add_edge("analyze_methodology", "analyze_findings")
        workflow.add_edge("explain_concepts", "critique")
        workflow.add_edge("analyze_findings", "explain_concepts")
        workflow.add_edge("critique", "draft_report")
        workflow.add_edge("draft_report", "render_artifacts")
        workflow.add_edge("multi_agent_synthesize", "render_artifacts")
        workflow.add_edge("render_artifacts", END)
        return workflow

    def _trace(self, state: PaperDigestState) -> TraceWriter:
        source_info = state["source_info"]
        return TraceWriter(source_info.run_dir, enabled=self.save_trace)

    def _ingest(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("ingest", f"Resolving source: {state['input_ref']}")
        source_info = resolve_source(state["input_ref"], self.settings.output_dir)
        trace = TraceWriter(source_info.run_dir, enabled=self.save_trace)
        trace.record(
            "ingest",
            input_summary={"input_ref": state["input_ref"]},
            output=source_info,
            notes=["Resolved PDF source into the run directory."],
        )
        trace.set_analysis("source_info", source_info)
        run_config = self._run_config()
        trace.set_analysis("run_config", run_config)
        trace.write_analysis()
        self._progress("ingest", f"Resolved run directory: {source_info.run_dir}")
        return {
            "source_info": source_info,
            "analysis": {"source_info": to_jsonable(source_info), "run_config": run_config},
        }

    def _extract(self, state: PaperDigestState) -> PaperDigestState:
        source_info = state["source_info"]
        parse_dir = source_info.parse_root()
        trace = self._trace(state)
        cache_path = parse_dir / "extracted_paper.json"
        extracted = None
        if not self.settings.force_parse:
            extracted = read_cached_extracted_paper(cache_path, source_info.sha256)
        if extracted is not None:
            self._progress("extract", f"Reusing cached PDF parse: {cache_path}")
            trace.record(
                "extract_cache_hit",
                input_summary={"source_pdf": source_info.source_pdf, "cache_path": cache_path},
                output={
                    "page_count": extracted.metadata.page_count,
                    "weak_pages": extracted.weak_pages,
                },
                notes=["Reused cached parser output because source hash matched."],
            )
        else:
            self._progress("extract", f"Parsing PDF text and metadata: {source_info.source_pdf}")
            extracted = extract_pdf(source_info)
            write_extracted_paper(extracted, cache_path)
        write_extracted_text(extracted, parse_dir / "extracted_text.md")
        (parse_dir / "metadata.json").write_text(
            extracted.metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        trace.record(
            "extract",
            input_summary={"source_pdf": source_info.source_pdf},
            output={
                "page_count": extracted.metadata.page_count,
                "weak_pages": extracted.weak_pages,
            },
            notes=["Prepared selectable PDF text and page-quality signals."],
        )
        analysis = dict(state.get("analysis", {}))
        analysis["metadata"] = to_jsonable(extracted.metadata)
        analysis["weak_pages"] = extracted.weak_pages
        trace.set_analysis("metadata", extracted.metadata)
        trace.set_analysis("weak_pages", extracted.weak_pages)
        trace.write_analysis()
        self._progress(
            "extract",
            (
                f"Prepared {extracted.metadata.page_count} page(s); "
                f"weak pages={extracted.weak_pages or 'none'}"
            ),
        )
        return {"extracted_paper": extracted, "analysis": analysis}

    def _route_after_extract(self, state: PaperDigestState) -> str:
        if self.settings.vision_parse_enabled:
            self._progress("route", "Vision parse enabled → vision_parse")
            return "vision"
        write_empty_visual_extractions(state["source_info"].parse_root())
        self._progress("route", "Vision parse disabled → skipping to analysis path")
        return self._route_after_parse(state)

    def _route_after_parse(self, _: PaperDigestState) -> str:
        route = "multi_agent" if self.settings.multi_agent_enabled else "single"
        self._progress("route", f"Routing to {route} analysis path")
        return route

    def _vision_parse(self, state: PaperDigestState) -> PaperDigestState:
        trace = self._trace(state)
        source_info = state["source_info"]
        extracted = state["extracted_paper"]
        self._progress(
            "vision-parse",
            (
                f"Mode={self.settings.vision_parse_mode}, "
                f"max_pages={self.settings.max_vision_pages}, "
                f"parallelism={self.settings.vision_parse_concurrency}"
            ),
        )
        visual_extractions = run_vision_parsing(
            source_info=source_info,
            extracted=extracted,
            llm=self.llm,
            trace=trace,
            max_pages=self.settings.max_vision_pages,
            mode=self.settings.vision_parse_mode,  # type: ignore[arg-type]
            concurrency=self.settings.vision_parse_concurrency,
        )
        extracted.visual_extractions = visual_extractions
        analysis = dict(state.get("analysis", {}))
        analysis["visual_extractions"] = to_jsonable(visual_extractions)
        trace.set_analysis("visual_extractions", visual_extractions)
        trace.write_analysis()
        self._progress(
            "vision-parse",
            f"Completed vision parse for {len(visual_extractions)} page(s)",
        )
        return {"extracted_paper": extracted, "analysis": analysis}

    def _classify(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("classify", "Classifying paper type")
        trace = self._trace(state)
        classification = self.llm.complete_json(
            step="classify",
            system_prompt=load_prompt("classify.md"),
            user_prompt=self._paper_context(state),
            response_model=PaperClassification,
            trace=trace,
            prompt_version="classify-v1",
        )
        self._progress("classify", f"Classified as {classification.kind.value}")
        return self._with_analysis(state, "classification", classification)

    def _analyze_methodology(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("methodology", "Explaining core methodology")
        trace = self._trace(state)
        methodology = self.llm.complete_json(
            step="analyze_methodology",
            system_prompt=load_prompt("methodology.md"),
            user_prompt=self._analysis_context(state),
            response_model=MethodologyAnalysis,
            trace=trace,
            prompt_version="methodology-v1",
        )
        self._progress("methodology", "Methodology analysis complete")
        return self._with_analysis(state, "methodology", methodology)

    def _analyze_findings(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("findings", "Extracting findings and results")
        trace = self._trace(state)
        findings = self.llm.complete_json(
            step="analyze_findings",
            system_prompt=load_prompt("findings.md"),
            user_prompt=self._analysis_context(state),
            response_model=FindingsAnalysis,
            trace=trace,
            prompt_version="findings-v1",
        )
        self._progress(
            "findings",
            f"Findings analysis complete ({len(findings.important_findings)} finding(s))",
        )
        return self._with_analysis(state, "findings", findings)

    def _explain_concepts(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("explanations", "Generating concept and formula explanations")
        trace = self._trace(state)
        explanations = self.llm.complete_json(
            step="explain_concepts",
            system_prompt=load_prompt("explanations.md"),
            user_prompt=self._analysis_context(state),
            response_model=ConceptExplanationSet,
            trace=trace,
            prompt_version="explanations-v1",
        )
        self._progress(
            "explanations",
            f"Concept explanations complete ({len(explanations.explanations)} concept(s))",
        )
        return self._with_analysis(state, "explanations", explanations)

    def _critique(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("critique", "Running multiple critique lenses")
        trace = self._trace(state)
        critiques = self.llm.complete_json(
            step="critique",
            system_prompt=load_prompt("critique.md"),
            user_prompt=self._structured_context(state),
            response_model=CritiqueSet,
            trace=trace,
            prompt_version="critique-v1",
        )
        self._progress(
            "critique",
            f"Critique complete ({len(critiques.critiques)} lens(es); "
            f"sota={critiques.final_sota_stance})",
        )
        return self._with_analysis(state, "critiques", critiques)

    def _draft_report(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("blog", "Drafting final blog-style report")
        trace = self._trace(state)
        synthesis = self.llm.complete_json(
            step="draft_blog_report",
            system_prompt=load_prompt("blog.md"),
            user_prompt=self._structured_context(state),
            response_model=BlogSynthesis,
            trace=trace,
            prompt_version="blog-v1",
        )
        extracted = state["extracted_paper"]
        report = DigestReport(
            title=synthesis.title,
            subtitle=synthesis.subtitle,
            article_markdown=synthesis.article_markdown,
            metadata=extracted.metadata,
            classification=state["classification"],
            executive_summary=synthesis.executive_summary,
            methodology=state["methodology"],
            findings=state["findings"],
            explanations=state["explanations"].explanations,
            critiques=state["critiques"].critiques,
            final_assessment=synthesis.final_assessment,
            practical_takeaways=synthesis.practical_takeaways,
            open_questions=synthesis.open_questions,
            references=synthesis.references,
        )
        self._progress("blog", f"Blog draft complete: {report.title!r}")
        return self._with_analysis(state, "report", report)

    def _multi_agent_synthesize(self, state: PaperDigestState) -> PaperDigestState:
        models = self.settings.agent_models
        self._progress(
            "multi-agent",
            (
                f"Running {len(models)} full analysis pipelines in parallel; "
                f"synthesizer={self.settings.synthesizer_model}"
            ),
        )
        worker_count = _agent_worker_count(len(models), self.settings.agent_concurrency)
        trace = self._trace(state)
        trace.record(
            "multi_agent_config",
            output={
                "agent_models": list(models),
                "agent_concurrency": worker_count,
                "synthesizer_model": self.settings.synthesizer_model,
            },
            notes=["Each agent model runs the existing analysis and blog pipeline concurrently."],
        )

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="paper-agent") as pool:
            drafts = list(
                pool.map(
                    lambda model: self._run_model_pipeline(state, model=model, trace=trace),
                    models,
                )
            )
        self._progress("multi-agent", "All agent pipelines complete; running synthesizer")
        synthesis = self.llm.complete_json(
            step="multi_agent_synthesizer",
            system_prompt=load_prompt("multi_agent_synthesizer.md"),
            user_prompt=self._multi_agent_context(state, drafts),
            response_model=BlogSynthesis,
            trace=trace,
            model=self.settings.synthesizer_model,
            prompt_version="multi-agent-synthesizer-v1",
        )
        canonical = self._canonical_draft(drafts)
        extracted = state["extracted_paper"]
        report = DigestReport(
            title=synthesis.title,
            subtitle=synthesis.subtitle,
            article_markdown=synthesis.article_markdown,
            metadata=extracted.metadata,
            classification=canonical.classification,
            executive_summary=synthesis.executive_summary,
            methodology=canonical.methodology,
            findings=canonical.findings,
            explanations=canonical.explanations.explanations,
            critiques=canonical.critiques.critiques,
            final_assessment=synthesis.final_assessment,
            practical_takeaways=synthesis.practical_takeaways,
            open_questions=synthesis.open_questions,
            references=synthesis.references,
        )

        analysis = dict(state.get("analysis", {}))
        analysis.update(
            {
                "multi_agent": {
                    "agent_models": list(models),
                    "agent_concurrency": worker_count,
                    "synthesizer_model": self.settings.synthesizer_model,
                    "canonical_model": canonical.model,
                },
                "model_pipeline_drafts": to_jsonable(drafts),
                "classification": to_jsonable(canonical.classification),
                "methodology": to_jsonable(canonical.methodology),
                "findings": to_jsonable(canonical.findings),
                "explanations": to_jsonable(canonical.explanations),
                "critiques": to_jsonable(canonical.critiques),
                "report": to_jsonable(report),
            }
        )
        for key, value in analysis.items():
            trace.set_analysis(key, value)
        trace.write_analysis()
        self._progress("multi-agent", f"Multi-agent synthesis complete: {report.title!r}")
        return {
            "classification": canonical.classification,
            "methodology": canonical.methodology,
            "findings": canonical.findings,
            "explanations": canonical.explanations,
            "critiques": canonical.critiques,
            "model_pipeline_drafts": drafts,
            "report": report,
            "analysis": analysis,
        }

    def _run_model_pipeline(
        self,
        state: PaperDigestState,
        *,
        model: str,
        trace: TraceWriter,
    ) -> ModelPipelineDraft:
        label = _model_step_label(model)
        self._progress("multi-agent", f"{model}: classify")
        classification = self.llm.complete_json(
            step=f"agent_{label}_classify",
            system_prompt=load_prompt("classify.md"),
            user_prompt=self._paper_context(state),
            response_model=PaperClassification,
            trace=trace,
            model=model,
            prompt_version="classify-v1",
        )

        self._progress("multi-agent", f"{model}: methodology")
        methodology = self.llm.complete_json(
            step=f"agent_{label}_methodology",
            system_prompt=load_prompt("methodology.md"),
            user_prompt=self._analysis_context_for(classification=classification, state=state),
            response_model=MethodologyAnalysis,
            trace=trace,
            model=model,
            prompt_version="methodology-v1",
        )

        self._progress("multi-agent", f"{model}: findings")
        findings = self.llm.complete_json(
            step=f"agent_{label}_findings",
            system_prompt=load_prompt("findings.md"),
            user_prompt=self._analysis_context_for(classification=classification, state=state),
            response_model=FindingsAnalysis,
            trace=trace,
            model=model,
            prompt_version="findings-v1",
        )

        self._progress("multi-agent", f"{model}: explanations")
        explanations = self.llm.complete_json(
            step=f"agent_{label}_explanations",
            system_prompt=load_prompt("explanations.md"),
            user_prompt=self._analysis_context_for(classification=classification, state=state),
            response_model=ConceptExplanationSet,
            trace=trace,
            model=model,
            prompt_version="explanations-v1",
        )

        self._progress("multi-agent", f"{model}: critique")
        critiques = self.llm.complete_json(
            step=f"agent_{label}_critique",
            system_prompt=load_prompt("critique.md"),
            user_prompt=self._structured_context_for(
                state,
                classification=classification,
                methodology=methodology,
                findings=findings,
                explanations=explanations,
            ),
            response_model=CritiqueSet,
            trace=trace,
            model=model,
            prompt_version="critique-v1",
        )

        self._progress("multi-agent", f"{model}: blog draft")
        blog = self.llm.complete_json(
            step=f"agent_{label}_blog",
            system_prompt=load_prompt("blog.md"),
            user_prompt=self._structured_context_for(
                state,
                classification=classification,
                methodology=methodology,
                findings=findings,
                explanations=explanations,
                critiques=critiques,
            ),
            response_model=BlogSynthesis,
            trace=trace,
            model=model,
            prompt_version="blog-v1",
        )
        return ModelPipelineDraft(
            model=model,
            classification=classification,
            methodology=methodology,
            findings=findings,
            explanations=explanations,
            critiques=critiques,
            blog=blog,
        )

    def _run_quality_check(self, state: PaperDigestState) -> PaperDigestState:
        """Post-pipeline evaluation: deterministic checks on the generated digest."""
        self._progress("quality", "Checking unsupported claims and weak evidence")
        quality_check = local_quality_check(state["report"], state["critiques"])
        trace = self._trace(state)
        trace.record("quality_check", output=quality_check)
        status = "passed" if quality_check.passed else "failed"
        note_count = len(quality_check.revision_notes)
        self._progress("quality", f"Quality evaluation {status} ({note_count} note(s))")
        updated = self._with_analysis(state, "quality_check", quality_check)
        return {**state, **updated}

    def _render_artifacts(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("render", "Writing Markdown, LaTeX, trace, and judge-ready artifacts")
        source_info = state["source_info"]
        run_dir = source_info.run_dir
        digest_md = run_dir / "digest.md"
        digest_tex = run_dir / "digest.tex"
        critiques_md = run_dir / "critiques.md"

        digest_md.write_text(render_digest_markdown(state["report"]), encoding="utf-8")
        digest_tex.write_text(render_digest_latex(state["report"]), encoding="utf-8")
        critiques_md.write_text(render_critiques_markdown(state["critiques"]), encoding="utf-8")
        digest_pdf = compile_latex(digest_tex) if self.compile_pdf else None

        artifacts = ArtifactPaths(
            run_dir=run_dir,
            digest_md=digest_md,
            digest_tex=digest_tex,
            digest_pdf=digest_pdf,
            trace_jsonl=run_dir / "trace.jsonl",
            analysis_json=run_dir / "analysis.json",
            visual_extractions_json=source_info.parse_root() / "visual_extractions.json",
            critiques_md=critiques_md,
        )
        trace = self._trace(state)
        trace.record("render_artifacts", output=artifacts)
        self._progress(
            "render",
            (
                f"Wrote digest.md, digest.tex, critiques.md, analysis.json "
                f"under {run_dir}"
            ),
        )
        return self._with_analysis(state, "artifacts", artifacts)

    def _with_analysis(self, state: PaperDigestState, key: str, value: Any) -> PaperDigestState:
        analysis = dict(state.get("analysis", {}))
        analysis[key] = to_jsonable(value)
        trace = self._trace(state)
        trace.set_analysis(key, value)
        for existing_key, existing_value in analysis.items():
            if existing_key != key:
                trace.set_analysis(existing_key, existing_value)
        trace.write_analysis()
        return {key: value, "analysis": analysis}

    def _paper_context(self, state: PaperDigestState, max_chars: int = 80000) -> str:
        extracted = state["extracted_paper"]
        visual = "\n\n".join(
            (
                f"[Visual page {item.page_number}]\n"
                f"{item.extracted_text}\n{item.visual_summary}\n"
                f"Equations: {item.equations}\nTables: {item.tables}\nFigures: {item.figures}"
            )
            for item in extracted.visual_extractions
        )
        metadata = f"Metadata:\n{extracted.metadata.model_dump_json()}"
        content = f"{metadata}\n\nText:\n{extracted.full_text}\n\nVisual:\n{visual}"
        if len(content) <= max_chars:
            return content
        visual_budget = min(len(visual), max_chars // 5)
        visual_excerpt = _preserve_front_and_back(visual, visual_budget) if visual else ""
        fixed = f"{metadata}\n\nText excerpts sampled across all pages:\n"
        text_budget = max_chars - len(fixed) - len("\n\nVisual excerpts:\n") - len(visual_excerpt)
        sampled_text = _sample_pages_text(extracted.pages, max(0, text_budget))
        return f"{fixed}{sampled_text}\n\nVisual excerpts:\n{visual_excerpt}"

    def _analysis_context(self, state: PaperDigestState) -> str:
        classification = state.get("classification")
        return self._analysis_context_for(classification=classification, state=state)

    def _analysis_context_for(
        self,
        *,
        classification: PaperClassification | None,
        state: PaperDigestState,
    ) -> str:
        return (
            f"Classification:\n{classification.model_dump_json() if classification else '{}'}\n\n"
            f"Paper content:\n{self._paper_context(state)}"
        )

    def _structured_context(self, state: PaperDigestState) -> str:
        structured = self._structured_parts(
            classification=state.get("classification"),
            methodology=state.get("methodology"),
            findings=state.get("findings"),
            explanations=state.get("explanations"),
            critiques=state.get("critiques"),
        )
        paper_context = self._paper_context(state, max_chars=50000)
        return f"Structured analysis:\n{structured}\n\nPaper context:\n{paper_context}"

    def _structured_context_for(
        self,
        state: PaperDigestState,
        *,
        classification: PaperClassification | None = None,
        methodology: MethodologyAnalysis | None = None,
        findings: FindingsAnalysis | None = None,
        explanations: ConceptExplanationSet | None = None,
        critiques: CritiqueSet | None = None,
    ) -> str:
        structured = self._structured_parts(
            classification=classification,
            methodology=methodology,
            findings=findings,
            explanations=explanations,
            critiques=critiques,
        )
        paper_context = self._paper_context(state, max_chars=50000)
        return f"Structured analysis:\n{structured}\n\nPaper context:\n{paper_context}"

    def _structured_parts(
        self,
        *,
        classification: PaperClassification | None = None,
        methodology: MethodologyAnalysis | None = None,
        findings: FindingsAnalysis | None = None,
        explanations: ConceptExplanationSet | None = None,
        critiques: CritiqueSet | None = None,
    ) -> dict[str, Any]:
        values = {
            "classification": classification,
            "methodology": methodology,
            "findings": findings,
            "explanations": explanations,
            "critiques": critiques,
        }
        return {key: to_jsonable(value) for key, value in values.items() if value is not None}

    def _multi_agent_context(
        self,
        state: PaperDigestState,
        drafts: list[ModelPipelineDraft],
        max_chars: int = 120000,
    ) -> str:
        payload = {
            "synthesizer_model": self.settings.synthesizer_model,
            "agent_models": [draft.model for draft in drafts],
            "candidate_pipelines": to_jsonable(drafts),
        }
        candidate_json = json.dumps(payload, ensure_ascii=False, indent=2)
        fixed = "Multi-agent candidate outputs:\n"
        paper_header = "\n\nPaper context for checking evidence:\n"
        paper_budget = max_chars - len(fixed) - len(candidate_json) - len(paper_header)
        paper_context = (
            self._paper_context(state, max_chars=min(50000, paper_budget))
            if paper_budget > 0
            else ""
        )
        content = f"{fixed}{candidate_json}{paper_header}{paper_context}"
        if len(content) <= max_chars:
            return content
        candidate_budget = max_chars - len(fixed) - len(paper_header) - len(paper_context)
        candidate_excerpt = (
            _preserve_front_and_back(candidate_json, candidate_budget)
            if candidate_budget > 0
            else ""
        )
        return f"{fixed}{candidate_excerpt}{paper_header}{paper_context}"

    def _canonical_draft(self, drafts: list[ModelPipelineDraft]) -> ModelPipelineDraft:
        for draft in drafts:
            if draft.model == self.settings.synthesizer_model:
                return draft
        for draft in drafts:
            if draft.model == self.settings.model:
                return draft
        return drafts[-1]

    def _progress(self, step: str, message: str) -> None:
        if self.settings.verbose:
            print(f"[paper-digest:{step}] {message}", flush=True)

    def _run_config(self) -> dict[str, Any]:
        if self.settings.multi_agent_enabled:
            mode = "multi_agent_fusion" if self.settings.fusion_enabled else "multi_agent"
        else:
            mode = "fusion" if self.settings.fusion_enabled else "single"
        config: dict[str, Any] = {
            "mode": mode,
            "model": self.settings.model,
            "vision_model": self.settings.vision_model,
            "vision_parse_enabled": self.settings.vision_parse_enabled,
            "vision_parse_mode": self.settings.vision_parse_mode,
            "reasoning_enabled": self.settings.reasoning_enabled,
            "reasoning_effort": self.settings.reasoning_effort,
            "created_at": datetime.now(UTC).isoformat(),
        }
        if self.settings.multi_agent_enabled:
            config["multi_agent"] = {
                "agent_models": list(self.settings.agent_models),
                "agent_concurrency": _agent_worker_count(
                    len(self.settings.agent_models),
                    self.settings.agent_concurrency,
                ),
                "synthesizer_model": self.settings.synthesizer_model,
            }
        if self.settings.fusion_enabled:
            config["fusion"] = {
                "analysis_models": list(self.settings.fusion_analysis_models),
                "judge_model": self.settings.fusion_judge_model,
                "max_tool_calls": self.settings.fusion_max_tool_calls,
                "temperature": self.settings.fusion_temperature,
                "force": self.settings.fusion_force,
            }
        return config


def _preserve_front_and_back(content: str, max_chars: int) -> str:
    """Keep the abstract/main text plus appendices where important analyses often live."""

    if max_chars <= 0 or len(content) <= max_chars:
        return content
    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars
    head = content[:head_chars].rstrip()
    tail = content[-tail_chars:].lstrip()
    return (
        f"{head}\n\n"
        "[MIDDLE OF PAPER TRUNCATED; APPENDIX/TAIL CONTEXT PRESERVED BELOW]\n\n"
        f"{tail}"
    )


def _sample_pages_text(pages: list[Any], max_chars: int) -> str:
    """Sample every page so middle/appendix evidence is not silently dropped."""

    if max_chars <= 0 or not pages:
        return ""
    per_page = max(400, max_chars // len(pages))
    excerpts = []
    for page in pages:
        text = page.text.strip()
        if len(text) > per_page:
            half = max(150, per_page // 2)
            text = (
                f"{text[:half].rstrip()}\n"
                "[PAGE MIDDLE TRUNCATED]\n"
                f"{text[-half:].lstrip()}"
            )
        excerpts.append(f"[Page {page.page_number}]\n{text}")
    sampled = "\n\n".join(excerpts)
    if len(sampled) <= max_chars:
        return sampled
    return sampled[:max_chars].rstrip() + "\n\n[TRUNCATED AFTER PAGE-SAMPLED CONTEXT]"


def _model_step_label(model: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()
    return label or "model"


def _agent_worker_count(model_count: int, configured_concurrency: int) -> int:
    if model_count <= 0:
        return 1
    if configured_concurrency <= 0:
        return model_count
    return max(1, min(configured_concurrency, model_count))
