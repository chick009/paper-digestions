"""LangGraph workflow for paper digestion."""

from __future__ import annotations

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
    FindingsAnalysis,
    MethodologyAnalysis,
    PaperClassification,
    PaperDigestState,
)
from paper_digest.tracing import TraceWriter, to_jsonable
from paper_digest.vision_parse import run_vision_parsing, write_empty_visual_extractions


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
        graph = self._build_graph().compile()
        return graph.invoke({"input_ref": input_ref})

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
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("render_artifacts", self._render_artifacts)

        workflow.add_edge(START, "ingest")
        workflow.add_edge("ingest", "extract")
        workflow.add_conditional_edges(
            "extract",
            self._needs_vision,
            {"vision": "vision_parse", "skip": "classify"},
        )
        workflow.add_edge("vision_parse", "classify")
        workflow.add_edge("classify", "analyze_methodology")
        workflow.add_edge("analyze_methodology", "analyze_findings")
        workflow.add_edge("explain_concepts", "critique")
        workflow.add_edge("analyze_findings", "explain_concepts")
        workflow.add_edge("critique", "draft_report")
        workflow.add_edge("draft_report", "quality_check")
        workflow.add_edge("quality_check", "render_artifacts")
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
        trace.write_analysis()
        return {"source_info": source_info, "analysis": {"source_info": to_jsonable(source_info)}}

    def _extract(self, state: PaperDigestState) -> PaperDigestState:
        source_info = state["source_info"]
        trace = self._trace(state)
        cache_path = source_info.run_dir / "extracted_paper.json"
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
        write_extracted_text(extracted, source_info.run_dir / "extracted_text.md")
        (source_info.run_dir / "metadata.json").write_text(
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
        return {"extracted_paper": extracted, "analysis": analysis}

    def _needs_vision(self, state: PaperDigestState) -> str:
        if self.settings.vision_parse_enabled:
            return "vision"
        write_empty_visual_extractions(state["source_info"].run_dir)
        return "skip"

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
        return self._with_analysis(state, "report", report)

    def _quality_check(self, state: PaperDigestState) -> PaperDigestState:
        self._progress("quality", "Checking unsupported claims and weak evidence")
        quality_check = local_quality_check(state["report"], state["critiques"])
        trace = self._trace(state)
        trace.record("quality_check", output=quality_check)
        return self._with_analysis(state, "quality_check", quality_check)

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
            visual_extractions_json=run_dir / "visual_extractions.json",
            critiques_md=critiques_md,
        )
        trace = self._trace(state)
        trace.record("render_artifacts", output=artifacts)
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
        return (
            f"Classification:\n{classification.model_dump_json() if classification else '{}'}\n\n"
            f"Paper content:\n{self._paper_context(state)}"
        )

    def _structured_context(self, state: PaperDigestState) -> str:
        keys = ["classification", "methodology", "findings", "explanations", "critiques"]
        structured = {
            key: to_jsonable(state[key])
            for key in keys
            if key in state and state[key] is not None
        }
        paper_context = self._paper_context(state, max_chars=50000)
        return f"Structured analysis:\n{structured}\n\nPaper context:\n{paper_context}"

    def _progress(self, step: str, message: str) -> None:
        if self.settings.verbose:
            print(f"[paper-digest:{step}] {message}", flush=True)


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
