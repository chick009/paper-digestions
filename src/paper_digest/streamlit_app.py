"""Streamlit UI for queuing PDFs/URLs and running the paper digest workflow."""

from __future__ import annotations

import contextlib
import json
import os
from io import StringIO
from pathlib import Path

import streamlit as st

from paper_digest.config import Settings
from paper_digest.graph import PaperDigestWorkflow
from paper_digest.ingest import slugify
from paper_digest.llm import LLMError, OpenRouterClient


def _parse_url_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _persist_upload(upload, output_dir: Path) -> Path:
    uploads_dir = (output_dir / "_uploads").resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)

    stem = slugify(Path(upload.name).stem) or "upload"
    dest = uploads_dir / f"{stem}.pdf"
    n = 0
    while dest.exists():
        n += 1
        dest = uploads_dir / f"{stem}-{n}.pdf"

    dest.write_bytes(upload.getbuffer())
    return dest


def _format_classification(data: dict) -> str:
    cls = data.get("classification")
    if not isinstance(cls, dict):
        return ""
    kind = cls.get("kind", "")
    confidence = cls.get("confidence", "")
    rationale = cls.get("rationale", "")
    if isinstance(rationale, str) and len(rationale) > 600:
        rationale = rationale[:600] + "…"
    return "\n".join(
        [
            f"**Kind:** `{kind}`",
            f"**Confidence:** {confidence}",
            "",
            str(rationale),
        ]
    )


def _render_truncated_markdown(text: str, *, show_full: bool, preview_chars: int) -> None:
    if len(text) <= preview_chars or show_full:
        st.markdown(text)
    else:
        st.markdown(text[:preview_chars] + "\n\n*…truncated…*")


def _list_digest_run_dirs(root: Path) -> list[Path]:
    """Subdirectories of `root` that contain a digest.md (evaluation-style layout)."""

    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "digest.md").is_file():
            out.append(p)
    return out


def _render_artifact_tabs(
    *,
    run_dir: Path,
    digest_md: Path,
    digest_tex: Path,
    critiques_md: Path,
    analysis_json: Path,
    digest_pdf: Path | None,
    input_ref: str | None,
    preview_chars: int,
    key_prefix: str,
) -> None:
    st.text(f"run_dir: {run_dir}")
    if input_ref:
        st.text(f"input_ref: {input_ref}")
    pdf_line = f"- digest.pdf: `{digest_pdf}`" if digest_pdf else None
    lines = [
        f"- digest.md: `{digest_md}`",
        f"- digest.tex: `{digest_tex}`",
        f"- critiques.md: `{critiques_md}`",
        f"- analysis.json: `{analysis_json}`",
        f"- trace.jsonl: `{run_dir / 'trace.jsonl'}`",
    ]
    if pdf_line:
        lines.append(pdf_line)
    st.markdown("\n".join(lines))

    tab_digest, tab_crit, tab_ana = st.tabs(["Digest", "Critiques", "Analysis"])

    with tab_digest:
        if digest_md.is_file():
            digest_text = digest_md.read_text(encoding="utf-8")
            show_full = st.checkbox(
                "Show full digest",
                key=f"{key_prefix}_digest_full",
                value=len(digest_text) <= preview_chars,
            )
            _render_truncated_markdown(
                digest_text, show_full=show_full, preview_chars=preview_chars
            )
            st.download_button(
                "Download digest.md",
                data=digest_text.encode("utf-8"),
                file_name="digest.md",
                mime="text/markdown",
                key=f"{key_prefix}_dl_digest",
            )
        else:
            st.info("digest.md not found.")

    with tab_crit:
        if critiques_md.is_file():
            crit_text = critiques_md.read_text(encoding="utf-8")
            show_full_c = st.checkbox(
                "Show full critiques",
                key=f"{key_prefix}_crit_full",
                value=len(crit_text) <= preview_chars,
            )
            _render_truncated_markdown(
                crit_text, show_full=show_full_c, preview_chars=preview_chars
            )
            st.download_button(
                "Download critiques.md",
                data=crit_text.encode("utf-8"),
                file_name="critiques.md",
                mime="text/markdown",
                key=f"{key_prefix}_dl_crit",
            )
        else:
            st.info("critiques.md not found.")

    with tab_ana:
        if analysis_json.is_file():
            try:
                analysis_data = json.loads(analysis_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                st.warning(f"Could not parse analysis.json: {exc}")
            else:
                st.markdown("### Classification")
                summary = _format_classification(analysis_data)
                if summary:
                    st.markdown(summary)
                else:
                    st.caption("No classification block.")

                st.markdown("### Quality check")
                qc = analysis_data.get("quality_check")
                if qc is not None:
                    st.json(qc)
                else:
                    st.caption("No quality_check block.")

                with st.expander("Raw analysis.json"):
                    st.json(analysis_data)
        else:
            st.info("analysis.json not found.")

    if digest_tex.is_file():
        st.download_button(
            "Download digest.tex",
            data=digest_tex.read_bytes(),
            file_name="digest.tex",
            mime="text/plain",
            key=f"{key_prefix}_dl_tex",
        )
    if digest_pdf is not None and digest_pdf.is_file():
        st.download_button(
            "Download digest.pdf",
            data=digest_pdf.read_bytes(),
            file_name="digest.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_pdf",
        )


def main() -> None:
    st.set_page_config(page_title="Paper Digest", layout="wide")
    st.title("Paper Digest")
    st.caption(
        "Run new digests or browse prior runs under your **Output directory** (default: output)."
    )

    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input(
            "OpenRouter API key (optional)",
            type="password",
            help="If empty, uses OPENROUTER_API_KEY from the environment or .env.",
            autocomplete="new-password",
        )
        if api_key.strip():
            os.environ["OPENROUTER_API_KEY"] = api_key.strip()

        model_raw = st.text_input("Text model override", placeholder="e.g. x-ai/grok-4.3")
        vision_raw = st.text_input(
            "Vision model override",
            placeholder="e.g. x-ai/grok-4.3",
        )
        multi_agent = st.checkbox("Multi-agent synthesis", value=False)
        agent_models_raw = st.text_input(
            "Multi-agent models",
            placeholder="moonshotai/kimi-k2.5, deepseek/deepseek-v4-pro, x-ai/grok-4.3",
        )
        agent_concurrency = st.number_input(
            "Multi-agent concurrency (0 = one worker per model)",
            min_value=0,
            value=0,
        )
        synthesizer_raw = st.text_input(
            "Synthesizer model",
            placeholder="x-ai/grok-4.3",
        )

        output_dir_str = st.text_input("Output directory", value="output")
        output_dir = Path(output_dir_str).expanduser().resolve()

        vision_parse = st.checkbox("Vision parsing", value=True)
        vision_mode = st.selectbox("Vision parse mode", ["auto", "all"], index=0)
        max_vision_pages = st.number_input("Max vision pages (0 = no limit)", min_value=0, value=8)
        vision_parallelism = st.number_input("Vision concurrency", min_value=1, value=5)
        force_parse = st.checkbox("Force parse (ignore cache)", value=False)
        reasoning = st.checkbox("Reasoning mode", value=True)
        trace = st.checkbox("Write trace.jsonl", value=True)
        compile_pdf = st.checkbox("Compile digest.pdf", value=True)

    tab_run, tab_samples = st.tabs(["Run digests", "Browse output samples"])

    preview_chars = 12_000

    with tab_run:
        url_text = st.text_area(
            "PDF URLs (one per line)",
            height=120,
            help="Must be direct http(s) links to a PDF (same as CLI).",
        )

        uploads = st.file_uploader(
            "PDF uploads",
            type=["pdf"],
            accept_multiple_files=True,
        )

        stop_on_error = st.checkbox("Stop queue on first error", value=False)

        run = st.button("Run digest queue", type="primary")

        if not run:
            st.info("Add URLs or PDFs above, then click **Run digest queue**.")
        else:
            urls = _parse_url_lines(url_text)
            jobs: list[tuple[str, str]] = []
            for url in urls:
                jobs.append((url, url))
            if uploads:
                for upload in uploads:
                    path = _persist_upload(upload, output_dir)
                    jobs.append((upload.name, str(path)))

            if not jobs:
                st.warning("Add at least one URL or PDF upload.")
            else:
                settings = Settings.from_env(
                    model=model_raw.strip() or None,
                    vision_model=vision_raw.strip() or None,
                    multi_agent_enabled=multi_agent,
                    agent_models=agent_models_raw.strip() or None,
                    agent_concurrency=int(agent_concurrency),
                    synthesizer_model=synthesizer_raw.strip() or None,
                    output_dir=output_dir,
                    vision_parse_enabled=vision_parse,
                    vision_parse_mode=vision_mode,
                    max_vision_pages=int(max_vision_pages),
                    vision_parse_concurrency=int(vision_parallelism),
                    force_parse=force_parse,
                    reasoning_enabled=reasoning,
                    verbose=True,
                )

                try:
                    llm = OpenRouterClient(settings)
                except LLMError as exc:
                    st.error(str(exc))
                else:
                    workflow = PaperDigestWorkflow(
                        settings=settings,
                        llm=llm,
                        save_trace=trace,
                        compile_pdf=compile_pdf,
                    )

                    log_lines: list[str] = []
                    log_display = st.empty()
                    results: list[dict] = []

                    for i, (label, ref) in enumerate(jobs, start=1):
                        log_lines.append(f"--- Job {i}/{len(jobs)}: {label} ---")
                        log_lines.append(f"input_ref={ref}")
                        log_display.code("\n".join(log_lines))

                        buf = StringIO()
                        try:
                            with contextlib.redirect_stdout(buf):
                                final_state = workflow.run(ref)
                            captured = buf.getvalue()
                            if captured:
                                log_lines.extend(line for line in captured.rstrip().splitlines())
                            log_lines.append("Finished OK.")
                            results.append(
                                {"ok": True, "label": label, "ref": ref, "state": final_state}
                            )
                        except Exception as exc:  # noqa: BLE001
                            log_lines.append(f"ERROR: {exc}")
                            results.append(
                                {"ok": False, "label": label, "ref": ref, "error": str(exc)}
                            )

                        log_display.code("\n".join(log_lines))

                        if stop_on_error and results and not results[-1]["ok"]:
                            break

                    st.subheader("Run log")
                    st.code("\n".join(log_lines))

                    st.subheader("Results")
                    for idx, r in enumerate(results):
                        if not r["ok"]:
                            with st.expander(f"✗ {r['label']}", expanded=False):
                                st.error(r.get("error", "Unknown error"))
                                st.text(f"input_ref={r['ref']}")
                            continue

                        state = r["state"]
                        artifacts = state["artifacts"]
                        with st.expander(f"✓ {r['label']}", expanded=True):
                            _render_artifact_tabs(
                                run_dir=Path(artifacts.run_dir),
                                digest_md=Path(artifacts.digest_md),
                                digest_tex=Path(artifacts.digest_tex),
                                critiques_md=Path(artifacts.critiques_md),
                                analysis_json=Path(artifacts.analysis_json),
                                digest_pdf=Path(artifacts.digest_pdf)
                                if artifacts.digest_pdf
                                else None,
                                input_ref=r["ref"],
                                preview_chars=preview_chars,
                                key_prefix=f"run_{idx}",
                            )

    with tab_samples:
        samples_root_str = st.text_input(
            "Digests root directory",
            value=str(output_dir),
            help=(
                "Matches **Output directory** by default. "
                "Each subfolder with digest.md is one paper."
            ),
        )
        samples_root = Path(samples_root_str).expanduser().resolve()
        run_dirs = _list_digest_run_dirs(samples_root)

        if not samples_root.is_dir():
            st.warning(f"Not a directory: `{samples_root}`")
        elif not run_dirs:
            st.info(
                f"No subfolders with **digest.md** under `{samples_root}`. "
                "Run a digest first, or point this path at a parent of per-paper folders."
            )
        else:
            labels = [p.name for p in run_dirs]
            choice = st.selectbox("Paper / run folder", options=labels, index=0)
            run_dir = samples_root / choice

            st.subheader(f"Outputs: `{choice}`")
            digest_pdf_path = run_dir / "digest.pdf"
            digest_pdf = digest_pdf_path if digest_pdf_path.is_file() else None
            input_ref: str | None = None
            analysis_path = run_dir / "analysis.json"
            if analysis_path.is_file():
                try:
                    data = json.loads(analysis_path.read_text(encoding="utf-8"))
                    si = data.get("source_info")
                    if isinstance(si, dict):
                        ref = si.get("input_ref")
                        input_ref = ref if isinstance(ref, str) else None
                except json.JSONDecodeError:
                    pass

            _render_artifact_tabs(
                run_dir=run_dir,
                digest_md=run_dir / "digest.md",
                digest_tex=run_dir / "digest.tex",
                critiques_md=run_dir / "critiques.md",
                analysis_json=analysis_path,
                digest_pdf=digest_pdf,
                input_ref=input_ref,
                preview_chars=preview_chars,
                key_prefix=f"samples_{slugify(choice)}",
            )


if __name__ == "__main__":
    main()
