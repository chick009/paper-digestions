# Paper Digest

Paper Digest is a CLI-first research reading assistant. It accepts a local PDF path or a direct PDF URL, classifies the source, analyzes the paper with multiple explanation and critique passes, and writes auditable artifacts under `output/<paper_slug>/`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export OPENROUTER_API_KEY="..."
paper-digest digest ./paper.pdf
```

The default output directory is `output/`. Each run writes Markdown, LaTeX, trace, structured analysis, and parser-plus-vision artifacts. If a LaTeX compiler such as `tectonic`, `latexmk`, or `pdflatex` is installed, the CLI also compiles `digest.pdf`.

## Streamlit UI

For a small browser UI (URL list, PDF uploads, run log, digest/critique previews, downloads):

```bash
pip install -e ".[ui]"
export OPENROUTER_API_KEY="..."
streamlit run src/paper_digest/streamlit_app.py
```

See **[docs/streamlit-ui.md](docs/streamlit-ui.md)** for how to open the app, tabs, and tips.

## Main Command

```bash
paper-digest digest <pdf-path-or-url> \
  --model openai/gpt-4o-mini \
  --vision-model openai/gpt-4o-mini \
  --output-dir output
```

## Evaluation Command

The repository includes a two-paper evaluation manifest under `test/expectations/papers.json`.

```bash
export OPENROUTER_API_KEY="..."
paper-digest evaluate test/expectations/papers.json \
  --model x-ai/grok-4.3 \
  --vision-model openai/gpt-5.4-mini \
  --output-dir test/evaluations \
  --vision-parse-mode all \
  --max-vision-pages 8 \
  --vision-parallelism 5 \
  --reuse-parse \
  --no-compile-pdf
```

This runs the digest pipeline for each PDF and then judges the generated summaries against the expected claims and questions in the manifest.
By default, parser outputs are reused from `extracted_paper.json` and `visual_extractions.json`
when they match the selected source/pages. Use `--force-parse` only when you want to refresh
those cached parser artifacts.

## Artifact Layout

```text
output/<paper_slug>/
  source.pdf
  extracted_text.md
  visual_extractions.json
  metadata.json
  trace.jsonl
  analysis.json
  critiques.md
  digest.md
  digest.tex
  digest.pdf
```

The trace stores explicit prompts, model-visible outputs, evidence references, uncertainties, and critique rationales. It does not claim to store hidden model chain-of-thought.
