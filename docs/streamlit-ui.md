# Streamlit UI

Browser UI to queue PDFs or direct PDF URLs, run the digest pipeline, and browse finished runs under your output folder.

## Open the app

From the **repository root** (the folder that contains `pyproject.toml` and `src/`):

1. **Activate your environment** (recommended):

   ```bash
   source .venv/bin/activate
   ```

2. **Install the UI extra** (once per environment, or after dependency changes):

   ```bash
   pip install -e ".[ui]"
   ```

3. **Set your API key** (or paste it in the app sidebar instead):

   ```bash
   export OPENROUTER_API_KEY="your-key"
   ```

4. **Start Streamlit**:

   ```bash
   streamlit run src/paper_digest/streamlit_app.py
   ```

5. **Open the app** in a browser at the URL Streamlit prints, usually:

   **http://localhost:8501**

Stop the server with **Ctrl+C** in that terminal.

## Tabs

- **Run digests** — Enter PDF URLs (one per line) and/or upload PDFs, set **Output directory** in the sidebar (default `output`), then click **Run digest queue**. Artifacts are written under `output/<paper_slug>/` (or whatever output directory you chose).

- **Browse output samples** — Pick a **Digests root directory** (defaults to the same path as **Output directory**). Each immediate subfolder that contains `digest.md` appears as one paper; use the dropdown to open digest, critiques, and analysis.

## Tips

- If your virtualenv has no `pip`, bootstrap it: `python -m ensurepip --upgrade`, then `pip install -e ".[ui]"`.
- Optional: `pip install watchdog` for faster auto-reload during development.
- Streamlit reruns the whole page on each interaction; the **Run digest queue** button only triggers a run on the click that immediately follows. After you change another widget, use **Browse output samples** or run again to see outputs.
