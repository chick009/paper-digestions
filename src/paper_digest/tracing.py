"""Trace and artifact persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value


class TraceWriter:
    """Append-only JSONL trace writer for inspectable workflow runs."""

    def __init__(self, run_dir: Path, enabled: bool = True) -> None:
        self.run_dir = run_dir
        self.enabled = enabled
        self.trace_path = run_dir / "trace.jsonl"
        self.analysis_path = run_dir / "analysis.json"
        self._analysis: dict[str, Any] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        step: str,
        *,
        prompt_version: str | None = None,
        input_summary: Any = None,
        output: Any = None,
        model: str | None = None,
        token_usage: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        validation_errors: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "prompt_version": prompt_version,
            "input_summary": to_jsonable(input_summary),
            "output": to_jsonable(output),
            "model": model,
            "token_usage": token_usage or {},
            "latency_ms": latency_ms,
            "validation_errors": validation_errors or [],
            "notes": notes or [],
        }
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def set_analysis(self, key: str, value: Any) -> None:
        self._analysis[key] = to_jsonable(value)

    def write_analysis(self) -> None:
        self.analysis_path.write_text(
            json.dumps(self._analysis, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
