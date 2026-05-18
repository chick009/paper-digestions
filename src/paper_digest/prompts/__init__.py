"""Versioned prompt templates."""

from __future__ import annotations

from importlib.resources import files


def load_prompt(name: str) -> str:
    prompt_path = files("paper_digest.prompts").joinpath(name)
    return prompt_path.read_text(encoding="utf-8")
