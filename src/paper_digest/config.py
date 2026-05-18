"""Runtime configuration for Paper Digest."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without adding another dependency."""

    dotenv_path = path or _find_dotenv()
    if dotenv_path is None or not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_dotenv() -> Path | None:
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class Settings:
    """Configuration loaded from environment variables and CLI overrides."""

    openrouter_api_key: str | None = None
    model: str = "x-ai/grok-4.3"
    vision_model: str = "openai/gpt-5.4-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    output_dir: Path = Path("output")
    app_title: str = "Paper Digest"
    site_url: str | None = None
    vision_parse_enabled: bool = True
    vision_parse_mode: str = "auto"
    max_vision_pages: int = 8
    vision_parse_concurrency: int = 5
    force_parse: bool = False
    reasoning_enabled: bool = True
    reasoning_effort: str = "high"
    verbose: bool = True
    request_timeout_seconds: float = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        vision_model: str | None = None,
        output_dir: Path | None = None,
        vision_parse_enabled: bool | None = None,
        vision_parse_mode: str | None = None,
        max_vision_pages: int | None = None,
        vision_parse_concurrency: int | None = None,
        force_parse: bool | None = None,
        reasoning_enabled: bool | None = None,
        reasoning_effort: str | None = None,
        verbose: bool | None = None,
    ) -> Settings:
        load_dotenv()
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            model=model or os.getenv("OPENROUTER_MODEL", cls.model),
            vision_model=vision_model or os.getenv("OPENROUTER_VISION_MODEL", cls.vision_model),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", cls.openrouter_base_url),
            output_dir=output_dir or Path(os.getenv("PAPER_DIGEST_OUTPUT_DIR", "output")),
            app_title=os.getenv("OPENROUTER_APP_TITLE", cls.app_title),
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            vision_parse_enabled=cls._env_bool("PAPER_DIGEST_VISION_PARSE", True)
            if vision_parse_enabled is None
            else vision_parse_enabled,
            vision_parse_mode=vision_parse_mode
            or os.getenv("PAPER_DIGEST_VISION_PARSE_MODE", cls.vision_parse_mode),
            max_vision_pages=max_vision_pages
            if max_vision_pages is not None
            else int(os.getenv("PAPER_DIGEST_MAX_VISION_PAGES", "8")),
            vision_parse_concurrency=vision_parse_concurrency
            if vision_parse_concurrency is not None
            else int(os.getenv("PAPER_DIGEST_VISION_PARSE_CONCURRENCY", "5")),
            force_parse=cls._env_bool("PAPER_DIGEST_FORCE_PARSE", False)
            if force_parse is None
            else force_parse,
            reasoning_enabled=cls._env_bool("OPENROUTER_REASONING", True)
            if reasoning_enabled is None
            else reasoning_enabled,
            reasoning_effort=reasoning_effort
            or os.getenv("OPENROUTER_REASONING_EFFORT", cls.reasoning_effort),
            verbose=cls._env_bool("PAPER_DIGEST_VERBOSE", True) if verbose is None else verbose,
            request_timeout_seconds=float(os.getenv("PAPER_DIGEST_TIMEOUT_SECONDS", "120")),
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
