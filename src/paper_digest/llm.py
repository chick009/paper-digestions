"""OpenRouter client helpers."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from paper_digest.config import Settings
from paper_digest.tracing import TraceWriter

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM request cannot produce a valid response."""


class LLMHTTPError(LLMError):
    """Raised when OpenRouter returns an HTTP error."""

    def __init__(self, step: str, status_code: int, message: str) -> None:
        super().__init__(f"{step} failed with HTTP {status_code}: {message}")
        self.step = step
        self.status_code = status_code
        self.message = message


class OpenRouterClient:
    """Small OpenRouter wrapper with structured-output validation."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY is required to call OpenRouter.")
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.openrouter_base_url,
            timeout=settings.request_timeout_seconds,
        )

    def complete_json(
        self,
        *,
        step: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        trace: TraceWriter | None = None,
        model: str | None = None,
        prompt_version: str = "v1",
        max_retries: int = 2,
    ) -> ModelT:
        schema = response_model.model_json_schema()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{user_prompt}\n\nReturn only valid JSON matching this schema:\n"
                    f"{json.dumps(schema, ensure_ascii=False)}"
                ),
            },
        ]
        return self._request_json(
            step=step,
            messages=messages,
            response_model=response_model,
            trace=trace,
            model=model or self.settings.model,
            prompt_version=prompt_version,
            max_retries=max_retries,
        )

    def vision_json(
        self,
        *,
        step: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
        response_model: type[ModelT],
        trace: TraceWriter | None = None,
        prompt_version: str = "v1",
        max_retries: int = 1,
    ) -> ModelT:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_encode_image(image_path)}",
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "Return only valid JSON matching this schema:\n"
                    f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False)}"
                ),
            }
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self._request_json(
            step=step,
            messages=messages,
            response_model=response_model,
            trace=trace,
            model=self.settings.vision_model,
            prompt_version=prompt_version,
            max_retries=max_retries,
        )

    def _request_json(
        self,
        *,
        step: str,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        trace: TraceWriter | None,
        model: str,
        prompt_version: str,
        max_retries: int,
    ) -> ModelT:
        validation_errors: list[str] = []
        started = time.perf_counter()
        for attempt in range(max_retries + 1):
            payload = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            if self.settings.reasoning_enabled:
                payload["reasoning"] = {"effort": self.settings.reasoning_effort}
            if self.settings.verbose:
                print(f"[llm] {step}: calling {model} (attempt {attempt + 1})", flush=True)
            response = self._client.post("/chat/completions", headers=self._headers(), json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _safe_response_text(exc.response)
                if trace:
                    trace.record(
                        step,
                        prompt_version=prompt_version,
                        input_summary={"attempt": attempt + 1, "message_count": len(messages)},
                        output={"error": detail},
                        model=model,
                        validation_errors=[detail],
                    )
                raise LLMHTTPError(step, exc.response.status_code, detail) from exc
            data = response.json()
            try:
                content = _extract_completion_content(data)
            except LLMError as exc:
                detail = json.dumps(data, ensure_ascii=False)[:2000]
                if trace:
                    trace.record(
                        step,
                        prompt_version=prompt_version,
                        input_summary={"attempt": attempt + 1, "message_count": len(messages)},
                        output={"unexpected_response": detail},
                        model=model,
                        validation_errors=[str(exc)],
                    )
                raise LLMError(f"{step}: {exc}") from exc
            try:
                parsed = response_model.model_validate_json(_extract_json(content))
                latency_ms = (time.perf_counter() - started) * 1000
                if trace:
                    trace.record(
                        step,
                        prompt_version=prompt_version,
                        input_summary={"attempt": attempt + 1, "message_count": len(messages)},
                        output=parsed,
                        model=model,
                        token_usage=data.get("usage", {}),
                        latency_ms=latency_ms,
                        validation_errors=validation_errors,
                        notes=_reasoning_notes(data),
                    )
                if self.settings.verbose:
                    print(f"[llm] {step}: validated {response_model.__name__}", flush=True)
                return parsed
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                validation_errors.append(str(exc))
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response did not validate. Return corrected JSON only. "
                            f"Validation error: {exc}"
                        ),
                    }
                )
        raise LLMError(f"{step} failed schema validation: {validation_errors[-1]}")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "X-Title": self.settings.app_title,
        }
        if self.settings.site_url:
            headers["HTTP-Referer"] = self.settings.site_url
        return headers


def _extract_completion_content(data: dict[str, Any]) -> str:
    """Normalize OpenRouter-compatible chat completion JSON into a text string."""
    if "error" in data:
        err = data["error"]
        if isinstance(err, dict):
            msg = err.get("message", json.dumps(err, ensure_ascii=False))
        else:
            msg = str(err)
        raise LLMError(f"API error: {msg}")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise LLMError("Missing or empty 'choices' in API response")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMError("Invalid choice shape in API response")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMError("Missing message in API choice")
    content = message.get("content")
    if content is None:
        raise LLMError("Missing message.content in API response")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        joined = "\n".join(p for p in parts if p).strip()
        if joined:
            return joined
        raise LLMError("Could not extract text from multimodal message.content")
    return str(content)


def _extract_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    if content.startswith("{"):
        return content
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return content[start : end + 1]


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _safe_response_text(response: httpx.Response) -> str:
    try:
        return response.text[:2000]
    except Exception:
        return response.reason_phrase


def _reasoning_notes(data: dict[str, Any]) -> list[str]:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return []
    notes: list[str] = []
    if "reasoning" in message:
        notes.append("Provider returned visible reasoning content; trace stores model output only.")
    if "reasoning_details" in message:
        notes.append("Provider returned reasoning details metadata.")
    return notes
