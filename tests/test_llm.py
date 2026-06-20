from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from paper_digest.config import Settings
from paper_digest.llm import OpenRouterClient


class TinyResponse(BaseModel):
    value: str


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": '{"value": "ok"}'}}]}


class FakeHTTPClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, path: str, *, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
        self.payloads.append({"path": path, "headers": headers, "json": json})
        return FakeResponse()


def test_complete_json_does_not_add_fusion_tool_by_default(monkeypatch) -> None:
    http_client = FakeHTTPClient()
    monkeypatch.setattr("paper_digest.llm.httpx.Client", lambda **_: http_client)
    client = OpenRouterClient(
        Settings(openrouter_api_key="test", reasoning_enabled=False, verbose=False)
    )

    result = client.complete_json(
        step="tiny",
        system_prompt="Return JSON.",
        user_prompt="Return ok.",
        response_model=TinyResponse,
    )

    assert result.value == "ok"
    payload = http_client.payloads[0]["json"]
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_complete_json_adds_openrouter_fusion_tool(monkeypatch) -> None:
    http_client = FakeHTTPClient()
    monkeypatch.setattr("paper_digest.llm.httpx.Client", lambda **_: http_client)
    client = OpenRouterClient(
        Settings(
            openrouter_api_key="test",
            fusion_enabled=True,
            fusion_analysis_models=("model-a", "model-b"),
            fusion_judge_model="judge-model",
            fusion_max_tool_calls=3,
            fusion_temperature=0.4,
            fusion_force=True,
            reasoning_enabled=True,
            reasoning_effort="medium",
            verbose=False,
        )
    )

    result = client.complete_json(
        step="tiny",
        system_prompt="Return JSON.",
        user_prompt="Return ok.",
        response_model=TinyResponse,
        model="outer-model",
    )

    assert result.value == "ok"
    payload = http_client.payloads[0]["json"]
    assert payload["model"] == "outer-model"
    assert payload["tool_choice"] == "required"
    assert payload["tools"] == [
        {
            "type": "openrouter:fusion",
            "parameters": {
                "analysis_models": ["model-a", "model-b"],
                "model": "judge-model",
                "max_tool_calls": 3,
                "temperature": 0.4,
                "reasoning": {"effort": "medium"},
            },
        }
    ]
