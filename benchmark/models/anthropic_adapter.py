"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.base import ModelResponse, VisionModelAdapter
from benchmark.utils.image_loader import detect_image_mime_type, encode_image_base64


class AnthropicAdapter(VisionModelAdapter):
    provider = "anthropic"

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        timeout_seconds: int = 180,
        max_retries: int = 2,
        thinking_effort: str | None = None,
        thinking_budget_tokens: int | None = None,
    ) -> None:
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.thinking_effort = thinking_effort
        self.thinking_budget_tokens = thinking_budget_tokens
        self.endpoint = "https://api.anthropic.com/v1/messages"

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        if self.thinking_effort and self.thinking_budget_tokens is not None:
            raise ValueError("Anthropic effort and thinking budget are mutually exclusive")

    def generate(self, prompt_text: str, image_path: Path) -> ModelResponse:
        mime_type = detect_image_mime_type(image_path)
        thinking_enabled = self.thinking_budget_tokens is not None or bool(self.thinking_effort)
        payload = {
            "model": self.model_name,
            "system": prompt_text,
            "max_tokens": self.max_output_tokens,
            # Anthropic requires temperature=1 when explicit thinking is enabled.
            "temperature": 1.0 if thinking_enabled else self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": encode_image_base64(image_path),
                            },
                        }
                    ],
                }
            ],
        }

        if self.thinking_budget_tokens is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
        elif self.thinking_effort:
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": self.thinking_effort}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        started_at = time.perf_counter()
        response_payload: dict[str, Any] | None = None
        status_code: int | None = None
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                status_code, response_payload = _post_json(
                    self.endpoint,
                    payload,
                    headers=headers,
                    timeout_seconds=self.timeout_seconds,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt >= self.max_retries or not _is_retryable_error(last_error):
                    return ModelResponse(
                        provider=self.provider,
                        model_name=self.model_name,
                        raw_text="",
                        raw_payload={},
                        request_payload=payload,
                        error=last_error,
                        latency_seconds=time.perf_counter() - started_at,
                        extra={"attempts": attempt + 1},
                    )
                time.sleep(2**attempt)

        if response_payload is None or status_code is None:
            return ModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                raw_text="",
                raw_payload={},
                request_payload=payload,
                error=last_error or "Anthropic request failed before a response was returned",
                latency_seconds=time.perf_counter() - started_at,
            )

        return ModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            raw_text=_extract_anthropic_text(response_payload),
            raw_payload=response_payload,
            request_payload=payload,
            status_code=status_code,
            usage=response_payload.get("usage"),
            latency_seconds=time.perf_counter() - started_at,
            error=_extract_anthropic_error(response_payload),
        )


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_seconds: int,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
            return response.status, json.loads(raw_body)
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            payload = {"error": {"message": raw_body}}
        raise RuntimeError(f"HTTP {exc.code}: {_extract_anthropic_error(payload)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("content", []):
        if item.get("type") == "text":
            chunks.append(item.get("text", ""))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _extract_anthropic_error(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message")
    if isinstance(error, str):
        return error
    return None


def _is_retryable_error(message: str) -> bool:
    lowered = message.lower()
    retryable_markers = [
        "broken pipe",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "bad record mac",
        "ssl",
        "network error",
        "remote end closed connection",
    ]
    return any(marker in lowered for marker in retryable_markers)
