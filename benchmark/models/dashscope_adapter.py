"""DashScope OpenAI-compatible adapter for Qwen models."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.base import ModelResponse, VisionModelAdapter
from benchmark.utils.image_loader import detect_image_mime_type, encode_image_data_url


class DashScopeAdapter(VisionModelAdapter):
    provider = "dashscope"

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        enable_thinking: bool = True,
        thinking_budget: int | None = 2048,
        max_output_tokens: int = 16384,
        timeout_seconds: int = 180,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.endpoint = f"{self.base_url}/chat/completions"

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")

    def generate(self, prompt_text: str, image_path: Path) -> ModelResponse:
        mime_type = detect_image_mime_type(image_path)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image_data_url(image_path, mime_type),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                },
            ],
            "stream": False,
            "max_tokens": self.max_output_tokens,
            "enable_thinking": self.enable_thinking,
        }
        if self.enable_thinking and self.thinking_budget is not None:
            payload["thinking_budget"] = self.thinking_budget

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
                error=last_error or "DashScope request failed before a response was returned",
                latency_seconds=time.perf_counter() - started_at,
            )

        return ModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            raw_text=_extract_dashscope_text(response_payload),
            raw_payload=response_payload,
            request_payload=payload,
            status_code=status_code,
            usage=response_payload.get("usage"),
            latency_seconds=time.perf_counter() - started_at,
            error=_extract_dashscope_error(response_payload) or _extract_incomplete_reason(response_payload),
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
        raise RuntimeError(f"HTTP {exc.code}: {_extract_dashscope_error(payload)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _extract_dashscope_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _extract_dashscope_error(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message")
    if isinstance(error, str):
        return error
    return None


def _extract_incomplete_reason(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    finish_reason = choices[0].get("finish_reason")
    if finish_reason and str(finish_reason).lower() in {"length", "max_tokens"}:
        return f"Incomplete response: {finish_reason}"
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
