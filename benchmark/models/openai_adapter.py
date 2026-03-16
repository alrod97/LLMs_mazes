"""OpenAI Responses API adapter for image-input maze runs."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.base import ModelResponse, VisionModelAdapter
from benchmark.utils.image_loader import MAX_OPENAI_IMAGE_BYTES, detect_image_mime_type, encode_image_data_url


class OpenAIAdapter(VisionModelAdapter):
    provider = "openai"

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        reasoning_effort: str = "medium",
        max_output_tokens: int = 8192,
        timeout_seconds: int = 180,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.endpoint = "https://api.openai.com/v1/responses"

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def generate(self, prompt_text: str, image_path: Path) -> ModelResponse:
        mime_type = detect_image_mime_type(image_path)
        image_size = image_path.stat().st_size
        if image_size > MAX_OPENAI_IMAGE_BYTES:
            return ModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                raw_text="",
                raw_payload={},
                error=f"Image exceeds OpenAI 8MB input limit: {image_path.name}",
            )

        payload = {
            "model": self.model_name,
            "instructions": prompt_text,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": encode_image_data_url(image_path, mime_type),
                        }
                    ],
                }
            ],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
            "tools": [],
            "text": {"format": {"type": "text"}},
            "store": False,
        }

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
                error=last_error or "OpenAI request failed before a response was returned",
                latency_seconds=time.perf_counter() - started_at,
            )

        raw_text = _extract_openai_text(response_payload)
        return ModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            raw_text=raw_text,
            raw_payload=response_payload,
            request_payload=payload,
            status_code=status_code,
            usage=response_payload.get("usage"),
            latency_seconds=time.perf_counter() - started_at,
            error=_extract_openai_error(response_payload) or _extract_openai_incomplete(response_payload),
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
        raise RuntimeError(f"HTTP {exc.code}: {_extract_openai_error(payload)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _extract_openai_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _extract_openai_error(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message")
    if isinstance(error, str):
        return error
    return None


def _extract_openai_incomplete(payload: dict[str, Any]) -> str | None:
    if payload.get("status") != "incomplete":
        return None
    details = payload.get("incomplete_details") or {}
    reason = details.get("reason", "unknown")
    return f"Incomplete response: {reason}"


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
