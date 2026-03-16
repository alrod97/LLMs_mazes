"""Google Gemini multimodal adapter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.base import ModelResponse, VisionModelAdapter
from benchmark.utils.image_loader import detect_image_mime_type, encode_image_base64


class GeminiAdapter(VisionModelAdapter):
    provider = "gemini"

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        timeout_seconds: int = 180,
    ) -> None:
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

    def generate(self, prompt_text: str, image_path: Path) -> ModelResponse:
        mime_type = detect_image_mime_type(image_path)
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": prompt_text,
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": encode_image_base64(image_path),
                            }
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model_name, safe='')}:generateContent?key={urllib.parse.quote(self.api_key, safe='')}"
        )

        headers = {"Content-Type": "application/json"}
        started_at = time.perf_counter()
        try:
            status_code, response_payload = _post_json(
                endpoint,
                payload,
                headers=headers,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                raw_text="",
                raw_payload={},
                request_payload=payload,
                error=str(exc),
                latency_seconds=time.perf_counter() - started_at,
            )

        return ModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            raw_text=_extract_gemini_text(response_payload),
            raw_payload=response_payload,
            request_payload=payload,
            status_code=status_code,
            usage=response_payload.get("usageMetadata"),
            latency_seconds=time.perf_counter() - started_at,
            error=_extract_gemini_error(response_payload),
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
        raise RuntimeError(f"HTTP {exc.code}: {_extract_gemini_error(payload)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_gemini_error(payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if isinstance(error, dict):
        return error.get("message")
    if isinstance(error, str):
        return error
    return None
