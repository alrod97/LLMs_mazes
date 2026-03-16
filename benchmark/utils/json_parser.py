"""Best-effort parsing for JSON-only model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParsedJSONResult:
    parsed: dict[str, Any] | None
    success: bool
    error: str | None = None
    extracted_text: str | None = None


def parse_json_response(raw_text: str) -> ParsedJSONResult:
    text = raw_text.strip()
    if not text:
        return ParsedJSONResult(parsed=None, success=False, error="Empty response text")

    direct = _try_parse(text)
    if direct.success:
        return direct

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ParsedJSONResult(parsed=None, success=False, error=direct.error)

    snippet = text[start : end + 1]
    extracted = _try_parse(snippet)
    if extracted.success:
        extracted.extracted_text = snippet
        return extracted

    return ParsedJSONResult(parsed=None, success=False, error=extracted.error or direct.error)


def _try_parse(candidate: str) -> ParsedJSONResult:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParsedJSONResult(
            parsed=None,
            success=False,
            error=f"JSON decode error at line {exc.lineno} column {exc.colno}: {exc.msg}",
        )

    if not isinstance(parsed, dict):
        return ParsedJSONResult(parsed=None, success=False, error="Top-level JSON value must be an object")

    return ParsedJSONResult(parsed=parsed, success=True)
