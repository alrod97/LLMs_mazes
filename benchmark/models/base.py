"""Base interfaces for multimodal benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelResponse:
    provider: str
    model_name: str
    raw_text: str
    raw_payload: Any = None
    request_payload: dict[str, Any] | None = None
    status_code: int | None = None
    usage: dict[str, Any] | None = None
    latency_seconds: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model_name}"


class VisionModelAdapter(ABC):
    """Simple interface for models that accept one image plus one prompt."""

    provider: str

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model_name}"

    @abstractmethod
    def generate(self, prompt_text: str, image_path: Path) -> ModelResponse:
        """Run a single multimodal benchmark call."""
