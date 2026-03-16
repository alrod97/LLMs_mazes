"""Local mock adapter for dry runs without external API calls."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.models.base import ModelResponse, VisionModelAdapter


class MockAdapter(VisionModelAdapter):
    provider = "mock"

    def generate(self, prompt_text: str, image_path: Path) -> ModelResponse:
        del prompt_text
        payload = {
            "grid_size": [8, 8],
            "start_found": True,
            "goal_found": True,
            "reachable": False,
            "shortest_path_length": None,
            "path": [],
            "confidence": "low",
            "image_ambiguity": "minor",
        }
        return ModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            raw_text=json.dumps(payload),
            raw_payload={"mock_image": image_path.name, "response": payload},
        )
