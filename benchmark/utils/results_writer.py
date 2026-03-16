"""Writers for benchmark outputs and reports."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_") or "item"


class RunWriter:
    def __init__(self, output_root: Path, run_name: str) -> None:
        self.run_dir = output_root / run_name
        self.raw_text_dir = self.run_dir / "raw_text"
        self.raw_payload_dir = self.run_dir / "raw_payload"
        self.parsed_dir = self.run_dir / "parsed"
        for directory in (self.run_dir, self.raw_text_dir, self.raw_payload_dir, self.parsed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def write_raw_text(self, model_id: str, maze_name: str, raw_text: str) -> Path:
        path = self._target_path(self.raw_text_dir, model_id, maze_name, ".txt")
        path.write_text(raw_text, encoding="utf-8")
        return path

    def write_raw_payload(self, model_id: str, maze_name: str, payload: Any) -> Path:
        path = self._target_path(self.raw_payload_dir, model_id, maze_name, ".json")
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def write_parsed_json(self, model_id: str, maze_name: str, payload: dict[str, Any]) -> Path:
        path = self._target_path(self.parsed_dir, model_id, maze_name, ".json")
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def write_summary_csv(self, rows: list[dict[str, Any]]) -> Path:
        path = self.run_dir / "summary.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_summary_jsonl(self, rows: list[dict[str, Any]]) -> Path:
        path = self.run_dir / "summary.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True))
                handle.write("\n")
        return path

    def write_markdown_report(self, content: str) -> Path:
        path = self.run_dir / "report.md"
        path.write_text(content, encoding="utf-8")
        return path

    def _target_path(self, base_dir: Path, model_id: str, maze_name: str, suffix: str) -> Path:
        model_dir = base_dir / _slugify(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{_slugify(maze_name)}{suffix}"
