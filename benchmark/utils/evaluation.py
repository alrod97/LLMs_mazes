"""Ground-truth loading and scoring helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MazeGroundTruth:
    maze_name: str
    reachable: bool
    shortest_path_length: int | None
    accepted_shortest_paths: tuple[str, ...]


def load_ground_truth(path: Path) -> dict[str, MazeGroundTruth]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ground_truth: dict[str, MazeGroundTruth] = {}

    for maze_name, record in payload.items():
        reachable = bool(record["reachable"])
        accepted_shortest_paths = tuple(record.get("accepted_shortest_paths", []))
        if not reachable and accepted_shortest_paths:
            raise ValueError(f"{maze_name} is unreachable but still has accepted paths")
        if reachable and not accepted_shortest_paths:
            raise ValueError(f"{maze_name} is reachable but has no accepted paths")

        lengths = {len(path) for path in accepted_shortest_paths}
        if len(lengths) > 1:
            raise ValueError(f"{maze_name} has accepted shortest paths with inconsistent lengths")

        shortest_path_length = next(iter(lengths)) if lengths else None
        ground_truth[maze_name] = MazeGroundTruth(
            maze_name=maze_name,
            reachable=reachable,
            shortest_path_length=shortest_path_length,
            accepted_shortest_paths=accepted_shortest_paths,
        )

    return ground_truth


def evaluate_prediction(
    ground_truth: MazeGroundTruth | None,
    parsed_response: dict[str, Any] | None,
    *,
    schema_valid: bool,
) -> dict[str, Any]:
    predicted_path = normalize_path(parsed_response.get("path") if parsed_response else None)
    metrics = {
        "ground_truth_available": ground_truth is not None,
        "expected_reachable": None,
        "expected_shortest_path_length": None,
        "accepted_shortest_paths_count": None,
        "predicted_path_string": predicted_path,
        "reachability_correct": None,
        "shortest_path_length_correct": None,
        "path_exact_match": None,
        "solved": None,
    }

    if ground_truth is None:
        return metrics

    metrics["expected_reachable"] = ground_truth.reachable
    metrics["expected_shortest_path_length"] = ground_truth.shortest_path_length
    metrics["accepted_shortest_paths_count"] = len(ground_truth.accepted_shortest_paths)

    if parsed_response is None or not schema_valid:
        metrics["reachability_correct"] = False
        metrics["shortest_path_length_correct"] = False if ground_truth.reachable else None
        metrics["path_exact_match"] = False if ground_truth.reachable else None
        metrics["solved"] = False
        return metrics

    predicted_reachable = parsed_response.get("reachable")
    predicted_length = parsed_response.get("shortest_path_length")
    reachability_correct = predicted_reachable == ground_truth.reachable
    metrics["reachability_correct"] = reachability_correct

    if not ground_truth.reachable:
        metrics["shortest_path_length_correct"] = predicted_length is None
        metrics["path_exact_match"] = predicted_path == ""
        metrics["solved"] = bool(reachability_correct and predicted_length is None and predicted_path == "")
        return metrics

    length_correct = predicted_length == ground_truth.shortest_path_length
    path_match = predicted_path in ground_truth.accepted_shortest_paths
    metrics["shortest_path_length_correct"] = length_correct
    metrics["path_exact_match"] = path_match
    metrics["solved"] = bool(reachability_correct and length_correct and path_match)
    return metrics


def normalize_path(path_value: Any) -> str:
    if not isinstance(path_value, list):
        return ""
    return "".join(token for token in path_value if isinstance(token, str))
