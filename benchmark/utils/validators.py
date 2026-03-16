"""Validation for benchmark response schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXPECTED_KEYS = {
    "grid_size",
    "start_found",
    "goal_found",
    "reachable",
    "shortest_path_length",
    "path",
    "confidence",
    "image_ambiguity",
}
VALID_MOVES = {"U", "D", "L", "R"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_IMAGE_AMBIGUITY = {"none", "minor", "major"}


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_response_schema(data: dict[str, Any] | None) -> ValidationResult:
    if data is None:
        return ValidationResult(valid=False, errors=["No parsed JSON object available"])

    errors: list[str] = []

    missing = sorted(EXPECTED_KEYS - data.keys())
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")

    extra = sorted(data.keys() - EXPECTED_KEYS)
    if extra:
        errors.append(f"Unexpected fields present: {', '.join(extra)}")

    grid_size = data.get("grid_size")
    if not isinstance(grid_size, list) or len(grid_size) != 2 or not all(type(value) is int for value in grid_size):
        errors.append('"grid_size" must be a list of two integers')
    elif any(value <= 0 for value in grid_size):
        errors.append('"grid_size" values must be positive integers')

    for field_name in ("start_found", "goal_found", "reachable"):
        if not isinstance(data.get(field_name), bool):
            errors.append(f'"{field_name}" must be a boolean')

    shortest_path_length = data.get("shortest_path_length")
    if shortest_path_length is not None and not isinstance(shortest_path_length, int):
        errors.append('"shortest_path_length" must be an integer or null')
    elif isinstance(shortest_path_length, int) and shortest_path_length < 0:
        errors.append('"shortest_path_length" must be non-negative')

    path = data.get("path")
    if not isinstance(path, list):
        errors.append('"path" must be a list')
    else:
        invalid_moves = [move for move in path if move not in VALID_MOVES]
        if invalid_moves:
            errors.append(f'"path" contains invalid moves: {", ".join(map(str, invalid_moves))}')

    confidence = data.get("confidence")
    if confidence not in VALID_CONFIDENCE:
        errors.append('"confidence" must be one of: high, medium, low')

    image_ambiguity = data.get("image_ambiguity")
    if image_ambiguity not in VALID_IMAGE_AMBIGUITY:
        errors.append('"image_ambiguity" must be one of: none, minor, major')

    reachable = data.get("reachable")
    if reachable is False:
        if shortest_path_length is not None:
            errors.append('If "reachable" is false, "shortest_path_length" must be null')
        if path != []:
            errors.append('If "reachable" is false, "path" must be []')

    if reachable is True:
        if not isinstance(shortest_path_length, int):
            errors.append('If "reachable" is true, "shortest_path_length" must be an integer')
        elif isinstance(path, list) and len(path) != shortest_path_length:
            errors.append('If "reachable" is true, len("path") must equal "shortest_path_length"')

    if data.get("start_found") is False or data.get("goal_found") is False:
        if confidence != "low":
            errors.append('If start or goal is not found, "confidence" must be "low"')

    return ValidationResult(valid=not errors, errors=errors)
