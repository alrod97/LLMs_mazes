"""Run benchmark with text grid input instead of images.

Usage:
    python3 -m benchmark.text_grid_runner \
        --model anthropic:claude-sonnet-4-6 \
        --anthropic-effort low \
        --maze-name gen_maze_001 --maze-name gen_maze_014
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.models.anthropic_adapter import AnthropicAdapter
from benchmark.models.openai_adapter import OpenAIAdapter
from benchmark.models.base import ModelResponse
from benchmark.prompts import BENCHMARK_PROMPT
from benchmark.utils.evaluation import evaluate_prediction, load_ground_truth
from benchmark.utils.json_parser import parse_json_response
from benchmark.utils.validators import validate_response_schema
from scripts.maze_generator.maze_model import CellType, MazeConfig
from scripts.maze_generator.generator import generate_maze
from scripts.maze_generator.__main__ import _SPECS

CELL_CHAR = {
    CellType.FLOOR: ".",
    CellType.WALL: "#",
    CellType.TRAP: "T",
    CellType.START: "S",
    CellType.GOAL: "G",
}

TEXT_PROMPT = """You are solving a maze puzzle from the text grid below.

The grid uses these symbols:
- S = start (player position)
- G = goal (treasure)
- . = walkable floor
- # = wall (blocked)
- T = trap (blocked, dangerous)

Important rules:
- Only use 4-directional movement: up, down, left, right.
- No diagonal moves.
- Wall tiles (#) and trap tiles (T) are blocked and cannot be traversed.
- The shortest path length is the number of moves, not the number of visited cells.

Find the shortest path from S to G.

Return the answer in this exact JSON format:

{
  "grid_size": [rows, cols],
  "start_found": true_or_false,
  "goal_found": true_or_false,
  "reachable": true_or_false,
  "shortest_path_length": integer_or_null,
  "path": ["R","D","L","U"] or [],
  "confidence": "high" | "medium" | "low",
  "image_ambiguity": "none" | "minor" | "major"
}

Rules for the JSON:
- If reachable is false, set "shortest_path_length" to null and "path" to [].
- If reachable is true, provide one valid shortest path.
- Output JSON only, with no extra text.

Here is the maze grid:

"""


def maze_to_text(spec) -> tuple[str, object]:
    config = MazeConfig(
        rows=spec.rows, cols=spec.cols,
        wall_density=spec.wall_density, trap_count=spec.trap_count,
        force_unreachable=spec.force_unreachable,
        border_walls=spec.border_walls, seed=spec.seed,
        start_pos=spec.start_pos, goal_pos=spec.goal_pos,
    )
    maze = generate_maze(config)
    lines = []
    for r in range(maze.rows):
        row = " ".join(CELL_CHAR[maze.grid[r][c]] for c in range(maze.cols))
        lines.append(row)
    return "\n".join(lines), maze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--maze-name", action="append", dest="maze_names", required=True)
    parser.add_argument("--anthropic-effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--ground-truth-file", type=Path,
                        default=Path("generated_mazes/maze_annotations.json"))
    parser.add_argument("--run-name", default="text_grid_run")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth_file) if args.ground_truth_file.exists() else {}
    spec_map = {s.name: s for s in _SPECS}

    provider, model_name = args.model.split(":", 1)

    # Build output dir
    out_dir = Path("benchmark/outputs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for maze_name in args.maze_names:
        spec = spec_map.get(maze_name)
        if not spec:
            print(f"  SKIP {maze_name}: not found in specs")
            continue

        grid_text, maze = maze_to_text(spec)
        prompt = TEXT_PROMPT + grid_text

        # Call the model with text-only (no image)
        started = time.perf_counter()
        if provider == "anthropic":
            import os, urllib.request, urllib.error
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            payload = {
                "model": model_name,
                "system": prompt,
                "max_tokens": args.max_output_tokens,
                "temperature": 1.0,
                "messages": [{"role": "user", "content": "Solve the maze above."}],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": args.anthropic_effort},
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=body, headers=headers, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                raw_text = ""
                for item in resp_data.get("content", []):
                    if item.get("type") == "text":
                        raw_text += item.get("text", "")
                error = None
                usage = resp_data.get("usage")
            except Exception as e:
                raw_text = ""
                error = str(e)
                usage = None
        else:
            print(f"  Provider {provider} not supported for text-only mode")
            continue

        latency = time.perf_counter() - started

        # Parse and evaluate
        parsed = parse_json_response(raw_text)
        validation = validate_response_schema(parsed.parsed)
        gt = ground_truth.get(maze_name)
        evaluation = evaluate_prediction(gt, parsed.parsed, schema_valid=validation.valid)

        solved = evaluation.get("solved", False)
        reach_ok = evaluation.get("reachability_correct", False)

        # Save raw text
        raw_dir = out_dir / "raw_text"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{maze_name}.txt").write_text(raw_text, encoding="utf-8")

        results.append({
            "maze_name": maze_name,
            "solved": solved,
            "reachability_correct": reach_ok,
            "json_parse_success": parsed.success,
            "schema_valid": validation.valid,
            "latency_seconds": round(latency, 2),
            "error": error,
            "usage": usage,
        })

        status = "SOLVED" if solved else "fail"
        print(f"  [{args.model}] {maze_name}: {status} (reach={reach_ok}, lat={latency:.1f}s)")

    # Write summary
    with open(out_dir / "summary.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    solved_count = sum(1 for r in results if r["solved"])
    print(f"\nTotal: {solved_count}/{len(results)} solved")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
