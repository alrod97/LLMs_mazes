"""CLI runner for the visually grounded maze benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.models.anthropic_adapter import AnthropicAdapter
from benchmark.models.base import ModelResponse, VisionModelAdapter
from benchmark.models.dashscope_adapter import DashScopeAdapter
from benchmark.models.gemini_adapter import GeminiAdapter
from benchmark.models.mock_adapter import MockAdapter
from benchmark.models.openai_adapter import OpenAIAdapter
from benchmark.prompts import BENCHMARK_PROMPT, BENCHMARK_PROMPT_NAME, BENCHMARK_PROMPT_VISUAL, BENCHMARK_PROMPT_VISUAL_NAME
from benchmark.utils.evaluation import evaluate_prediction, load_ground_truth
from benchmark.utils.image_loader import MazeImage, load_maze_images
from benchmark.utils.json_parser import ParsedJSONResult, parse_json_response
from benchmark.utils.results_writer import RunWriter
from benchmark.utils.validators import ValidationResult, validate_response_schema


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multimodal maze benchmark.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("mazes_imgs"),
        help="Directory containing maze image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/outputs"),
        help="Directory where benchmark outputs should be written.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model spec in provider:model form. Repeat to add multiple models.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run directory name.",
    )
    parser.add_argument(
        "--maze-name",
        action="append",
        dest="maze_names",
        help="Maze stem to run, for example maze_9. Repeat to select multiple mazes.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort passed to OpenAI reasoning models.",
    )
    parser.add_argument(
        "--anthropic-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "max"],
        help="Anthropic adaptive thinking effort. Use with Claude Sonnet/Opus models that support effort.",
    )
    parser.add_argument(
        "--anthropic-thinking-budget",
        type=int,
        default=None,
        help="Anthropic extended thinking budget_tokens. Use instead of --anthropic-effort.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used across adapters when supported.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens requested from each model.",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=Path("benchmark/ground_truth/maze_annotations.json"),
        help="Optional JSON file with per-maze ground-truth shortest-path annotations.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one model call fails.",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=2,
        help="How many times to rerun a maze/model pair when JSON parsing fails.",
    )
    parser.add_argument(
        "--visual-prompt",
        action="store_true",
        help="Use the visual-intuition prompt that forbids grid/matrix reasoning.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    images = load_maze_images(args.image_dir)
    if args.maze_names:
        selected = set(args.maze_names)
        images = [image for image in images if image.name in selected]
        if not images:
            raise ValueError(f"No mazes matched --maze-name values: {', '.join(args.maze_names)}")
    model_specs = args.models or ["openai:gpt-5.4", "openai:gpt-5.1"]
    adapters = [build_adapter(spec, args) for spec in model_specs]
    ground_truth = load_ground_truth(args.ground_truth_file) if args.ground_truth_file.exists() else {}

    run_name = args.run_name or default_run_name(model_specs)
    writer = RunWriter(args.output_dir, run_name)

    if args.visual_prompt:
        active_prompt = BENCHMARK_PROMPT_VISUAL
        active_prompt_name = BENCHMARK_PROMPT_VISUAL_NAME
    else:
        active_prompt = BENCHMARK_PROMPT
        active_prompt_name = BENCHMARK_PROMPT_NAME

    summary_rows: list[dict[str, Any]] = []
    for adapter in adapters:
        for maze in images:
            response, parsed_result, validation_result, attempts_used = generate_with_parse_retries(
                adapter=adapter,
                prompt_text=active_prompt,
                image_path=maze.path,
                parse_retries=args.parse_retries,
            )
            row = handle_single_result(
                writer=writer,
                adapter=adapter,
                maze=maze,
                response=response,
                ground_truth=ground_truth.get(maze.name),
                parsed_result=parsed_result,
                validation_result=validation_result,
                attempts_used=attempts_used,
            )
            summary_rows.append(row)
            print(
                f"[{row['model_id']}] {maze.path.name}: "
                f"parsed={row['json_parse_success']} valid={row['schema_valid']} error={row['error_message'] or '-'}"
            )
            if args.stop_on_error and row["error_message"]:
                writer.write_summary_csv(summary_rows)
                writer.write_summary_jsonl(summary_rows)
                writer.write_markdown_report(render_markdown_report(run_name, model_specs, images, summary_rows))
                return 1

    writer.write_summary_csv(summary_rows)
    writer.write_summary_jsonl(summary_rows)
    writer.write_markdown_report(render_markdown_report(run_name, model_specs, images, summary_rows))
    print(f"Outputs written to: {writer.run_dir}")
    return 0


def build_adapter(spec: str, args: argparse.Namespace) -> VisionModelAdapter:
    if ":" not in spec:
        raise ValueError(f"Invalid model spec '{spec}'. Expected provider:model.")

    provider, model_name = spec.split(":", maxsplit=1)
    provider = provider.strip().lower()
    model_name = model_name.strip()

    if provider == "openai":
        return OpenAIAdapter(
            model_name,
            reasoning_effort=args.openai_reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
        )
    if provider == "anthropic":
        return AnthropicAdapter(
            model_name,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
            thinking_effort=args.anthropic_effort,
            thinking_budget_tokens=args.anthropic_thinking_budget,
        )
    if provider == "gemini":
        return GeminiAdapter(
            model_name,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
        )
    if provider == "dashscope":
        return DashScopeAdapter(
            model_name,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout_seconds,
        )
    if provider == "mock":
        return MockAdapter(model_name)

    raise ValueError(f"Unsupported provider: {provider}")


def handle_single_result(
    *,
    writer: RunWriter,
    adapter: VisionModelAdapter,
    maze: MazeImage,
    response: ModelResponse,
    ground_truth: Any,
    parsed_result: ParsedJSONResult,
    validation_result: ValidationResult,
    attempts_used: int,
) -> dict[str, Any]:
    evaluation_result = evaluate_prediction(
        ground_truth,
        parsed_result.parsed,
        schema_valid=validation_result.valid,
    )

    writer.write_raw_text(adapter.model_id, maze.name, response.raw_text)
    writer.write_raw_payload(
        adapter.model_id,
        maze.name,
        {
            "request": sanitize_request_payload(response.request_payload),
            "response": response.raw_payload,
            "status_code": response.status_code,
            "usage": response.usage,
            "latency_seconds": response.latency_seconds,
            "error": response.error,
        },
    )
    if parsed_result.success and parsed_result.parsed is not None:
        writer.write_parsed_json(adapter.model_id, maze.name, parsed_result.parsed)

    parsed = parsed_result.parsed or {}
    return {
        "run_name": writer.run_dir.name,
        "prompt_name": BENCHMARK_PROMPT_NAME,
        "model_id": adapter.model_id,
        "provider": response.provider,
        "model_name": response.model_name,
        "maze_name": maze.name,
        "maze_file": str(maze.path),
        "json_parse_success": parsed_result.success,
        "schema_valid": validation_result.valid,
        "reachable": parsed.get("reachable"),
        "shortest_path_length": parsed.get("shortest_path_length"),
        "path_length": len(parsed.get("path", [])) if isinstance(parsed.get("path"), list) else None,
        "confidence": parsed.get("confidence"),
        "image_ambiguity": parsed.get("image_ambiguity"),
        "grid_size": json.dumps(parsed.get("grid_size")) if "grid_size" in parsed else None,
        "status_code": response.status_code,
        "latency_seconds": round(response.latency_seconds or 0.0, 3),
        "usage": json.dumps(response.usage) if response.usage is not None else None,
        "parse_error": parsed_result.error,
        "validation_errors": " | ".join(validation_result.errors) if validation_result.errors else "",
        "error_message": response.error or "",
        "attempts_used": attempts_used,
        "expected_reachable": evaluation_result["expected_reachable"],
        "expected_shortest_path_length": evaluation_result["expected_shortest_path_length"],
        "accepted_shortest_paths_count": evaluation_result["accepted_shortest_paths_count"],
        "predicted_path_string": evaluation_result["predicted_path_string"],
        "reachability_correct": evaluation_result["reachability_correct"],
        "shortest_path_length_correct": evaluation_result["shortest_path_length_correct"],
        "path_exact_match": evaluation_result["path_exact_match"],
        "solved": evaluation_result["solved"],
    }


def default_run_name(model_specs: list[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    providers = sorted({spec.split(":", maxsplit=1)[0] for spec in model_specs})
    return f"{timestamp}_{'_'.join(providers)}_maze_benchmark"


def generate_with_parse_retries(
    *,
    adapter: VisionModelAdapter,
    prompt_text: str,
    image_path: Path,
    parse_retries: int,
) -> tuple[ModelResponse, ParsedJSONResult, ValidationResult, int]:
    last_response: ModelResponse | None = None
    last_parsed: ParsedJSONResult | None = None
    last_validation: ValidationResult | None = None

    for attempt_index in range(parse_retries + 1):
        response = adapter.generate(prompt_text, image_path)
        parsed_result = parse_json_response(response.raw_text)
        validation_result = validate_response_schema(parsed_result.parsed)
        last_response = response
        last_parsed = parsed_result
        last_validation = validation_result
        if parsed_result.success:
            return response, parsed_result, validation_result, attempt_index + 1

    assert last_response is not None
    assert last_parsed is not None
    assert last_validation is not None
    return last_response, last_parsed, last_validation, parse_retries + 1


def sanitize_request_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None

    cloned = json.loads(json.dumps(payload))
    for message in cloned.get("input", []):
        for content in message.get("content", []):
            if content.get("type") == "input_image" and "image_url" in content:
                content["image_url"] = "<redacted_data_url>"

    for message in cloned.get("messages", []):
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                source = item.get("source", {})
                if "data" in source:
                    source["data"] = "<redacted_base64>"
            if item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict) and "url" in image_url:
                    image_url["url"] = "<redacted_data_url>"

    for content in cloned.get("contents", []):
        for part in content.get("parts", []):
            inline_data = part.get("inlineData")
            if isinstance(inline_data, dict) and "data" in inline_data:
                inline_data["data"] = "<redacted_base64>"

    return cloned


def render_markdown_report(
    run_name: str,
    model_specs: list[str],
    images: list[MazeImage],
    rows: list[dict[str, Any]],
) -> str:
    per_model = defaultdict(list)
    for row in rows:
        per_model[row["model_id"]].append(row)

    lines = [
        "# Maze Benchmark Summary",
        "",
        f"- Run: `{run_name}`",
        f"- Prompt: `{BENCHMARK_PROMPT_NAME}`",
        f"- Models: {', '.join(f'`{spec}`' for spec in model_specs)}",
        f"- Mazes processed: {len(images)}",
        "",
        "## Parse Summary",
        "",
        "| Model | Parsed OK | Schema OK |",
        "| --- | ---: | ---: |",
    ]

    for model_id, model_rows in sorted(per_model.items()):
        parsed_ok = sum(1 for row in model_rows if row["json_parse_success"])
        schema_ok = sum(1 for row in model_rows if row["schema_valid"])
        lines.append(f"| `{model_id}` | {parsed_ok}/{len(model_rows)} | {schema_ok}/{len(model_rows)} |")

    lines.extend(
        [
            "",
            "## Score Summary",
            "",
            "| Model | Reachability | Length | Exact Shortest Path | Solved |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for model_id, model_rows in sorted(per_model.items()):
        gt_rows = [row for row in model_rows if row["expected_reachable"] != "" and row["expected_reachable"] is not None]
        reachable_correct = sum(1 for row in gt_rows if str(row["reachability_correct"]) == "True")
        length_correct = sum(1 for row in gt_rows if str(row["shortest_path_length_correct"]) == "True")
        path_match = sum(1 for row in gt_rows if str(row["path_exact_match"]) == "True")
        solved = sum(1 for row in gt_rows if str(row["solved"]) == "True")
        total = len(gt_rows)
        lines.append(f"| `{model_id}` | {reachable_correct}/{total} | {length_correct}/{total} | {path_match}/{total} | {solved}/{total} |")

    lines.extend(
        [
            "",
            "## Per-Maze Results",
            "",
            "| Model | Maze | Parsed | Reachability | Length | Path | Solved | Confidence | Error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in rows:
        lines.append(
            "| "
            f"`{row['model_id']}` | `{row['maze_name']}` | "
            f"{yes_no(row['json_parse_success'])} | {yes_no(row['reachability_correct'])} | "
            f"{yes_no(row['shortest_path_length_correct'])} | {yes_no(row['path_exact_match'])} | "
            f"{yes_no(row['solved'])} | {stringify(row['confidence'])} | "
            f"{stringify(row['error_message'] or row['parse_error'] or row['validation_errors'])} |"
        )

    lines.extend(
        [
            "",
            "## TODO",
            "",
            "- Expand the accepted-path annotations if additional shortest-path variants are discovered.",
            "- Add image-derived path verification so alternative shortest paths can be accepted automatically.",
            "- Compare reasoning-effort sweeps against the medium baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def yes_no(value: Any) -> str:
    return "yes" if value else "no"


def stringify(value: Any) -> str:
    if value is None:
        return "-"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
