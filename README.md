# Visual Maze Benchmark

This repository contains a small benchmark for visually grounded maze reasoning with multimodal models. Each model receives only a maze image and a fixed prompt, then must decide whether the treasure is reachable and, if so, return one valid shortest path in JSON.

The dataset is the image folder itself. There is no symbolic matrix input, no generated mazes, and no hidden solver inside the model call.

## What Is Included

- `benchmark/`: the benchmark runner, prompt, adapters, parser, validation, and scoring helpers
- `mazes_imgs/`: the 10 maze images used as the dataset
- `benchmark/ground_truth/maze_annotations.json`: manually curated shortest-path annotations used for scoring
- `maze_benchmark_blog.html`: the write-up for the experiment
- `index.html`: a simple redirect so the post is GitHub Pages-friendly

## Benchmark Design

The benchmark keeps the setup intentionally simple:

- load maze images directly from a folder
- send the same fixed prompt to every model
- disable tool use in the API call
- avoid API-enforced structured outputs
- parse and validate JSON locally
- score predictions against manually transcribed shortest-path annotations

The main metric is `solved`: the model must identify reachability correctly and, on reachable mazes, return an accepted shortest path with the correct length.

## Project Layout

```text
benchmark/
  main.py
  prompts.py
  ground_truth/
    maze_annotations.json
  models/
    base.py
    openai_adapter.py
    anthropic_adapter.py
    dashscope_adapter.py
    gemini_adapter.py
    mock_adapter.py
  utils/
    evaluation.py
    image_loader.py
    json_parser.py
    results_writer.py
    validators.py
  outputs/
mazes_imgs/
  maze_1.jpg
  ...
maze_benchmark_blog.html
index.html
README.md
```

## Requirements

- Python 3.10+
- standard library only
- API keys via environment variables when running live models:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GEMINI_API_KEY`
  - `DASHSCOPE_API_KEY`

You can copy `.env.example` and export only the keys you need.

## Quick Start

Smoke test the full pipeline without external APIs:

```bash
python3 -m benchmark.main --model mock:baseline
```

Run the default OpenAI benchmark:

```bash
python3 -m benchmark.main
```

That default run uses:

- `openai:gpt-5.4`
- `openai:gpt-5.1`
- `--openai-reasoning-effort medium`
- `--max-output-tokens 8192`

## Example Commands

OpenAI:

```bash
python3 -m benchmark.main \
  --model openai:gpt-5.4 \
  --model openai:gpt-5.1 \
  --openai-reasoning-effort medium \
  --max-output-tokens 8192
```

Anthropic:

```bash
python3 -m benchmark.main \
  --model anthropic:claude-opus-4-6 \
  --model anthropic:claude-sonnet-4-6 \
  --anthropic-effort medium \
  --max-output-tokens 16384
```

```bash
python3 -m benchmark.main \
  --model anthropic:claude-haiku-4-5-20251001 \
  --anthropic-thinking-budget 4096 \
  --max-output-tokens 16384
```

Google:

The API ids below correspond to the Gemini 3.1 Pro Preview and Gemini 3 Flash runs used in the write-up.

```bash
python3 -m benchmark.main \
  --model gemini:gemini-3-pro-preview \
  --model gemini:gemini-3-flash-preview \
  --max-output-tokens 16384
```

DashScope / Qwen:

```bash
python3 -m benchmark.main \
  --model dashscope:qwen3.5-plus \
  --model dashscope:qwen3.5-flash \
  --max-output-tokens 4096
```

Run a subset of mazes:

```bash
python3 -m benchmark.main \
  --model openai:gpt-5.4 \
  --maze-name maze_1 \
  --maze-name maze_2
```

## Outputs

Every run writes a timestamped directory under `benchmark/outputs/` with:

- `raw_text/`: raw model text per maze
- `raw_payload/`: request and response snapshots with image payloads redacted
- `parsed/`: parsed JSON when recovery succeeds
- `summary.csv`: one row per model x maze
- `summary.jsonl`: JSONL version of the same rows
- `report.md`: short run summary

Generated outputs are intentionally gitignored. The repository keeps the runner and blog post, not local experiment artifacts.

## Prompt And Scoring

- Fixed prompt: `benchmark/prompts.py`
- Ground truth annotations: `benchmark/ground_truth/maze_annotations.json`
- Scoring logic: `benchmark/utils/evaluation.py`

The scorer treats a maze as solved only when:

- reachability is correct
- shortest-path length is correct
- the returned path matches one accepted shortest path annotation

## Blog Post

Open `maze_benchmark_blog.html` directly in a browser, or serve the repository locally:

```bash
python3 -m http.server 8000
```

Then open:

- `http://127.0.0.1:8000/`
- or `http://127.0.0.1:8000/maze_benchmark_blog.html`

## Notes

- This repository is designed to be easy to extend with additional providers or evaluation rules.
- The benchmark does not attempt to verify internal reasoning traces.
- Weak performance on this task should be interpreted narrowly: it is evidence about this visual maze task, not a general ranking of model quality.
