"""Fixed prompts used across all benchmark runs."""

BENCHMARK_PROMPT_NAME = "maze_visual_reasoning_v1"

BENCHMARK_PROMPT = """You are solving a visual maze puzzle from the image only.

Important rules:
- Base your answer only on what you can see in the image.
- Do not use any external tools, code, search, calculators, or graph-search programs.
- Do not assume anything that is not visually present.
- Only use 4-directional movement: up, down, left, right.
- No diagonal moves.
- Wall tiles are blocked.
- Walkable tiles can be traversed.
- Trap tiles, if present, are dangerous and must be avoided; they are not traversable.
- The start tile is the player character.
- The goal tile is the treasure.
- The shortest path length is the number of moves, not the number of visited cells.
- If the image is ambiguous, say so explicitly instead of guessing.

First visually identify the grid and the special tiles, then solve the maze.

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
- If start or goal cannot be identified with confidence, set confidence to "low".
- If reachable is false, set "shortest_path_length" to null and "path" to [].
- If reachable is true, provide one valid shortest path.
- Output JSON only, with no extra text.
"""
