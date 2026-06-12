# llm-text-pipeline

Classifies an incoming text into a category and generates a reply whose tone
depends on that category. The category-to-style routing lives in plain Python
code, not inside a single large prompt, so it is explicit and unit-testable.

## How it works

Two stages with routing in between:

1. **Classify** (LLM call) -> `summary`, `category`, `sentiment`, three `key_points`.
2. **Route** (`routing.py`) -> pick the answer style for the category.
3. **Generate** (LLM call) -> the final reply written in that style.

Categories: `support`, `feedback`, `complaint`, `sales`, `general_question`.
Each maps to its own answer style (e.g. complaint -> empathetic, sales -> short
and persuasive, support -> structured technical steps).

## Setup

```bash
uv venv
uv pip install -e .
cp .env.example .env   # then put your OPENAI_API_KEY in .env
```

Environment variables (`.env`):

- `OPENAI_API_KEY` - required.
- `OPENAI_MODEL` - optional, defaults to `gpt-4.1-mini`.

The editable install (`-e .`) is what makes `llm_text_pipeline` importable; without
it `python main.py` fails with `ModuleNotFoundError: No module named 'llm_text_pipeline'`.

## Run

```bash
.venv/bin/python main.py
```

Runs the demo corpus from `examples/sample_inputs.py` through the pipeline, writes
one JSON per sample to `outputs/`, and prints `expected` vs `predicted` category
plus a final match count.

## Test

```bash
.venv/bin/python -m pytest
```

Tests run offline with scripted fake clients - no API key and no network needed.
