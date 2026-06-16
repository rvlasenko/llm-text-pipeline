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

## Use it (CLI)

After `uv pip install -e .` the `llm-pipeline` command is available. It takes one
text and prints the full structured result as JSON.

```bash
llm-pipeline "I was charged twice and nobody answered my emails. I want a refund."
llm-pipeline --file message.txt
echo "How do I reset my password?" | llm-pipeline
llm-pipeline "..." --output result.json   # also write the JSON to a file
```

Input comes from (in order of precedence): the positional argument, `--file`, or
piped stdin. Exit codes: `0` success, `1` input or pipeline error (message on
stderr), `2` invalid arguments. JSON goes to stdout, logs to stderr, so the output
can be piped into tools like `jq`.

### Example

Input:

> I was charged twice this month and nobody answered my emails for a whole week.
> This is unacceptable and I want a refund.

Output:

```json
{
  "result": {
    "summary": "The user was charged twice this month and did not receive any response to their emails for a week. They are requesting a refund due to this issue.",
    "category": "complaint",
    "sentiment": "negative",
    "key_points": [
      "Charged twice this month",
      "No response to emails for a whole week",
      "Requesting a refund"
    ],
    "final_answer": "I'm very sorry for the double charge and the delay in responding to your emails; I understand how frustrating this must be. I will escalate your case immediately to ensure a prompt refund and a thorough review of your billing issue."
  },
  "trace": {
    "meaning": {
      "core_intent": "The user wants a refund for being charged twice this month.",
      "facts": [
        "The user was charged twice this month.",
        "Nobody answered the user's emails for a whole week."
      ],
      "implicit_need": "The user needs prompt customer support and resolution of the billing issue."
    },
    "self_check": {
      "contradicts_input": false,
      "missing_details": [],
      "verdict": "pass",
      "notes": "The answer addresses all key points from the original message appropriately."
    }
  }
}
```

## Run

```bash
.venv/bin/python main.py
```

Runs the demo corpus from `examples/sample_inputs.py` through the pipeline, writes
one JSON per sample to `outputs/`, and prints `expected` vs `predicted` category
plus a final match count.

## Test

`pytest` lives in the `dev` dependency group, which the plain editable install does
not pull in. Install it once, then run the suite:

```bash
uv pip install -e . --group dev
.venv/bin/python -m pytest
```

Tests run offline with scripted fake clients - no API key and no network needed.
