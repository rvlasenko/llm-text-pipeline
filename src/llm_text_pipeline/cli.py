import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

from llm_text_pipeline.llm_client import LLMClient
from llm_text_pipeline.pipeline import process_text
from llm_text_pipeline.result_writer import save_json_result

logger = logging.getLogger(__name__)


def resolve_input_text(
    text: str | None,
    file: Path | None,
    stdin: TextIO,
) -> str:
    if text is not None:
        source = text
    elif file is not None:
        source = file.read_text(encoding="utf-8")
    elif not stdin.isatty():
        source = stdin.read()
    else:
        raise ValueError(
            "No input text. Pass TEXT as an argument, use --file, or pipe text via stdin."
        )

    if not source.strip():
        raise ValueError("Input text must not be empty")

    return source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-pipeline",
        description="Classify a text and generate a styled reply as structured JSON.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "text",
        nargs="?",
        help="Text to process. Omit to read from stdin.",
    )
    source.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Read input text from this file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Also write the JSON result to this file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level pipeline logs on stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    try:
        text = resolve_input_text(args.text, args.file, sys.stdin)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        client = LLMClient()
        result = process_text(text=text, client=client)
    except Exception as error:
        logger.debug("pipeline failed", exc_info=True)
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    if args.output is not None:
        save_json_result(result=result, output_path=args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
