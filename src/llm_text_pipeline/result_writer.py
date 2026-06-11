import json
from pathlib import Path

from llm_text_pipeline.schemas import TextAnalysisResult


def save_json_result(
    result: TextAnalysisResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
