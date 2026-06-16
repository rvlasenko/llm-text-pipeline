import json
from pathlib import Path

from pydantic import BaseModel


def save_json_result(
    result: BaseModel,
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
