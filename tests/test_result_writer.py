import json
from pathlib import Path

from llm_text_pipeline.result_writer import save_json_result
from llm_text_pipeline.schemas import Sentiment, TextAnalysisResult, TextCategory


def _build_result(summary: str = "A short summary.") -> TextAnalysisResult:
    return TextAnalysisResult(
        summary=summary,
        category=TextCategory.SUPPORT,
        sentiment=Sentiment.NEGATIVE,
        key_points=["one", "two", "three"],
        final_answer="Please contact support.",
    )


def test_save_json_result_writes_roundtrippable_json(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"

    save_json_result(result=_build_result(), output_path=output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == _build_result().model_dump()


def test_save_json_result_creates_missing_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "dir" / "result.json"

    save_json_result(result=_build_result(), output_path=output_path)

    assert output_path.exists()


def test_save_json_result_preserves_non_ascii_characters(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"

    save_json_result(
        result=_build_result(summary="Café user is unhappy 😕"),
        output_path=output_path,
    )

    # ensure_ascii=False must keep the original glyphs, not \uXXXX escapes.
    raw = output_path.read_text(encoding="utf-8")
    assert "Café user is unhappy 😕" in raw


def test_save_json_result_overwrites_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"

    save_json_result(result=_build_result(summary="first"), output_path=output_path)
    save_json_result(result=_build_result(summary="second"), output_path=output_path)

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["summary"] == "second"
