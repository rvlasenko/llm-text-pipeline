import pytest
from pydantic import ValidationError

from llm_text_pipeline.schemas import (
    ClassificationResult,
    GeneratedAnswer,
    Sentiment,
    TextAnalysisResult,
    TextCategory,
)


def test_classification_result_has_no_final_answer_field() -> None:
    assert "final_answer" not in ClassificationResult.model_fields


def test_classification_result_accepts_valid_payload() -> None:
    result = ClassificationResult.model_validate(
        {
            "summary": "User cannot log in.",
            "category": "support",
            "sentiment": "negative",
            "key_points": ["one", "two", "three"],
        }
    )
    assert result.category is TextCategory.SUPPORT
    assert result.sentiment is Sentiment.NEGATIVE


@pytest.mark.parametrize("key_points", [["only", "two"], ["a", "b", "c", "d"], []])
def test_classification_result_rejects_wrong_key_points_count(
    key_points: list[str],
) -> None:
    with pytest.raises(ValidationError):
        ClassificationResult.model_validate(
            {
                "summary": "x",
                "category": "support",
                "sentiment": "neutral",
                "key_points": key_points,
            }
        )


def test_classification_result_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult.model_validate(
            {
                "summary": "x",
                "category": "pricing",
                "sentiment": "neutral",
                "key_points": ["a", "b", "c"],
            }
        )


def test_generated_answer_requires_final_answer() -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswer.model_validate({})

    answer = GeneratedAnswer.model_validate({"final_answer": "Done."})
    assert answer.final_answer == "Done."


def test_text_analysis_result_requires_final_answer() -> None:
    assert "final_answer" in TextAnalysisResult.model_fields
