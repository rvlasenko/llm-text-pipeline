import json

import pytest
from openai.types.chat import ChatCompletionMessageParam

from llm_text_pipeline.pipeline import process_text_summary
from llm_text_pipeline.routing import get_answer_style
from llm_text_pipeline.schemas import (
    Sentiment,
    TextAnalysisResult,
    TextCategory,
)


class ScriptedLLMClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[list[ChatCompletionMessageParam]] = []

    def generate_completion(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("unexpected extra LLM call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _classification_json(
    category: str = "support",
    sentiment: str = "negative",
) -> str:
    return json.dumps(
        {
            "summary": "The user cannot access premium features.",
            "category": category,
            "sentiment": sentiment,
            "key_points": ["Bought subscription.", "Payment ok.", "No access."],
        }
    )


def _generation_json(final_answer: str = "Please try logging in again.") -> str:
    return json.dumps({"final_answer": final_answer})


def _user_content(messages: list[ChatCompletionMessageParam]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def test_returns_combined_result() -> None:
    client = ScriptedLLMClient(
        [_classification_json(category="support"), _generation_json("Reset it.")]
    )

    result = process_text_summary(text="I cannot log in.", client=client)

    assert isinstance(result, TextAnalysisResult)
    assert result.category is TextCategory.SUPPORT
    assert result.sentiment is Sentiment.NEGATIVE
    assert result.final_answer == "Reset it."
    assert len(result.key_points) == 3
    assert len(client.calls) == 2


def test_category_drives_generation_style() -> None:
    client = ScriptedLLMClient(
        [_classification_json(category="complaint"), _generation_json()]
    )

    process_text_summary(text="This is unacceptable.", client=client)

    generation_messages = client.calls[1]
    assert get_answer_style(TextCategory.COMPLAINT) in _user_content(generation_messages)


def test_empty_input_is_rejected_before_any_llm_call() -> None:
    client = ScriptedLLMClient([])

    with pytest.raises(ValueError, match="must not be empty"):
        process_text_summary(text="   \n  ", client=client)

    assert client.calls == []


def test_invalid_classification_json_raises() -> None:
    client = ScriptedLLMClient(["this is not json"])

    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        process_text_summary(text="Hello", client=client)


def test_classification_missing_field_raises() -> None:
    bad = json.dumps({"summary": "x", "category": "support", "key_points": ["a", "b", "c"]})
    client = ScriptedLLMClient([bad])

    with pytest.raises(ValueError, match="does not match schema"):
        process_text_summary(text="Hello", client=client)


def test_classification_invalid_category_raises() -> None:
    client = ScriptedLLMClient([_classification_json(category="pricing")])

    with pytest.raises(ValueError, match="does not match schema"):
        process_text_summary(text="How much is it?", client=client)


def test_stage_two_failure_propagates_and_writes_nothing() -> None:
    client = ScriptedLLMClient(
        [_classification_json(category="support"), RuntimeError("network down")]
    )

    with pytest.raises(RuntimeError, match="network down"):
        process_text_summary(text="I need help.", client=client)


def test_stage_two_invalid_json_raises() -> None:
    client = ScriptedLLMClient(
        [_classification_json(category="support"), "not json either"]
    )

    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        process_text_summary(text="I need help.", client=client)


@pytest.mark.parametrize(
    "category",
    [c.value for c in TextCategory],
)
def test_accepts_all_categories(category: str) -> None:
    client = ScriptedLLMClient(
        [_classification_json(category=category), _generation_json()]
    )

    result = process_text_summary(text="Sample input", client=client)

    assert result.category.value == category
