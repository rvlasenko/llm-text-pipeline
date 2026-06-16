import json

import pytest
from openai.types.chat import ChatCompletionMessageParam

from llm_text_pipeline.pipeline import process_text
from llm_text_pipeline.routing import get_answer_style
from llm_text_pipeline.schemas import (
    PipelineResult,
    SelfCheckVerdict,
    Sentiment,
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


def _meaning_json(core_intent: str = "User cannot access premium features.") -> str:
    return json.dumps(
        {
            "core_intent": core_intent,
            "facts": ["Paid yesterday.", "Received receipt."],
            "implicit_need": "Wants access restored.",
        }
    )


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


def _self_check_json(
    contradicts_input: bool = False,
    missing_details: list[str] | None = None,
    notes: str = "Looks consistent.",
) -> str:
    return json.dumps(
        {
            "contradicts_input": contradicts_input,
            "missing_details": missing_details if missing_details is not None else [],
            "notes": notes,
        }
    )


def _full_script(
    category: str = "support",
    final_answer: str = "Reset it.",
    self_check: str | Exception | None = None,
) -> list[str | Exception]:
    return [
        _meaning_json(),
        _classification_json(category=category),
        _generation_json(final_answer),
        self_check if self_check is not None else _self_check_json(),
    ]


def _user_content(messages: list[ChatCompletionMessageParam]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def test_returns_pipeline_result_with_user_result_and_trace() -> None:
    client = ScriptedLLMClient(
        _full_script(category="support", final_answer="Reset it.")
    )

    result = process_text(text="I cannot log in.", client=client)

    assert isinstance(result, PipelineResult)
    assert result.result.category is TextCategory.SUPPORT
    assert result.result.sentiment is Sentiment.NEGATIVE
    assert result.result.final_answer == "Reset it."
    assert len(result.result.key_points) == 3
    assert result.trace.self_check.verdict is SelfCheckVerdict.PASS
    assert result.trace.meaning.core_intent
    assert len(client.calls) == 4


def test_chain_order_is_meaning_classify_generate_self_check() -> None:
    client = ScriptedLLMClient(_full_script())

    process_text(text="I cannot log in.", client=client)

    assert len(client.calls) == 4
    assert "core_intent" in _user_content(client.calls[0])
    assert "category" in _user_content(client.calls[1])
    assert "final_answer" in _user_content(client.calls[2])
    assert "contradicts_input" in _user_content(client.calls[3])


def test_meaning_is_passed_into_classification_prompt() -> None:
    client = ScriptedLLMClient(
        [
            _meaning_json(core_intent="downstream-intent-marker"),
            _classification_json(),
            _generation_json(),
            _self_check_json(),
        ]
    )

    process_text(text="some text", client=client)

    assert "downstream-intent-marker" in _user_content(client.calls[1])
    assert "downstream-intent-marker" in _user_content(client.calls[2])


def test_category_drives_generation_style() -> None:
    client = ScriptedLLMClient(_full_script(category="complaint"))

    process_text(text="This is unacceptable.", client=client)

    assert get_answer_style(TextCategory.COMPLAINT) in _user_content(client.calls[2])


def test_self_check_failure_degrades_to_unknown_without_crashing() -> None:
    client = ScriptedLLMClient(
        _full_script(self_check=RuntimeError("judge network down"))
    )

    result = process_text(text="I need help.", client=client)

    assert result.result.final_answer
    assert result.trace.self_check.verdict is SelfCheckVerdict.UNKNOWN
    assert result.trace.self_check.contradicts_input is None
    assert "judge network down" in result.trace.self_check.notes


def test_self_check_invalid_json_degrades_to_unknown() -> None:
    client = ScriptedLLMClient(_full_script(self_check="not valid json"))

    result = process_text(text="I need help.", client=client)

    assert result.trace.self_check.verdict is SelfCheckVerdict.UNKNOWN


def test_self_check_contradiction_yields_fail_verdict() -> None:
    client = ScriptedLLMClient(
        _full_script(self_check=_self_check_json(contradicts_input=True))
    )

    result = process_text(text="I need help.", client=client)

    assert result.trace.self_check.verdict is SelfCheckVerdict.FAIL


def test_self_check_missing_details_yields_warn_verdict() -> None:
    client = ScriptedLLMClient(
        _full_script(self_check=_self_check_json(missing_details=["lost a point"]))
    )

    result = process_text(text="I need help.", client=client)

    assert result.trace.self_check.verdict is SelfCheckVerdict.WARN


def test_empty_input_is_rejected_before_any_llm_call() -> None:
    client = ScriptedLLMClient([])

    with pytest.raises(ValueError, match="must not be empty"):
        process_text(text="   \n  ", client=client)

    assert client.calls == []


def test_meaning_invalid_json_retries_then_raises() -> None:
    client = ScriptedLLMClient(["this is not json", "still not json"])

    with pytest.raises(ValueError, match="failed after retry"):
        process_text(text="Hello", client=client)

    assert len(client.calls) == 2


def test_classification_invalid_category_retries_then_raises() -> None:
    client = ScriptedLLMClient(
        [
            _meaning_json(),
            _classification_json(category="pricing"),
            _classification_json(category="pricing"),
        ]
    )

    with pytest.raises(ValueError, match="failed after retry"):
        process_text(text="How much is it?", client=client)

    assert len(client.calls) == 3


def test_generation_failure_propagates_and_aborts_before_self_check() -> None:
    client = ScriptedLLMClient(
        [_meaning_json(), _classification_json(), RuntimeError("network down")]
    )

    with pytest.raises(RuntimeError, match="network down"):
        process_text(text="I need help.", client=client)

    assert len(client.calls) == 3


@pytest.mark.parametrize("category", [c.value for c in TextCategory])
def test_accepts_all_categories(category: str) -> None:
    client = ScriptedLLMClient(_full_script(category=category))

    result = process_text(text="Sample input", client=client)

    assert result.result.category.value == category


def test_step_recovers_on_retry() -> None:
    client = ScriptedLLMClient(
        [
            "not json yet",
            _meaning_json(),
            _classification_json(),
            _generation_json(),
            _self_check_json(),
        ]
    )

    result = process_text(text="I cannot log in.", client=client)

    assert result.result.category is TextCategory.SUPPORT
    assert len(client.calls) == 5


def test_retry_appends_corrective_nudge() -> None:
    client = ScriptedLLMClient(
        [
            "bad",
            _meaning_json(),
            _classification_json(),
            _generation_json(),
            _self_check_json(),
        ]
    )

    process_text(text="hi", client=client)

    assert len(client.calls[1]) == len(client.calls[0]) + 1
    assert client.calls[1][-1]["role"] == "user"
    assert "valid JSON" in client.calls[1][-1]["content"]


def test_too_long_answer_is_rejected_then_retried() -> None:
    long_answer = _generation_json(final_answer="x" * 700)
    client = ScriptedLLMClient(
        [_meaning_json(), _classification_json(), long_answer, long_answer]
    )

    with pytest.raises(ValueError, match="failed after retry"):
        process_text(text="I need help.", client=client)


def test_empty_response_retries_then_recovers() -> None:
    client = ScriptedLLMClient(
        [
            ValueError("LLM returned empty message content"),
            _meaning_json(),
            _classification_json(),
            _generation_json(),
            _self_check_json(),
        ]
    )

    result = process_text(text="I cannot log in.", client=client)

    assert result.result.category is TextCategory.SUPPORT
    assert len(client.calls) == 5


def test_empty_response_failing_twice_raises() -> None:
    client = ScriptedLLMClient(
        [
            ValueError("LLM returned empty message content"),
            ValueError("LLM returned empty message content"),
        ]
    )

    with pytest.raises(ValueError, match="failed after retry"):
        process_text(text="Hello", client=client)

    assert len(client.calls) == 2
