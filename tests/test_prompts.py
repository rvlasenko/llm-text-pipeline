from llm_text_pipeline.prompts import (
    CATEGORY_DEFINITION,
    build_classification_messages,
    build_generation_messages,
    build_meaning_messages,
    build_self_check_messages,
)
from llm_text_pipeline.routing import get_answer_style
from llm_text_pipeline.schemas import MeaningResult, TextCategory


def _user_content(messages: list[dict]) -> str:
    return next(message["content"] for message in messages if message["role"] == "user")


def _meaning(core_intent: str = "user wants access restored") -> MeaningResult:
    return MeaningResult(
        core_intent=core_intent,
        facts=["paid yesterday", "got receipt"],
        implicit_need="resolve quickly",
    )


def test_category_definitions_cover_every_category() -> None:
    assert set(CATEGORY_DEFINITION.keys()) == set(TextCategory)


def test_meaning_prompt_embeds_input_and_requests_intent_fields() -> None:
    content = _user_content(build_meaning_messages("meaning-marker-77"))
    assert "meaning-marker-77" in content
    assert "core_intent" in content
    assert "implicit_need" in content


def test_meaning_prompt_does_not_request_category() -> None:
    content = _user_content(build_meaning_messages("some text"))
    assert "category" not in content


def test_classification_prompt_lists_every_category() -> None:
    content = _user_content(build_classification_messages("some text", _meaning()))
    for category in TextCategory:
        assert category.value in content


def test_classification_prompt_embeds_the_input_text() -> None:
    content = _user_content(
        build_classification_messages("unique-marker-123", _meaning())
    )
    assert "unique-marker-123" in content


def test_classification_prompt_injects_upstream_meaning() -> None:
    content = _user_content(
        build_classification_messages(
            "some text", _meaning(core_intent="intent-marker-abc")
        )
    )
    assert "intent-marker-abc" in content


def test_classification_prompt_does_not_request_final_answer() -> None:
    content = _user_content(build_classification_messages("some text", _meaning()))
    assert "final_answer" not in content


def test_generation_prompt_injects_style_and_text() -> None:
    style = get_answer_style(TextCategory.COMPLAINT)
    content = _user_content(
        build_generation_messages(
            text="my-input-text",
            category=TextCategory.COMPLAINT,
            style=style,
            meaning=_meaning(),
        )
    )
    assert style in content
    assert "my-input-text" in content


def test_generation_prompt_injects_upstream_meaning() -> None:
    content = _user_content(
        build_generation_messages(
            text="some text",
            category=TextCategory.SUPPORT,
            style=get_answer_style(TextCategory.SUPPORT),
            meaning=_meaning(core_intent="gen-intent-marker"),
        )
    )
    assert "gen-intent-marker" in content


def test_generation_prompt_differs_by_category() -> None:
    complaint = _user_content(
        build_generation_messages(
            text="same text",
            category=TextCategory.COMPLAINT,
            style=get_answer_style(TextCategory.COMPLAINT),
            meaning=_meaning(),
        )
    )
    sales = _user_content(
        build_generation_messages(
            text="same text",
            category=TextCategory.SALES,
            style=get_answer_style(TextCategory.SALES),
            meaning=_meaning(),
        )
    )
    assert complaint != sales


def test_self_check_prompt_embeds_text_answer_and_reference() -> None:
    content = _user_content(
        build_self_check_messages(
            text="original-input-xyz",
            final_answer="answer-marker-qrs",
            reference_points=["important-point-111"],
        )
    )
    assert "original-input-xyz" in content
    assert "answer-marker-qrs" in content
    assert "important-point-111" in content


def test_self_check_prompt_requests_contradiction_and_missing_details() -> None:
    content = _user_content(
        build_self_check_messages(
            text="t",
            final_answer="a",
            reference_points=["p"],
        )
    )
    assert "contradicts_input" in content
    assert "missing_details" in content
