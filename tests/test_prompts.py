from llm_text_pipeline.prompts import (
    CATEGORY_DEFINITION,
    build_classification_messages,
    build_generation_messages,
)
from llm_text_pipeline.routing import get_answer_style
from llm_text_pipeline.schemas import TextCategory


def _user_content(messages: list[dict]) -> str:
    return next(message["content"] for message in messages if message["role"] == "user")


def test_category_definitions_cover_every_category() -> None:
    assert set(CATEGORY_DEFINITION.keys()) == set(TextCategory)


def test_classification_prompt_lists_every_category() -> None:
    content = _user_content(build_classification_messages("some text"))
    for category in TextCategory:
        assert category.value in content


def test_classification_prompt_embeds_the_input_text() -> None:
    content = _user_content(build_classification_messages("unique-marker-123"))
    assert "unique-marker-123" in content


def test_classification_prompt_does_not_request_final_answer() -> None:
    content = _user_content(build_classification_messages("some text"))
    assert "final_answer" not in content


def test_generation_prompt_injects_style_and_text() -> None:
    style = get_answer_style(TextCategory.COMPLAINT)
    content = _user_content(
        build_generation_messages(
            text="my-input-text",
            category=TextCategory.COMPLAINT,
            style=style,
        )
    )
    assert style in content
    assert "my-input-text" in content


def test_generation_prompt_differs_by_category() -> None:
    complaint = _user_content(
        build_generation_messages(
            text="same text",
            category=TextCategory.COMPLAINT,
            style=get_answer_style(TextCategory.COMPLAINT),
        )
    )
    sales = _user_content(
        build_generation_messages(
            text="same text",
            category=TextCategory.SALES,
            style=get_answer_style(TextCategory.SALES),
        )
    )
    assert complaint != sales
