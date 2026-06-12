from llm_text_pipeline.routing import ANSWER_STYLE, get_answer_style
from llm_text_pipeline.schemas import TextCategory


def test_answer_style_covers_every_category() -> None:
    assert set(ANSWER_STYLE.keys()) == set(TextCategory)


def test_get_answer_style_returns_mapped_style() -> None:
    for category in TextCategory:
        assert get_answer_style(category) == ANSWER_STYLE[category]


def test_every_style_is_non_empty() -> None:
    for category in TextCategory:
        assert get_answer_style(category).strip()


def test_styles_are_distinct_per_category() -> None:
    styles = [get_answer_style(category) for category in TextCategory]
    assert len(set(styles)) == len(styles)


def test_named_categories_carry_their_intended_tone() -> None:
    assert "empathetic" in get_answer_style(TextCategory.COMPLAINT).lower()
    assert "persuasive" in get_answer_style(TextCategory.SALES).lower()
    assert "step" in get_answer_style(TextCategory.SUPPORT).lower()
