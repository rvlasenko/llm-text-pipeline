import pytest
from openai.types.chat import ChatCompletionMessageParam

from llm_text_pipeline.schemas import (
    Sentiment,
    TextAnalysisResult,
    TextCategory,
)

from llm_text_pipeline.pipeline import process_text_summary


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.received_messages: list[ChatCompletionMessageParam] | None = None

    def generate_completion(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> str:
        self.received_messages = messages
        return self.response


def test_process_text_summary_returns_valid_result() -> None:
    client = FakeLLMClient(
        response="""
{
  "summary": "The user cannot access premium features after payment.",
  "category": "support",
  "sentiment": "negative",
  "key_points": [
    "The user bought a subscription.",
    "The payment was successful.",
    "Premium features are still unavailable."
  ],
  "final_answer": "Please check your account status and try logging in again. If the issue remains, contact support with your receipt."
}
"""
    )

    result = process_text_summary(
        text="I paid but cannot access premium features.",
        client=client,
    )

    assert isinstance(result, TextAnalysisResult)
    assert result.category is TextCategory.SUPPORT
    assert result.sentiment is Sentiment.NEGATIVE
    assert len(result.key_points) == 3


def test_process_text_summary_raises_for_invalid_json() -> None:
    client = FakeLLMClient(response="This is not JSON.")

    with pytest.raises(ValueError, match="LLM returned invalid JSON"):
        process_text_summary(
            text="Hello",
            client=client,
        )


def test_process_text_summary_raises_for_missing_required_field() -> None:
    client = FakeLLMClient(
        response="""
{
  "summary": "The user needs help.",
  "category": "support",
  "key_points": [
    "Point one",
    "Point two",
    "Point three"
  ],
  "final_answer": "Please contact support."
}
"""
    )

    with pytest.raises(ValueError, match="LLM response does not match schema"):
        process_text_summary(
            text="I need help.",
            client=client,
        )


def test_process_text_summary_raises_for_wrong_key_points_type() -> None:
    client = FakeLLMClient(
        response="""
{
  "summary": "The user needs help.",
  "category": "support",
  "sentiment": "negative",
  "key_points": "This should be a list.",
  "final_answer": "Please contact support."
}
"""
    )

    with pytest.raises(ValueError, match="LLM response does not match schema"):
        process_text_summary(
            text="I need help.",
            client=client,
        )


def test_process_text_summary_raises_for_invalid_category() -> None:
    client = FakeLLMClient(
        response="""
{
  "summary": "The user asks about pricing.",
  "category": "pricing",
  "sentiment": "neutral",
  "key_points": [
    "The user asks about pricing.",
    "The user wants plan details.",
    "The user may be considering a purchase."
  ],
  "final_answer": "Please review the pricing page or contact sales."
}
"""
    )

    with pytest.raises(ValueError, match="LLM response does not match schema"):
        process_text_summary(
            text="How much does the pro plan cost?",
            client=client,
        )


@pytest.mark.parametrize(
    "key_points_json",
    [
        '["only", "two"]',
        '["one", "two", "three", "four"]',
        "[]",
    ],
)
def test_process_text_summary_raises_for_wrong_key_points_count(
    key_points_json: str,
) -> None:
    client = FakeLLMClient(
        response=f"""
{{
  "summary": "The user needs help.",
  "category": "support",
  "sentiment": "negative",
  "key_points": {key_points_json},
  "final_answer": "Please contact support."
}}
"""
    )

    with pytest.raises(ValueError, match="LLM response does not match schema"):
        process_text_summary(
            text="I need help.",
            client=client,
        )


@pytest.mark.parametrize(
    "response",
    [
        """
{
  "summary": "The user cannot access premium features after payment.",
  "category": "support",
  "sentiment": "negative",
  "key_points": [
    "The user bought a subscription.",
    "Payment was successful.",
    "Premium features are unavailable."
  ],
  "final_answer": "Please try logging in again and contact support if the issue remains."
}
""",
        """
{
  "summary": "The user is unhappy with the latest product update.",
  "category": "feedback",
  "sentiment": "negative",
  "key_points": [
    "The user dislikes the update.",
    "The experience feels worse.",
    "The user shares product feedback."
  ],
  "final_answer": "Thank you for sharing your feedback. We will use it to improve the product experience."
}
""",
        """
{
  "summary": "The user reports a billing issue and wants it resolved.",
  "category": "complaint",
  "sentiment": "negative",
  "key_points": [
    "The user was charged unexpectedly.",
    "The user is dissatisfied.",
    "The user wants the issue fixed."
  ],
  "final_answer": "I am sorry about the unexpected charge. Please contact billing support so the team can review and resolve it."
}
""",
        """
{
  "summary": "The user asks about pricing and available plans.",
  "category": "sales",
  "sentiment": "neutral",
  "key_points": [
    "The user asks about pricing.",
    "The user wants plan information.",
    "The user may be considering a purchase."
  ],
  "final_answer": "You can compare available plans on the pricing page or contact sales for help choosing the right option."
}
""",
        """
{
  "summary": "The user asks how password reset works.",
  "category": "general_question",
  "sentiment": "neutral",
  "key_points": [
    "The user asks about password reset.",
    "The user wants an explanation.",
    "No personal issue is reported."
  ],
  "final_answer": "You can reset your password using the forgot password link on the login page."
}
""",
    ],
)
def test_process_text_summary_accepts_valid_responses(response: str) -> None:
    client = FakeLLMClient(response=response)

    result = process_text_summary(
        text="Sample input",
        client=client,
    )

    assert isinstance(result, TextAnalysisResult)
    assert len(result.key_points) == 3
