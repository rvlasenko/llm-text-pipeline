from enum import StrEnum

from pydantic import BaseModel, Field


class TextCategory(StrEnum):
    SUPPORT = "support"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    SALES = "sales"
    GENERAL_QUESTION = "general_question"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class TextAnalysisResult(BaseModel):
    summary: str
    category: TextCategory
    sentiment: Sentiment

    key_points: list[str] = Field(
        min_length=3,
        max_length=3,
    )

    final_answer: str
