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


class SelfCheckVerdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


class MeaningResult(BaseModel):
    core_intent: str
    facts: list[str]
    implicit_need: str


class ClassificationResult(BaseModel):
    summary: str
    category: TextCategory
    sentiment: Sentiment

    key_points: list[str] = Field(
        min_length=3,
        max_length=3,
    )


class GeneratedAnswer(BaseModel):
    final_answer: str = Field(min_length=1)


class TextAnalysisResult(BaseModel):
    summary: str
    category: TextCategory
    sentiment: Sentiment

    key_points: list[str] = Field(
        min_length=3,
        max_length=3,
    )

    final_answer: str

    @classmethod
    def from_parts(
        cls,
        classification: ClassificationResult,
        answer: GeneratedAnswer,
    ) -> "TextAnalysisResult":
        return cls(
            summary=classification.summary,
            category=classification.category,
            sentiment=classification.sentiment,
            key_points=classification.key_points,
            final_answer=answer.final_answer,
        )


class SelfCheckJudgement(BaseModel):
    contradicts_input: bool
    missing_details: list[str]
    notes: str


class SelfCheckResult(BaseModel):
    contradicts_input: bool | None
    missing_details: list[str]
    verdict: SelfCheckVerdict
    notes: str


class PipelineTrace(BaseModel):
    meaning: MeaningResult
    self_check: SelfCheckResult


class PipelineResult(BaseModel):
    result: TextAnalysisResult
    trace: PipelineTrace
