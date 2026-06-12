import json
from json import JSONDecodeError

from pydantic import ValidationError

from llm_text_pipeline.llm_client import CompletionClient
from llm_text_pipeline.prompts import (
    build_classification_messages,
    build_generation_messages,
)
from llm_text_pipeline.routing import get_answer_style
from llm_text_pipeline.schemas import (
    ClassificationResult,
    GeneratedAnswer,
    TextAnalysisResult,
)


def _decode_json(raw_result: str) -> object:
    try:
        return json.loads(raw_result)
    except JSONDecodeError as error:
        raise ValueError(f"LLM returned invalid JSON: {raw_result}") from error


def _classify(text: str, client: CompletionClient) -> ClassificationResult:
    raw_result = client.generate_completion(build_classification_messages(text))
    try:
        return ClassificationResult.model_validate(_decode_json(raw_result))
    except ValidationError as error:
        raise ValueError(f"LLM response does not match schema: {error}") from error


def _generate_answer(
    text: str,
    classification: ClassificationResult,
    client: CompletionClient,
) -> GeneratedAnswer:
    style = get_answer_style(classification.category)
    messages = build_generation_messages(text, classification.category, style)
    raw_result = client.generate_completion(messages)
    try:
        return GeneratedAnswer.model_validate(_decode_json(raw_result))
    except ValidationError as error:
        raise ValueError(f"LLM response does not match schema: {error}") from error


def process_text_summary(text: str, client: CompletionClient) -> TextAnalysisResult:
    if not text.strip():
        raise ValueError("Input text must not be empty")

    classification = _classify(text, client)
    answer = _generate_answer(text, classification, client)
    return TextAnalysisResult.from_parts(classification, answer)
