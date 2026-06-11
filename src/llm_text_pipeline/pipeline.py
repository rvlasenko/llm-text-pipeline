import json
from json import JSONDecodeError

from pydantic import ValidationError

from llm_text_pipeline.llm_client import CompletionClient
from llm_text_pipeline.prompts import build_text_processing_messages
from llm_text_pipeline.schemas import TextAnalysisResult


def process_text_summary(text: str, client: CompletionClient) -> TextAnalysisResult:
    messages = build_text_processing_messages(text)
    raw_result = client.generate_completion(messages)

    try:
        parsed_result = json.loads(raw_result)

    except JSONDecodeError as error:
        raise ValueError(f"LLM returned invalid JSON: {raw_result}") from error

    try:
        return TextAnalysisResult.model_validate(parsed_result)

    except ValidationError as error:
        raise ValueError(f"LLM response does not match schema: {error}") from error
