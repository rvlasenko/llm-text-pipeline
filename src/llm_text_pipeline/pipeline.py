import json
from typing import Any
from json import JSONDecodeError

from src.llm_text_pipeline.llm_client import LLMClient
from src.llm_text_pipeline.prompts import build_text_processing_messages


def process_text_summary(text: str, client: LLMClient) -> dict[str, Any]:
    prompt = build_text_processing_messages(text)
    raw_result = client.generate_completion(prompt)

    try:
        return json.loads(raw_result)

    except JSONDecodeError as error:
        raise ValueError(f"LLM returned invalid JSON: {raw_result}") from error
