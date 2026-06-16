import json
import logging
from json import JSONDecodeError

from pydantic import ValidationError

from llm_text_pipeline.llm_client import CompletionClient
from llm_text_pipeline.prompts import (
    build_classification_messages,
    build_generation_messages,
    build_meaning_messages,
    build_self_check_messages,
)
from llm_text_pipeline.routing import get_answer_style
from llm_text_pipeline.schemas import (
    ClassificationResult,
    GeneratedAnswer,
    MeaningResult,
    PipelineResult,
    PipelineTrace,
    SelfCheckJudgement,
    SelfCheckResult,
    TextAnalysisResult,
)
from llm_text_pipeline.self_check import evaluate_self_check, unknown_self_check

logger = logging.getLogger(__name__)


def _decode_json(raw_result: str) -> object:
    try:
        return json.loads(raw_result)
    except JSONDecodeError as error:
        raise ValueError(f"LLM returned invalid JSON: {raw_result}") from error


def _extract_meaning(text: str, client: CompletionClient) -> MeaningResult:
    raw_result = client.generate_completion(build_meaning_messages(text))
    try:
        return MeaningResult.model_validate(_decode_json(raw_result))
    except ValidationError as error:
        raise ValueError(f"LLM response does not match schema: {error}") from error


def _classify(
    text: str,
    meaning: MeaningResult,
    client: CompletionClient,
) -> ClassificationResult:
    raw_result = client.generate_completion(
        build_classification_messages(text, meaning)
    )
    try:
        return ClassificationResult.model_validate(_decode_json(raw_result))
    except ValidationError as error:
        raise ValueError(f"LLM response does not match schema: {error}") from error


def _generate_answer(
    text: str,
    classification: ClassificationResult,
    meaning: MeaningResult,
    client: CompletionClient,
) -> GeneratedAnswer:
    style = get_answer_style(classification.category)
    messages = build_generation_messages(
        text=text,
        category=classification.category,
        style=style,
        meaning=meaning,
    )
    raw_result = client.generate_completion(messages)
    try:
        return GeneratedAnswer.model_validate(_decode_json(raw_result))
    except ValidationError as error:
        raise ValueError(f"LLM response does not match schema: {error}") from error


def _self_check(
    text: str,
    answer: GeneratedAnswer,
    classification: ClassificationResult,
    meaning: MeaningResult,
    client: CompletionClient,
) -> SelfCheckResult:
    reference_points = classification.key_points + meaning.facts
    messages = build_self_check_messages(
        text=text,
        final_answer=answer.final_answer,
        reference_points=reference_points,
    )
    # Isolation boundary: self-check is an optional passive safety net. A valid
    # answer already exists, so any failure here (transient call error, malformed
    # or off-schema judge output) must NOT crash the pipeline; it degrades to an
    # explicit UNKNOWN verdict. Verdict assembly runs outside this try, so a bug in
    # the deterministic decision logic is not masked by this handler.
    try:
        raw_result = client.generate_completion(messages)
        judgement = SelfCheckJudgement.model_validate(_decode_json(raw_result))
    except (ValueError, ValidationError) as error:
        logger.warning("step=self_check verdict=unknown reason=invalid_output: %s", error)
        return unknown_self_check(f"Self-check produced unusable output: {error}")
    except Exception as error:  # transient/infrastructure failure of the optional step
        logger.warning(
            "step=self_check verdict=unknown reason=call_failed", exc_info=True
        )
        return unknown_self_check(f"Self-check call failed: {error}")

    return evaluate_self_check(judgement)


def process_text(text: str, client: CompletionClient) -> PipelineResult:
    if not text.strip():
        raise ValueError("Input text must not be empty")

    meaning = _extract_meaning(text, client)
    logger.info("step=extract_meaning facts=%d", len(meaning.facts))

    classification = _classify(text, meaning, client)
    logger.info(
        "step=classify category=%s sentiment=%s key_points=%d",
        classification.category.value,
        classification.sentiment.value,
        len(classification.key_points),
    )

    answer = _generate_answer(text, classification, meaning, client)
    logger.info(
        "step=generate category=%s answer_len=%d",
        classification.category.value,
        len(answer.final_answer),
    )

    self_check = _self_check(text, answer, classification, meaning, client)
    logger.info(
        "step=self_check verdict=%s contradicts=%s missing_details=%d",
        self_check.verdict.value,
        self_check.contradicts_input,
        len(self_check.missing_details),
    )

    return PipelineResult(
        result=TextAnalysisResult.from_parts(classification, answer),
        trace=PipelineTrace(meaning=meaning, self_check=self_check),
    )
