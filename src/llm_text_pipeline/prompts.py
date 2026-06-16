from openai.types.chat import ChatCompletionMessageParam

from llm_text_pipeline.schemas import MeaningResult, Sentiment, TextCategory

CATEGORY_DEFINITION: dict[TextCategory, str] = {
    TextCategory.SUPPORT: "user asks for help with an account, feature, access issue, setup, troubleshooting, or product usage",
    TextCategory.FEEDBACK: "user shares an opinion, suggestion, praise, criticism, or product experience without asking for direct help",
    TextCategory.COMPLAINT: "user expresses dissatisfaction, frustration, refund demand, billing issue, broken expectation, or serious negative experience",
    TextCategory.SALES: "user asks about pricing, plans, discounts, purchasing, upgrades, product fit, or a buying decision",
    TextCategory.GENERAL_QUESTION: "user asks for information, explanation, policy details, or how something works",
}


def _bullet_list(values: list[str]) -> str:
    return "\n".join(f"  - {value}" for value in values)


def _category_options() -> str:
    return " | ".join(category.value for category in TextCategory)


def _category_definitions_block() -> str:
    return "\n".join(
        f"- {category.value}: {CATEGORY_DEFINITION[category]}"
        for category in TextCategory
    )


def _meaning_block(meaning: MeaningResult) -> str:
    return (
        f"- core_intent: {meaning.core_intent}\n"
        f"- implicit_need: {meaning.implicit_need}\n"
        f"- facts:\n{_bullet_list(meaning.facts)}"
    )


JSON_ONLY = """Return only valid JSON.
Do not use markdown.
Do not add explanations."""


MEANING_SYSTEM_PROMPT = f"""You extract the underlying meaning of a user's message.
{JSON_ONLY}
"""

MEANING_USER_TEMPLATE = """Read the text and return JSON with this structure:

{{
  "core_intent": "...",
  "facts": ["...", "..."],
  "implicit_need": "..."
}}

Requirements:
- Return only valid JSON.
- Do not use markdown.
- core_intent: one sentence describing what the user actually wants.
- facts: concrete facts explicitly stated in the text (may be empty list).
- implicit_need: what the user needs but did not say directly.

Text:
{text}
"""


CLASSIFICATION_SYSTEM_PROMPT = f"""You are a careful text classification assistant.
{JSON_ONLY}
"""

CLASSIFICATION_USER_TEMPLATE = """Analyze this text and return JSON with this structure:

{{
  "summary": "...",
  "category": "{category_options}",
  "sentiment": "positive | negative | neutral",
  "key_points": ["...", "...", "..."]
}}

Requirements:
- Return only valid JSON.
- Do not use markdown.
- category must be one of:
{category_bullets}
- sentiment must be one of:
{sentiment_bullets}
- key_points must contain exactly 3 items.
- summary must be maximum 2 sentences.

Category definitions:
{category_definitions}

Extracted meaning (context from a previous step):
{meaning_context}

Text:
{text}
"""


GENERATION_SYSTEM_PROMPT = """You write the final reply to the user's message.
Return only valid JSON with this structure: {"final_answer": "..."}.
Do not use markdown.
Do not add explanations.
"""

GENERATION_USER_TEMPLATE = """The user's message was classified as: {category}.

Write the final reply following this style:
{style}

Extracted meaning (context from a previous step):
{meaning_context}

Requirements:
- Return only valid JSON: {{"final_answer": "..."}}.
- final_answer must be maximum 2 sentences.
- Answer the user's message directly.

User's message:
{text}
"""


SELF_CHECK_SYSTEM_PROMPT = f"""You verify a drafted reply against the original message.
{JSON_ONLY}
"""

SELF_CHECK_USER_TEMPLATE = """Check the proposed answer against the original message.

Return JSON with this structure:

{{
  "contradicts_input": true | false,
  "missing_details": ["...", "..."],
  "notes": "..."
}}

Requirements:
- Return only valid JSON.
- Do not use markdown.
- contradicts_input: true if the answer states anything that conflicts with the original message.
- missing_details: important points from the reference list that the answer fails to address (may be empty).
- notes: one short sentence explaining the judgement.

Original message:
{text}

Proposed answer:
{final_answer}

Reference points that should not be lost:
{reference_points}
"""


def _messages(system: str, user_content: str) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_meaning_messages(
    text: str,
) -> list[ChatCompletionMessageParam]:
    return _messages(
        MEANING_SYSTEM_PROMPT,
        MEANING_USER_TEMPLATE.format(text=text),
    )


def build_classification_messages(
    text: str,
    meaning: MeaningResult,
) -> list[ChatCompletionMessageParam]:
    return _messages(
        CLASSIFICATION_SYSTEM_PROMPT,
        CLASSIFICATION_USER_TEMPLATE.format(
            category_options=_category_options(),
            category_bullets=_bullet_list(
                [category.value for category in TextCategory]
            ),
            sentiment_bullets=_bullet_list(
                [sentiment.value for sentiment in Sentiment]
            ),
            category_definitions=_category_definitions_block(),
            meaning_context=_meaning_block(meaning),
            text=text,
        ),
    )


def build_generation_messages(
    text: str,
    category: TextCategory,
    style: str,
    meaning: MeaningResult,
) -> list[ChatCompletionMessageParam]:
    return _messages(
        GENERATION_SYSTEM_PROMPT,
        GENERATION_USER_TEMPLATE.format(
            category=category.value,
            style=style,
            meaning_context=_meaning_block(meaning),
            text=text,
        ),
    )


def build_self_check_messages(
    text: str,
    final_answer: str,
    reference_points: list[str],
) -> list[ChatCompletionMessageParam]:
    return _messages(
        SELF_CHECK_SYSTEM_PROMPT,
        SELF_CHECK_USER_TEMPLATE.format(
            text=text,
            final_answer=final_answer,
            reference_points=_bullet_list(reference_points),
        ),
    )
