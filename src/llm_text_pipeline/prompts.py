from openai.types.chat import ChatCompletionMessageParam

from llm_text_pipeline.schemas import Sentiment, TextCategory

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


CLASSIFICATION_SYSTEM_PROMPT = """You are a careful text classification assistant.
Return only valid JSON.
Do not use markdown.
Do not add explanations.
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

Requirements:
- Return only valid JSON: {{"final_answer": "..."}}.
- final_answer must be maximum 2 sentences.
- Answer the user's message directly.

User's message:
{text}
"""


def build_classification_messages(
    text: str,
) -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": CLASSIFICATION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": CLASSIFICATION_USER_TEMPLATE.format(
                category_options=_category_options(),
                category_bullets=_bullet_list(
                    [category.value for category in TextCategory]
                ),
                sentiment_bullets=_bullet_list(
                    [sentiment.value for sentiment in Sentiment]
                ),
                category_definitions=_category_definitions_block(),
                text=text,
            ),
        },
    ]


def build_generation_messages(
    text: str,
    category: TextCategory,
    style: str,
) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERATION_USER_TEMPLATE.format(
                category=category.value,
                style=style,
                text=text,
            ),
        },
    ]
