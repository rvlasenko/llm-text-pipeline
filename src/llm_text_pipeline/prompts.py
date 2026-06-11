from openai.types.chat import (
    ChatCompletionMessageParam,
)

SYSTEM_PROMPT = """You are a careful text processing assistant.
Return only valid JSON.
Do not use markdown.
Do not add explanations.
"""

USER_TEMPLATE_STRICT = """Analyze this text and return JSON with this structure:

{{
  "summary": "...",
  "category": "support | feedback | complaint | sales | general_question"
  "sentiment": "positive | negative | neutral",
  "key_points": ["...", "...", "..."],
  "final_answer": "..."
}}

Requirements:
- Return only valid JSON.
- Do not use markdown.
- category must be one of:
  - feedback
  - support
  - complaint
  - sales
  - general_question
- sentiment must be one of:
  - positive
  - negative
  - neutral
- key_points must contain exactly 3 items.
- summary must be maximum 2 sentences.
- final_answer must be maximum 2 sentences.

Category definitions:
- support: user asks for help with an account, feature, access issue, setup, troubleshooting, or product usage
- feedback: user shares an opinion, suggestion, praise, criticism, or product experience without asking for direct help
- complaint: user expresses dissatisfaction, frustration, refund demand, billing issue, broken expectation, or serious negative experience
- sales: user asks about pricing, plans, discounts, purchasing, upgrades, product fit, or buying decision
- general_question: user asks for information, explanation, policy details, or how something works


Text:
{text}
"""


def build_text_processing_messages(
    text: str,
) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE_STRICT.format(text=text)},
    ]
