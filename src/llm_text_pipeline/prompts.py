from openai.types.chat import (
    ChatCompletionMessageParam,
)


def build_text_processing_messages(
    text: str,
) -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful text processing assistant. "
                "Return ONLY valid JSON. Do not use markdown. Do not add explanations."
            ),
        },
        {
            "role": "user",
            "content": f"""
Analyze this text and return JSON with this structure:

{{
  "summary": "...",
  "key_points": ["...", "...", "..."],
  "helpful_response": "..."
}}

Text:
{text}
""",
        },
    ]
