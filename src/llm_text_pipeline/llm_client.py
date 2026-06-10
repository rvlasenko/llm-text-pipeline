import os

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
)


class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=api_key)

    def generate_completion(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        if not response.choices:
            raise ValueError("LLM returned empty choices list")

        message = response.choices[0].message

        if not message.content:
            raise ValueError("LLM returned empty message content")

        return message.content
