from types import SimpleNamespace

import pytest

from llm_text_pipeline.llm_client import LLMClient


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self._response = response

    def create(self, **kwargs: object) -> object:
        return self._response


def _client_with_response(monkeypatch: pytest.MonkeyPatch, response: object) -> LLMClient:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = LLMClient()
    # Replace the real OpenAI client so no network call happens.
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(response))
    )
    return client


def test_init_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing"):
        LLMClient()


def test_generate_completion_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    message = SimpleNamespace(content="hello")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = _client_with_response(monkeypatch, response)

    assert client.generate_completion(messages=[]) == "hello"


def test_generate_completion_raises_on_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(choices=[])
    client = _client_with_response(monkeypatch, response)

    with pytest.raises(ValueError, match="empty choices list"):
        client.generate_completion(messages=[])


def test_generate_completion_raises_on_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = SimpleNamespace(content="")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = _client_with_response(monkeypatch, response)

    with pytest.raises(ValueError, match="empty message content"):
        client.generate_completion(messages=[])
