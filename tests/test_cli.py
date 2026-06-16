import io
import json

import pytest

from llm_text_pipeline import cli
from llm_text_pipeline.schemas import (
    MeaningResult,
    PipelineResult,
    PipelineTrace,
    SelfCheckResult,
    SelfCheckVerdict,
    Sentiment,
    TextAnalysisResult,
    TextCategory,
)


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _fake_result() -> PipelineResult:
    return PipelineResult(
        result=TextAnalysisResult(
            summary="A short summary.",
            category=TextCategory.SUPPORT,
            sentiment=Sentiment.NEUTRAL,
            key_points=["one", "two", "three"],
            final_answer="Here is your answer.",
        ),
        trace=PipelineTrace(
            meaning=MeaningResult(
                core_intent="ci",
                facts=["fact"],
                implicit_need="need",
            ),
            self_check=SelfCheckResult(
                contradicts_input=False,
                missing_details=[],
                verdict=SelfCheckVerdict.PASS,
                notes="ok",
            ),
        ),
    )


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "LLMClient", lambda: object())
    monkeypatch.setattr(cli, "process_text", lambda text, client: _fake_result())


def test_resolve_prefers_text_argument() -> None:
    assert resolve("hello", None, io.StringIO("ignored")) == "hello"


def test_resolve_reads_from_file(tmp_path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("from file", encoding="utf-8")

    assert resolve(None, path, io.StringIO("ignored")) == "from file"


def test_resolve_reads_from_stdin() -> None:
    assert resolve(None, None, io.StringIO("piped text")) == "piped text"


def test_resolve_empty_text_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        resolve("   \n ", None, io.StringIO(""))


def test_resolve_no_source_on_tty_raises() -> None:
    with pytest.raises(ValueError, match="No input text"):
        resolve(None, None, _FakeTTY())


def resolve(text, file, stdin) -> str:
    return cli.resolve_input_text(text=text, file=file, stdin=stdin)


def test_main_prints_valid_json_and_exits_zero(
    stub_pipeline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["I cannot log in."])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["category"] == "support"
    assert payload["trace"]["self_check"]["verdict"] == "pass"


def test_main_passes_input_text_to_pipeline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, str] = {}

    def _spy(text: str, client: object) -> PipelineResult:
        seen["text"] = text
        return _fake_result()

    monkeypatch.setattr(cli, "LLMClient", lambda: object())
    monkeypatch.setattr(cli, "process_text", _spy)

    cli.main(["please reset my password"])

    assert seen["text"] == "please reset my password"


def test_main_writes_output_file_when_requested(
    stub_pipeline: None, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "out.json"

    exit_code = cli.main(["some text", "--output", str(output_path)])

    assert exit_code == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["result"]["final_answer"] == "Here is your answer."


def test_main_empty_input_returns_one(
    stub_pipeline: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["   "])

    assert exit_code == 1
    assert "error" in capsys.readouterr().err


def test_main_missing_api_key_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise() -> object:
        raise ValueError("OPENAI_API_KEY is missing")

    monkeypatch.setattr(cli, "LLMClient", _raise)

    exit_code = cli.main(["some text"])

    assert exit_code == 1
    assert "OPENAI_API_KEY is missing" in capsys.readouterr().err


def test_main_text_and_file_are_mutually_exclusive(tmp_path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("from file", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["inline text", "--file", str(path)])

    assert exc_info.value.code == 2


def test_main_pipeline_error_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(text: str, client: object) -> PipelineResult:
        raise RuntimeError("LLM exploded")

    monkeypatch.setattr(cli, "LLMClient", lambda: object())
    monkeypatch.setattr(cli, "process_text", _boom)

    exit_code = cli.main(["some text"])

    assert exit_code == 1
    assert "LLM exploded" in capsys.readouterr().err
