from examples.sample_inputs import SAMPLE_INPUTS
from llm_text_pipeline.schemas import TextCategory


def test_corpus_has_at_least_eight_samples() -> None:
    assert len(SAMPLE_INPUTS) >= 8


def test_corpus_covers_every_category() -> None:
    covered = {sample.expected_category for sample in SAMPLE_INPUTS}
    assert covered == set(TextCategory)


def test_sample_texts_are_non_empty() -> None:
    for sample in SAMPLE_INPUTS:
        assert sample.text.strip()
