from llm_text_pipeline.schemas import SelfCheckVerdict
from llm_text_pipeline.self_check import calculate_self_check_verdict


def test_contradiction_yields_fail() -> None:
    verdict = calculate_self_check_verdict(
        contradicts_input=True,
        missing_details=[],
    )
    assert verdict is SelfCheckVerdict.FAIL


def test_contradiction_dominates_missing_details() -> None:
    verdict = calculate_self_check_verdict(
        contradicts_input=True,
        missing_details=["lost one"],
    )
    assert verdict is SelfCheckVerdict.FAIL


def test_missing_details_without_contradiction_yields_warn() -> None:
    verdict = calculate_self_check_verdict(
        contradicts_input=False,
        missing_details=["lost one"],
    )
    assert verdict is SelfCheckVerdict.WARN


def test_clean_result_yields_pass() -> None:
    verdict = calculate_self_check_verdict(
        contradicts_input=False,
        missing_details=[],
    )
    assert verdict is SelfCheckVerdict.PASS


def test_calculator_never_returns_unknown() -> None:
    for contradicts in (True, False):
        for missing in ([], ["x"]):
            verdict = calculate_self_check_verdict(
                contradicts_input=contradicts,
                missing_details=missing,
            )
            assert verdict is not SelfCheckVerdict.UNKNOWN
