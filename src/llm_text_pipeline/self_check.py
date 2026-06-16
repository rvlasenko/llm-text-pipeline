from llm_text_pipeline.schemas import (
    SelfCheckJudgement,
    SelfCheckResult,
    SelfCheckVerdict,
)


def calculate_self_check_verdict(
    contradicts_input: bool,
    missing_details: list[str],
) -> SelfCheckVerdict:
    if contradicts_input:
        return SelfCheckVerdict.FAIL
    if missing_details:
        return SelfCheckVerdict.WARN
    return SelfCheckVerdict.PASS


def evaluate_self_check(judgement: SelfCheckJudgement) -> SelfCheckResult:
    verdict = calculate_self_check_verdict(
        contradicts_input=judgement.contradicts_input,
        missing_details=judgement.missing_details,
    )
    return SelfCheckResult(
        contradicts_input=judgement.contradicts_input,
        missing_details=judgement.missing_details,
        verdict=verdict,
        notes=judgement.notes,
    )


def unknown_self_check(reason: str) -> SelfCheckResult:
    return SelfCheckResult(
        contradicts_input=None,
        missing_details=[],
        verdict=SelfCheckVerdict.UNKNOWN,
        notes=reason,
    )
