import pytest

from helpers import ALL_YES, respond
from judgekit import (
    CallInfo,
    Judge,
    JudgeResponse,
    LLMCallError,
    LLMJudge,
    Rubric,
    StructuredOutputError,
    TextStructuredLLM,
    render_system,
)
from judgekit.testing import ScriptedLLM, ScriptedTextLLM

# ---------------------------------------------------------------- prompt


def test_system_prompt_lists_every_item_with_guidance(rubric: Rubric) -> None:
    s = render_system(rubric)
    assert s.startswith("Review the reasoning, not the outcome.")
    for item in rubric.items:
        assert f"- {item.id}: {item.question}" in s
    assert "(An invented date counts as no.)" in s


def test_system_prompt_hides_severity(rubric: Rubric) -> None:
    s = render_system(rubric).lower()
    assert "critical" not in s and "minor" not in s and "severity" not in s


def test_system_prompt_is_static_and_case_goes_in_user_message(rubric: Rubric) -> None:
    llm = ScriptedLLM([ALL_YES, ALL_YES])
    judge = LLMJudge(llm, rubric)
    judge.evaluate("case one")
    judge.evaluate("case two")
    (s1, u1, _), (s2, u2, _) = llm.calls
    assert s1 == s2  # identical prefix -> provider prompt caching
    assert (u1, u2) == ("case one", "case two")


# ---------------------------------------------------------------- verdicts


def test_passing_verdict_carries_calls_and_id(rubric: Rubric) -> None:
    info = CallInfo(model="m", family="f", input_tokens=100, output_tokens=20)
    v = LLMJudge(ScriptedLLM([ALL_YES], info), rubric, name="j1").evaluate("case")
    assert v.passed and v.available
    assert v.judge_id == "j1"
    assert v.calls == (info,)


def test_critical_no_fails(rubric: Rubric) -> None:
    v = LLMJudge(ScriptedLLM([respond(crit="no", maj="yes", mino="yes")]), rubric).evaluate("c")
    assert not v.passed and v.available


def test_case_dependent_script(rubric: Rubric) -> None:
    def grade(system: str, user: str) -> JudgeResponse:
        return respond(crit="no" if "invented" in user else "yes", maj="yes", mino="yes")

    judge = LLMJudge(ScriptedLLM([grade, grade]), rubric)
    assert judge.evaluate("a fine explanation").passed
    assert not judge.evaluate("an invented fact").passed


def test_avoid_families_reaches_the_llm(rubric: Rubric) -> None:
    llm = ScriptedLLM([ALL_YES])
    LLMJudge(llm, rubric).evaluate("c", avoid_families=frozenset({"writer_family"}))
    assert llm.calls[0][2] == frozenset({"writer_family"})


def test_llm_judge_satisfies_protocol(rubric: Rubric) -> None:
    assert isinstance(LLMJudge(ScriptedLLM([]), rubric), Judge)


# ---------------------------------------------------------------- failure paths (fail safe)


def test_call_failure_gives_unavailable_failed_verdict(rubric: Rubric) -> None:
    v = LLMJudge(ScriptedLLM([LLMCallError("quota exhausted")]), rubric).evaluate("c")
    assert not v.passed and not v.available
    assert "quota exhausted" in v.error
    assert all(not i.answered for i in v.items)


def test_invalid_output_gives_unavailable_with_calls(rubric: Rubric) -> None:
    calls = (CallInfo(model="m"), CallInfo(model="m", attempts=2))
    llm = ScriptedLLM([StructuredOutputError("bad json", raw="{", calls=calls)])
    v = LLMJudge(llm, rubric).evaluate("c")
    assert not v.available and "invalid judge output" in v.error
    assert v.calls == calls


def test_bugs_are_not_swallowed(rubric: Rubric) -> None:
    with pytest.raises(RuntimeError):
        LLMJudge(ScriptedLLM([RuntimeError("bug")]), rubric).evaluate("c")


def test_empty_case_rejected(rubric: Rubric) -> None:
    with pytest.raises(ValueError):
        LLMJudge(ScriptedLLM([ALL_YES]), rubric).evaluate("   ")


def test_exhausted_script_is_a_call_error(rubric: Rubric) -> None:
    v = LLMJudge(ScriptedLLM([]), rubric).evaluate("c")
    assert not v.available and "script exhausted" in v.error


# ---------------------------------------------------------------- end to end with a text client


def test_text_client_end_to_end_with_retry(rubric: Rubric) -> None:
    good = (
        '```json\n{"answers": [{"id": "crit", "answer": "yes"}, '
        '{"id": "maj", "answer": "no", "note": "skipped a point"}, '
        '{"id": "mino", "answer": "yes"}]}\n```'
    )
    text = ScriptedTextLLM(["I think it is fine.", good])  # no JSON first -> retried
    judge = LLMJudge(TextStructuredLLM(text), rubric)
    v = judge.evaluate("case")
    assert not v.passed and v.available
    assert v.result("maj").note == "skipped a point"
    assert len(v.calls) == 2
