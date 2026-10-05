import pytest
from pydantic import ValidationError

from judgekit import (
    Answer,
    Judge,
    JudgeResponse,
    Rubric,
    Severity,
    Verdict,
    parse_rubric,
    score,
    unavailable,
)
from judgekit.judge import NOTE_MAX


def _resp(**answers: str) -> JudgeResponse:
    return JudgeResponse(answers=[Answer(id=k, answer=v) for k, v in answers.items()])  # type: ignore[arg-type]


# ---------------------------------------------------------------- aggregation


def test_all_yes_passes(rubric: Rubric) -> None:
    v = score(rubric, _resp(crit="yes", maj="yes", mino="yes"), judge_id="m1")
    assert v.passed
    assert v.failures == ()
    assert v.judge_id == "m1"
    assert (v.rubric, v.rubric_version) == ("r", "1")


@pytest.mark.parametrize(("failing", "passed"), [("crit", False), ("maj", False), ("mino", True)])
def test_one_no_fails_only_on_fail_on_tiers(rubric: Rubric, failing: str, passed: bool) -> None:
    answers = {"crit": "yes", "maj": "yes", "mino": "yes", failing: "no"}
    v = score(rubric, _resp(**answers))
    assert v.passed is passed
    assert [f.item_id for f in v.failures] == [failing]


def test_minor_failure_is_recorded_even_when_verdict_passes(rubric: Rubric) -> None:
    v = score(rubric, _resp(crit="yes", maj="yes", mino="no"))
    assert v.passed
    assert [f.item_id for f in v.failures_at(Severity.MINOR)] == ["mino"]
    assert v.failures_at(Severity.CRITICAL) == ()


def test_custom_fail_on_changes_outcome() -> None:
    lenient = parse_rubric(
        {
            "name": "r",
            "version": "1",
            "instructions": "x",
            "fail_on": ["critical"],
            "items": [{"id": "maj", "question": "m?"}],
        }
    )
    assert score(lenient, _resp(maj="no")).passed


# ---------------------------------------------------------------- strictness edge cases


def test_unanswered_item_fails(rubric: Rubric) -> None:
    v = score(rubric, _resp(crit="yes", mino="yes"))
    assert not v.passed
    r = v.result("maj")
    assert (r.passed, r.answered) == (False, False)
    assert "not answered" in r.note


def test_unanswered_minor_item_does_not_fail_verdict(rubric: Rubric) -> None:
    v = score(rubric, _resp(crit="yes", maj="yes"))
    assert v.passed
    assert not v.result("mino").answered


def test_empty_response_fails(rubric: Rubric) -> None:
    v = score(rubric, JudgeResponse(answers=[]))
    assert not v.passed
    assert len(v.failures) == 3


def test_conflicting_duplicate_answers_fail(rubric: Rubric) -> None:
    resp = JudgeResponse(
        answers=[
            Answer(id="crit", answer="yes"),
            Answer(id="crit", answer="no", note="date invented"),
            Answer(id="maj", answer="yes"),
            Answer(id="mino", answer="yes"),
        ]
    )
    r = score(rubric, resp).result("crit")
    assert not r.passed and r.answered
    assert r.note == "conflicting answers; date invented"


def test_agreeing_duplicates_are_fine_and_notes_deduplicated(rubric: Rubric) -> None:
    resp = JudgeResponse(
        answers=[
            Answer(id="maj", answer="no", note="skipped a point"),
            Answer(id="maj", answer="no", note="skipped a point"),
            Answer(id="crit", answer="yes"),
            Answer(id="mino", answer="yes"),
        ]
    )
    r = score(rubric, resp).result("maj")
    assert r.answered and not r.passed
    assert r.note == "skipped a point"


def test_unexpected_ids_are_reported_not_scored(rubric: Rubric) -> None:
    v = score(rubric, _resp(crit="yes", maj="yes", mino="yes", extra="no", another="no"))
    assert v.passed
    assert v.unexpected_ids == ("another", "extra")
    assert [i.item_id for i in v.items] == ["crit", "maj", "mino"]


def test_results_follow_rubric_order(rubric: Rubric) -> None:
    v = score(rubric, _resp(mino="yes", crit="yes", maj="yes"))
    assert [i.item_id for i in v.items] == list(rubric.ids)


def test_result_lookup_unknown_id(rubric: Rubric) -> None:
    with pytest.raises(KeyError):
        score(rubric, _resp()).result("missing")


# ---------------------------------------------------------------- response schema


def test_answer_rejects_anything_but_yes_no() -> None:
    with pytest.raises(ValidationError):
        Answer(id="a", answer="maybe")  # type: ignore[arg-type]


def test_note_length_capped() -> None:
    Answer(id="a", answer="no", note="x" * NOTE_MAX)
    with pytest.raises(ValidationError):
        Answer(id="a", answer="no", note="x" * (NOTE_MAX + 1))


def test_response_parses_from_model_json() -> None:
    raw = '{"answers": [{"id": "crit", "answer": "no", "note": "invented"}]}'
    resp = JudgeResponse.model_validate_json(raw)
    assert resp.answers[0].note == "invented"


def test_verdict_is_immutable(rubric: Rubric) -> None:
    v = score(rubric, _resp(crit="yes", maj="yes", mino="yes"))
    with pytest.raises(ValidationError):
        v.passed = False  # type: ignore[misc]


# ---------------------------------------------------------------- Judge protocol


class _StubJudge:
    def __init__(self, rubric: Rubric) -> None:
        self.rubric = rubric

    @property
    def judge_id(self) -> str:
        return "stub"

    def evaluate(self, case: str, *, avoid_families: frozenset[str] = frozenset()) -> Verdict:
        answer = "no" if "invented" in case else "yes"
        return score(self.rubric, _resp(crit=answer, maj="yes", mino="yes"), self.judge_id)


def test_stub_satisfies_judge_protocol(rubric: Rubric) -> None:
    judge: Judge = _StubJudge(rubric)
    assert isinstance(judge, Judge)
    assert judge.evaluate("a fine explanation").passed
    assert not judge.evaluate("an invented fact").passed


def test_non_judge_is_rejected_by_protocol_check() -> None:
    assert not isinstance(object(), Judge)


# ---------------------------------------------------------------- unavailable verdicts


def test_unavailable_verdict_fails_every_item(rubric: Rubric) -> None:
    v = unavailable(rubric, "j", "timeout")
    assert not v.passed and not v.available
    assert v.error == "timeout"
    assert all(not i.answered and i.note == "timeout" for i in v.items)


def test_unavailable_with_blank_error_still_has_a_reason(rubric: Rubric) -> None:
    v = unavailable(rubric, "j", "")
    assert v.error == "judge unavailable" and not v.available
    assert v.items[0].note == "judge unavailable"


def test_scored_verdict_is_available(rubric: Rubric) -> None:
    assert score(rubric, _resp(crit="yes", maj="yes", mino="yes")).available
