"""The `Judge` interface and verdicts.

A judge grades the REASONING behind a decision, never the decision itself. The LLM only answers
yes/no per rubric item (`JudgeResponse`); code turns that into a `Verdict` (`score`), so pass/fail
follows the rubric's severity tiers and can never be talked into by the model.

Strict by design: an item the judge did not answer, or answered both ways, counts as a failure.
A judge that could not run at all returns an `unavailable` verdict: failed, with `error` set, so
callers can tell "the judge said no" from "there was no judgment" (fail safe, never silent).
"""

from collections import defaultdict
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from judgekit.llm import CallInfo
from judgekit.rubric import Rubric, Severity

NOTE_MAX = 500


class Answer(BaseModel):
    """One rubric answer as the judge model returns it."""

    id: str
    answer: Literal["yes", "no"]
    note: str = Field(default="", max_length=NOTE_MAX, description="why, if no")


class JudgeResponse(BaseModel):
    """The structured output a judge model is asked for."""

    answers: list[Answer]


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    severity: Severity
    passed: bool
    answered: bool
    note: str = ""


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    rubric: str
    rubric_version: str
    items: tuple[ItemResult, ...]
    passed: bool
    judge_id: str = ""  # which judge produced it (model name, panel name, ...)
    unexpected_ids: tuple[str, ...] = ()  # answers for ids the rubric does not have (ignored)
    error: str = ""  # set when the judge could not run; the verdict then fails
    calls: tuple[CallInfo, ...] = ()  # LLM calls behind this verdict (audit, cost)

    @property
    def available(self) -> bool:
        return not self.error

    @property
    def failures(self) -> tuple[ItemResult, ...]:
        return tuple(i for i in self.items if not i.passed)

    def failures_at(self, severity: Severity) -> tuple[ItemResult, ...]:
        return tuple(i for i in self.failures if i.severity == severity)

    def result(self, item_id: str) -> ItemResult:
        for i in self.items:
            if i.item_id == item_id:
                return i
        raise KeyError(f"verdict has no item {item_id!r}")


def score(rubric: Rubric, response: JudgeResponse, judge_id: str = "") -> Verdict:
    """Turn the judge model's raw answers into a verdict under the rubric's severity rules."""
    by_id: dict[str, list[Answer]] = defaultdict(list)
    for a in response.answers:
        by_id[a.id].append(a)
    results = []
    for item in rubric.items:
        answers = by_id.get(item.id, [])
        values = {a.answer for a in answers}
        notes = "; ".join(dict.fromkeys(a.note for a in answers if a.note))
        if not answers:
            passed, answered, note = False, False, "not answered by the judge"
        elif len(values) > 1:
            passed, answered = False, True
            note = "conflicting answers" + (f"; {notes}" if notes else "")
        else:
            passed, answered, note = values == {"yes"}, True, notes
        results.append(ItemResult(item_id=item.id, severity=item.severity, passed=passed,
                                  answered=answered, note=note))  # fmt: skip
    passed = not any(not r.passed and rubric.fails_verdict(r.severity) for r in results)
    unexpected = tuple(sorted(set(by_id) - set(rubric.ids)))
    return Verdict(
        rubric=rubric.name,
        rubric_version=rubric.version,
        items=tuple(results),
        passed=passed,
        judge_id=judge_id,
        unexpected_ids=unexpected,
    )


def unavailable(
    rubric: Rubric, judge_id: str, error: str, calls: tuple[CallInfo, ...] = ()
) -> Verdict:
    """The verdict of a judge that could not run: every item unanswered, failed, `error` set."""
    error = error or "judge unavailable"
    items = tuple(
        ItemResult(item_id=i.id, severity=i.severity, passed=False, answered=False, note=error)
        for i in rubric.items
    )
    return Verdict(rubric=rubric.name, rubric_version=rubric.version, items=items, passed=False,
                   judge_id=judge_id, error=error, calls=calls)  # fmt: skip


@runtime_checkable
class Judge(Protocol):
    """Anything that grades a rendered case (evidence + decision + explanation) into a verdict.

    Implementations: an LLM judge, a multi-judge panel, or a test double.
    """

    @property
    def judge_id(self) -> str: ...

    def evaluate(self, case: str, *, avoid_families: frozenset[str] = frozenset()) -> Verdict:
        """Grade `case`; `avoid_families` keeps the judge off the model family it grades."""
        ...
