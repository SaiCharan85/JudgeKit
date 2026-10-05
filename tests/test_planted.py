import random

import pytest

from helpers import respond
from judgekit import (
    CaseResult,
    ErrorType,
    LLMCallError,
    Mutation,
    Rubric,
    Sample,
    Verdict,
    plant,
    run_suite,
    score,
    unavailable,
)
from judgekit.mutators import insert_sentence, number_swap

SAMPLES = [
    Sample(id=f"s{i}", fields={"evidence": f"Amount {100 * (i + 1)}.",
                               "explanation": f"Pay {100 * (i + 1)}. All facts checked."})
    for i in range(6)
]  # fmt: skip
WRONG_AMOUNT = ErrorType("wrong_amount", frozenset({"crit"}), number_swap(factors=(2.0,)))
FABRICATED = ErrorType("fabricated", frozenset({"crit"}), insert_sentence(["A witness agreed."]))


def render(sample: Sample) -> str:
    return f"EVIDENCE: {sample.fields['evidence']}\nEXPLANATION: {sample.fields['explanation']}"


class KeywordJudge:
    """Flags crit when the explanation's amount differs from the evidence or a witness appears;
    flags mino on every case containing 'checked' (a noisy minor item)."""

    def __init__(
        self, rubric: Rubric, catch_witness: bool = True, down: frozenset[str] = frozenset()
    ):
        self.rubric = rubric
        self.catch_witness = catch_witness
        self.down = down
        self.seen: list[str] = []

    @property
    def judge_id(self) -> str:
        return "keyword"

    def evaluate(self, case: str, *, avoid_families: frozenset[str] = frozenset()) -> Verdict:
        self.seen.append(case)
        if any(d in case for d in self.down):
            return unavailable(self.rubric, self.judge_id, "down")
        ev = case.split("\n")[0].split()[-1].rstrip(".")
        bad = f"Pay {ev}." not in case or (self.catch_witness and "witness" in case)
        return score(self.rubric, respond(crit="no" if bad else "yes", maj="yes", mino="yes"))


# ---------------------------------------------------------------- planting


def test_plant_makes_one_copy_per_sample_and_type() -> None:
    suite = plant(SAMPLES, [WRONG_AMOUNT, FABRICATED], seed=1)
    assert suite.counts() == {"wrong_amount": 6, "fabricated": 6}
    assert len(suite.clean) == 6
    assert suite.n_cases == 18
    p = suite.planted[0]
    assert p.case_id == f"{p.sample_id}::{p.error_type}"
    assert p.sample.fields != next(s for s in SAMPLES if s.id == p.sample_id).fields


def test_plant_is_deterministic_and_seed_sensitive() -> None:
    a = plant(SAMPLES, [WRONG_AMOUNT, FABRICATED], seed=3)
    b = plant(SAMPLES, [WRONG_AMOUNT, FABRICATED], seed=3)
    c = plant(SAMPLES, [WRONG_AMOUNT, FABRICATED], seed=4)
    assert a == b
    assert a.planted != c.planted


def test_max_per_type_and_max_clean_for_pilots() -> None:
    suite = plant(SAMPLES, [WRONG_AMOUNT], seed=0, max_per_type=2, max_clean=3)
    assert suite.counts() == {"wrong_amount": 2}
    assert len(suite.clean) == 3
    ids = [s.id for s in suite.clean]
    assert ids == sorted(ids, key=lambda x: int(x[1:]))  # original order kept


def test_not_applicable_counted() -> None:
    no_numbers = [Sample(id="t", fields={"evidence": "x", "explanation": "No digits."})]
    suite = plant(no_numbers, [WRONG_AMOUNT])
    assert suite.planted == ()
    assert suite.not_applicable == {"wrong_amount": 1}


def test_mutation_that_changes_nothing_is_skipped() -> None:
    noop = ErrorType(
        "noop",
        frozenset({"crit"}),
        lambda smp, r: Mutation(changes=dict(smp.fields), description="same"),
    )
    suite = plant(SAMPLES[:2], [noop])
    assert suite.planted == () and suite.not_applicable == {"noop": 2}


def test_unknown_field_in_mutation_is_an_error() -> None:
    bad = ErrorType(
        "bad", frozenset({"crit"}), lambda smp, r: Mutation(changes={"nope": "x"}, description="x")
    )
    with pytest.raises(KeyError):
        plant(SAMPLES[:1], [bad])


def test_plant_validations() -> None:
    with pytest.raises(ValueError, match="sample ids"):
        plant([SAMPLES[0], SAMPLES[0]], [WRONG_AMOUNT])
    with pytest.raises(ValueError, match="names must be unique"):
        plant(SAMPLES, [WRONG_AMOUNT, WRONG_AMOUNT])
    with pytest.raises(ValueError, match="no expected rubric items"):
        plant(SAMPLES, [ErrorType("x", frozenset(), number_swap())])


def test_mutators_get_independent_seeded_rngs() -> None:
    seen: list[float] = []

    def spy(smp: Sample, r: random.Random) -> Mutation | None:
        seen.append(r.random())
        return None

    plant(SAMPLES[:2], [ErrorType("spy", frozenset({"crit"}), spy)], seed=5)
    first = list(seen)
    seen.clear()
    plant(SAMPLES[:2], [ErrorType("spy", frozenset({"crit"}), spy)], seed=5)
    assert seen == first and first[0] != first[1]


# ---------------------------------------------------------------- running and stats


def test_run_suite_rates(rubric: Rubric) -> None:
    suite = plant(SAMPLES, [WRONG_AMOUNT, FABRICATED], seed=0)
    res = run_suite(KeywordJudge(rubric, catch_witness=False), suite, render, rubric)
    stats = {t.error_type: t for t in res.by_type()}
    assert stats["wrong_amount"].rate == 1.0
    assert stats["wrong_amount"].expected_rate == 1.0
    assert stats["fabricated"].rate == 0.0
    assert res.overall_catch().rate == 0.5
    assert res.clean_stats().rate == 0.0
    assert res.judge_id == "keyword" and res.seed == 0


def test_unavailable_verdicts_are_excluded_not_counted_as_catches(rubric: Rubric) -> None:
    suite = plant(SAMPLES, [FABRICATED], seed=0)
    judge = KeywordJudge(rubric, down=frozenset({"Amount 100.", "Amount 200."}))
    res = run_suite(judge, suite, render, rubric)
    fab = res.by_type()[0]
    assert (fab.n, fab.unavailable, fab.judged) == (6, 2, 4)
    assert fab.rate == 1.0  # 4 of 4 judged, not 6 of 6 and not 4 of 6
    clean = res.clean_stats()
    assert (clean.unavailable, clean.flagged, clean.rate) == (2, 0, 0.0)


def test_rates_are_none_when_nothing_judged(rubric: Rubric) -> None:
    suite = plant(SAMPLES[:1], [FABRICATED])
    res = run_suite(KeywordJudge(rubric, down=frozenset({"EVIDENCE"})), suite, render, rubric)
    assert res.clean_stats().rate is None and res.by_type()[0].expected_rate is None
    assert "n/a" in res.to_markdown()


def test_caught_for_the_wrong_reason_is_separated(rubric: Rubric) -> None:
    wrong_reason = ErrorType("wrong_amount", frozenset({"maj"}), number_swap(factors=(2.0,)))
    suite = plant(SAMPLES[:3], [wrong_reason])
    stats = run_suite(KeywordJudge(rubric), suite, render, rubric).by_type()[0]
    assert stats.rate == 1.0 and stats.expected_rate == 0.0


def test_expected_items_must_exist_in_rubric(rubric: Rubric) -> None:
    typo = ErrorType("wrong_amount", frozenset({"crti"}), number_swap())
    suite = plant(SAMPLES[:1], [typo])
    with pytest.raises(ValueError, match="crti"):
        run_suite(KeywordJudge(rubric), suite, render, rubric)


def test_item_flags_and_markdown(rubric: Rubric) -> None:
    suite = plant(SAMPLES, [WRONG_AMOUNT], seed=0)
    res = run_suite(KeywordJudge(rubric), suite, render, rubric)
    assert res.item_flags() == {"clean": {}, "wrong_amount": {"crit": 6}}
    md = res.to_markdown()
    assert "| wrong_amount | 6 | 6 | 100.0% | 100.0% |" in md
    assert "False alarms on clean samples: 0.0%" in md


# ---------------------------------------------------------------- progress and resume


def test_on_case_called_for_each_new_case(rubric: Rubric) -> None:
    suite = plant(SAMPLES[:2], [WRONG_AMOUNT])
    seen: list[CaseResult] = []
    run_suite(KeywordJudge(rubric), suite, render, rubric, on_case=seen.append)
    assert [c.case_id for c in seen] == ["s0", "s1", *[p.case_id for p in suite.planted]]


def test_resume_skips_saved_cases_without_calling_the_judge(rubric: Rubric) -> None:
    suite = plant(SAMPLES[:3], [WRONG_AMOUNT])
    saved: list[CaseResult] = []
    first = KeywordJudge(rubric)
    run_suite(first, suite, render, rubric, on_case=saved.append)
    partial = saved[:4]  # pretend the run died after 4 of 6 cases
    second = KeywordJudge(rubric)
    res = run_suite(second, suite, render, rubric, resume=partial)
    assert len(second.seen) == 2  # only the missing cases were judged
    assert [c.case_id for c in res.cases] == [c.case_id for c in saved]


def test_results_round_trip_through_json(rubric: Rubric) -> None:
    from judgekit import SuiteResult

    suite = plant(SAMPLES[:2], [WRONG_AMOUNT])
    res = run_suite(KeywordJudge(rubric), suite, render, rubric)
    again = SuiteResult.model_validate_json(res.model_dump_json())
    assert again == res
    assert again.by_type()[0].rate == res.by_type()[0].rate


def test_llm_errors_inside_judges_become_unavailable_cases(rubric: Rubric) -> None:
    from judgekit import LLMJudge
    from judgekit.testing import ScriptedLLM

    suite = plant(SAMPLES[:1], [WRONG_AMOUNT])
    judge = LLMJudge(ScriptedLLM([LLMCallError("quota"), LLMCallError("quota")]), rubric)
    res = run_suite(judge, suite, render, rubric)
    assert all(not c.available for c in res.cases)
    assert res.overall_catch().rate is None
