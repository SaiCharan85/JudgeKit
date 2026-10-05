import pytest

from helpers import ALL_YES, respond
from judgekit import (
    CaseResult,
    Estimate,
    HumanLabel,
    Rubric,
    SuiteResult,
    Verdict,
    agreement,
    bootstrap,
    cohen_kappa,
    compare,
    kappa_ci,
    score,
    suite_estimates,
    unavailable,
    verdicts_by_case,
    vote_threshold_tradeoff,
)

# ---------------------------------------------------------------- kappa


def test_kappa_hand_computed() -> None:
    # observed 3/4; chance = .5*.25 + .5*.75 = .5 -> (0.75 - 0.5) / 0.5
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_kappa_perfect_and_opposite() -> None:
    assert cohen_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)
    assert cohen_kappa([True, False], [False, True]) == pytest.approx(-1.0)


def test_kappa_multiclass_labels() -> None:
    assert cohen_kappa(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_kappa_undefined_cases() -> None:
    assert cohen_kappa([], []) is None
    assert cohen_kappa([True, True], [True, True]) is None  # chance agreement already 1


def test_kappa_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([1, 0], [1])


# ---------------------------------------------------------------- bootstrap


def mean(u: list[float]) -> float | None:
    return sum(u) / len(u) if u else None


def test_bootstrap_is_seeded_and_brackets_the_value() -> None:
    data = [0.0, 1.0] * 20
    a = bootstrap(data, mean, n_boot=500, seed=1)
    assert a == bootstrap(data, mean, n_boot=500, seed=1)
    assert a.value == pytest.approx(0.5)
    assert a.low is not None and a.high is not None
    assert a.low < 0.5 < a.high and a.n == 40


def test_bootstrap_constant_data_has_zero_width() -> None:
    e = bootstrap([1.0] * 10, mean, n_boot=100)
    assert (e.value, e.low, e.high) == (1.0, 1.0, 1.0)


def test_bootstrap_empty_and_undefined() -> None:
    e = bootstrap([], mean)
    assert (e.value, e.low, e.high, e.n) == (None, None, None, 0)


@pytest.mark.parametrize(("n_boot", "alpha"), [(0, 0.05), (10, 0.0), (10, 1.0)])
def test_bootstrap_validates_arguments(n_boot: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        bootstrap([1.0], mean, n_boot=n_boot, alpha=alpha)


def test_estimate_excludes() -> None:
    assert Estimate(value=0.2, low=0.1, high=0.3, n=5).excludes(0.0)
    assert Estimate(value=-0.2, low=-0.3, high=-0.1, n=5).excludes(0.0)
    assert not Estimate(value=0.0, low=-0.1, high=0.1, n=5).excludes(0.0)
    assert not Estimate(value=None, low=None, high=None, n=0).excludes()


# ---------------------------------------------------------------- human agreement


def _v(rubric: Rubric, **answers: str) -> Verdict:
    base = {"crit": "yes", "maj": "yes", "mino": "yes", **answers}
    return score(rubric, respond(**base))


def test_agreement_confusion_and_rates(rubric: Rubric) -> None:
    verdicts = {
        "a": _v(rubric),  # judge pass
        "b": _v(rubric, crit="no"),  # judge fail
        "c": _v(rubric, crit="no"),  # judge fail
        "d": _v(rubric),  # judge pass
        "e": unavailable(rubric, "j", "down"),
    }
    labels = [
        HumanLabel(case_id="a", passed=True),
        HumanLabel(case_id="b", passed=False),
        HumanLabel(case_id="c", passed=True),  # false alarm
        HumanLabel(case_id="d", passed=False),  # miss
        HumanLabel(case_id="e", passed=True),
        HumanLabel(case_id="z", passed=True),  # no verdict
    ]
    ag = agreement(verdicts, labels)
    assert (ag.n, ag.unavailable, ag.missing) == (4, 1, 1)
    assert (ag.both_fail, ag.judge_only_fail, ag.human_only_fail, ag.both_pass) == (1, 1, 1, 1)
    assert ag.raw_agreement == pytest.approx(0.5)
    assert ag.kappa == pytest.approx(0.0)
    assert ag.judge_precision == pytest.approx(0.5) and ag.judge_recall == pytest.approx(0.5)


def test_item_level_kappa_uses_only_labeled_items(rubric: Rubric) -> None:
    verdicts = {"a": _v(rubric, crit="no"), "b": _v(rubric), "c": _v(rubric)}
    labels = [
        HumanLabel(case_id="a", passed=False, items={"crit": False}),
        HumanLabel(case_id="b", passed=True, items={"crit": True, "maj": True}),
        HumanLabel(case_id="c", passed=True),
    ]
    ag = agreement(verdicts, labels)
    assert ag.item_n == {"crit": 2, "maj": 1}
    assert ag.item_kappa["crit"] == pytest.approx(1.0)
    assert ag.item_kappa["maj"] is None  # one label, one value: undefined


def test_agreement_with_no_usable_pairs(rubric: Rubric) -> None:
    ag = agreement({}, [HumanLabel(case_id="x", passed=True)])
    assert ag.n == 0 and ag.raw_agreement is None and ag.kappa is None
    assert ag.judge_precision is None and ag.judge_recall is None


def test_duplicate_labels_rejected(rubric: Rubric) -> None:
    with pytest.raises(ValueError):
        agreement({}, [HumanLabel(case_id="x", passed=True)] * 2)


def test_kappa_ci(rubric: Rubric) -> None:
    verdicts = {f"c{i}": _v(rubric, crit="no" if i % 2 else "yes") for i in range(40)}
    labels = [HumanLabel(case_id=f"c{i}", passed=i % 2 == 0) for i in range(40)]
    e = kappa_ci(verdicts, labels, n_boot=200)
    assert e.value == pytest.approx(1.0) and e.n == 40


# ---------------------------------------------------------------- suite estimates


def _case(
    rubric: Rubric, sample: str, etype: str | None, flagged: bool, by: str = "crit"
) -> CaseResult:
    verdict = _v(rubric, **{by: "no"}) if flagged else _v(rubric)
    return CaseResult(
        case_id=sample if etype is None else f"{sample}::{etype}",
        sample_id=sample,
        error_type=etype,
        expected_items=frozenset() if etype is None else frozenset({"crit"}),
        verdict=verdict,
    )


def _suite(rubric: Rubric, cases: list[CaseResult], judge: str = "j") -> SuiteResult:
    return SuiteResult(rubric="r", rubric_version="1", judge_id=judge, seed=0, cases=tuple(cases))


def test_suite_estimates_match_point_rates(rubric: Rubric) -> None:
    cases = []
    for i in range(20):
        s = f"s{i}"
        cases += [
            _case(rubric, s, None, flagged=i < 2),  # 10% false alarms
            _case(rubric, s, "typo", flagged=i < 15),  # 75% caught
            _case(rubric, s, "gap", flagged=i < 10, by="maj"),  # 50%, never by expected item
        ]
    res = _suite(rubric, cases)
    est = suite_estimates(res, n_boot=300)
    assert est["false_alarm"].value == pytest.approx(0.10)
    assert est["catch:typo"].value == pytest.approx(0.75)
    assert est["catch:gap"].value == pytest.approx(0.50)
    assert est["catch_expected:gap"].value == pytest.approx(0.0)
    assert est["catch"].value == pytest.approx(0.625)
    assert est["catch"].n == 20  # resampled by sample, not by case
    for e in est.values():
        assert e.low is not None and e.high is not None and 0.0 <= e.low <= e.high <= 1.0
    assert est["catch:typo"].value == res.by_type()[1].rate


def test_suite_estimates_exclude_unavailable(rubric: Rubric) -> None:
    down = CaseResult(case_id="s9::typo", sample_id="s9", error_type="typo",
                      expected_items=frozenset({"crit"}),
                      verdict=unavailable(rubric, "j", "down"))  # fmt: skip
    res = _suite(rubric, [_case(rubric, "s1", "typo", True), down])
    assert suite_estimates(res, n_boot=50)["catch"].value == pytest.approx(1.0)


def test_unknown_metric_rejected(rubric: Rubric) -> None:
    res = _suite(rubric, [_case(rubric, "s1", "typo", True)])
    with pytest.raises(ValueError, match="unknown metric"):
        compare(res, res, "recall")


# ---------------------------------------------------------------- paired comparison


def test_compare_detects_a_real_gap(rubric: Rubric) -> None:
    weak = _suite(rubric, [_case(rubric, f"s{i}", "typo", i % 2 == 0) for i in range(40)])
    strong = _suite(rubric, [_case(rubric, f"s{i}", "typo", True) for i in range(40)])
    d = compare(weak, strong, "catch", n_boot=500)
    assert d.value == pytest.approx(0.5)
    assert d.excludes(0.0)


def test_compare_identical_runs_is_zero(rubric: Rubric) -> None:
    res = _suite(rubric, [_case(rubric, f"s{i}", "typo", i % 3 == 0) for i in range(30)])
    d = compare(res, res, "catch", n_boot=200)
    assert (d.value, d.low, d.high) == (0.0, 0.0, 0.0)


def test_compare_uses_only_cases_both_judged(rubric: Rubric) -> None:
    a = _suite(rubric, [_case(rubric, "s1", "typo", True), _case(rubric, "s2", "typo", False)])
    b_cases = [
        _case(rubric, "s1", "typo", True),
        CaseResult(case_id="s2::typo", sample_id="s2", error_type="typo",
                   expected_items=frozenset({"crit"}), verdict=unavailable(rubric, "j", "x")),
    ]  # fmt: skip
    d = compare(a, _suite(rubric, b_cases), "catch", n_boot=50)
    assert d.value == pytest.approx(0.0) and d.n == 1


def test_compare_rejects_mismatched_samples(rubric: Rubric) -> None:
    a = _suite(rubric, [_case(rubric, "s1", None, False)])
    moved = CaseResult(case_id="s1", sample_id="other", error_type=None, verdict=_v(rubric))
    with pytest.raises(ValueError, match="different samples"):
        compare(a, _suite(rubric, [moved]), "false_alarm")


def test_verdicts_by_case(rubric: Rubric) -> None:
    res = _suite(rubric, [_case(rubric, "s1", None, False), _case(rubric, "s1", "typo", True)])
    assert set(verdicts_by_case(res)) == {"s1", "s1::typo"}


# ---------------------------------------------------------------- vote thresholds


def _panel_case(
    rubric: Rubric, sample: str, etype: str | None, member_flags: list[bool]
) -> CaseResult:
    members = tuple(_v(rubric, crit="no") if f else _v(rubric) for f in member_flags)
    verdict = score(rubric, ALL_YES).model_copy(update={"members": members, "judge_id": "p"})
    return CaseResult(
        case_id=sample if etype is None else f"{sample}::{etype}",
        sample_id=sample,
        error_type=etype,
        expected_items=frozenset() if etype is None else frozenset({"crit"}),
        verdict=verdict,
    )


def test_threshold_tradeoff_is_monotone(rubric: Rubric) -> None:
    cases = [
        _panel_case(rubric, "s1", "typo", [True, True, True]),
        _panel_case(rubric, "s2", "typo", [True, True, False]),
        _panel_case(rubric, "s3", "typo", [True, False, False]),
        _panel_case(rubric, "s1", None, [True, False, False]),
        _panel_case(rubric, "s2", None, [False, False, False]),
    ]
    rows, skipped = vote_threshold_tradeoff(_suite(rubric, cases), rubric)
    assert skipped == 0
    assert [r.k for r in rows] == [1, 2, 3]
    assert [r.catch_rate for r in rows] == pytest.approx([1.0, 2 / 3, 1 / 3])
    assert [r.false_alarm_rate for r in rows] == pytest.approx([0.5, 0.0, 0.0])
    assert rows[0].catch_expected_rate == pytest.approx(1.0)


def test_threshold_skips_incomplete_panels(rubric: Rubric) -> None:
    full = _panel_case(rubric, "s1", "typo", [True, True, True])
    early = _panel_case(rubric, "s2", "typo", [True, True])  # stopped early
    with_down = _panel_case(rubric, "s3", "typo", [True, True, True])
    members = (*with_down.verdict.members[:2], unavailable(rubric, "j2", "down"))
    with_down = with_down.model_copy(
        update={"verdict": with_down.verdict.model_copy(update={"members": members})}
    )
    rows, skipped = vote_threshold_tradeoff(_suite(rubric, [full, early, with_down]), rubric)
    assert skipped == 2 and rows[0].n_planted == 1


def test_threshold_needs_member_verdicts(rubric: Rubric) -> None:
    with pytest.raises(ValueError, match="no panel member verdicts"):
        vote_threshold_tradeoff(_suite(rubric, [_case(rubric, "s1", "typo", True)]), rubric)


def test_minor_items_do_not_count_toward_thresholds(rubric: Rubric) -> None:
    members = tuple(_v(rubric, mino="no") for _ in range(2))
    verdict = score(rubric, ALL_YES).model_copy(update={"members": members})
    case = CaseResult(case_id="s1", sample_id="s1", error_type=None, verdict=verdict)
    rows, _ = vote_threshold_tradeoff(_suite(rubric, [case]), rubric)
    assert all(r.false_alarm_rate == 0.0 for r in rows)
