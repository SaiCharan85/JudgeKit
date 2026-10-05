import random
from collections.abc import Callable

import pytest

from helpers import respond
from judgekit import (
    BiasCase,
    BiasProbe,
    BiasResult,
    Rubric,
    Sample,
    Verdict,
    estimate_bias_calls,
    politeness_probe,
    position_probe,
    repeat_probe,
    run_bias,
    score,
    sectioned,
    unavailable,
    verbosity_probe,
)
from judgekit.bias import DEFAULT_FILLER

BASE = sectioned(["evidence", "explanation"])
REORDERED = sectioned(["explanation", "evidence"])
SAMPLES = [
    Sample(
        id=f"s{i}",
        fields={"evidence": f"Value {i}.", "explanation": f"The value is {i}. It was checked."},
    )
    for i in range(30)
]


class RuleJudge:
    """Fails crit when `fails(case)`; records every case it sees."""

    def __init__(self, rubric: Rubric, fails: Callable[[str], bool], down: str = "") -> None:
        self.rubric = rubric
        self.fails = fails
        self.down = down
        self.seen: list[str] = []

    @property
    def judge_id(self) -> str:
        return "rule"

    def evaluate(self, case: str, *, avoid_families: frozenset[str] = frozenset()) -> Verdict:
        self.seen.append(case)
        if self.down and self.down in case:
            return unavailable(self.rubric, "rule", "down")
        crit = "no" if self.fails(case) else "yes"
        return score(self.rubric, respond(crit=crit, maj="yes", mino="yes"), "rule")


def fair(_: str) -> bool:
    return False


# ---------------------------------------------------------------- renderers and probes


def test_sectioned_renders_in_order() -> None:
    assert (
        BASE(SAMPLES[1])
        == "## EVIDENCE\nValue 1.\n\n## EXPLANATION\nThe value is 1. It was checked."
    )
    assert REORDERED(SAMPLES[1]).startswith("## EXPLANATION")


def test_sectioned_validates() -> None:
    with pytest.raises(ValueError):
        sectioned([])
    with pytest.raises(KeyError):
        sectioned(["missing"])(SAMPLES[0])


def test_verbosity_adds_only_filler() -> None:
    probe = verbosity_probe(n_sentences=2)
    assert probe.vary is not None
    out = probe.vary(SAMPLES[3], random.Random(0))
    assert out is not None
    text = out.fields["explanation"]
    assert "The value is 3." in text and "It was checked." in text
    assert sum(f in text for f in DEFAULT_FILLER) == 2
    assert out.fields["evidence"] == SAMPLES[3].fields["evidence"]


def test_verbosity_keeps_question_and_exclamation_sentences() -> None:
    probe = verbosity_probe(n_sentences=1)
    s = Sample(id="q", fields={"explanation": "Is it due? Yes! Paid."})
    assert probe.vary is not None
    out = probe.vary(s, random.Random(1))
    assert out is not None
    for part in ("Is it due?", "Yes!", "Paid."):
        assert part in out.fields["explanation"]


@pytest.mark.parametrize("n", [0, len(DEFAULT_FILLER) + 1])
def test_verbosity_validates(n: int) -> None:
    with pytest.raises(ValueError):
        verbosity_probe(n_sentences=n)


def test_politeness_wraps_text() -> None:
    probe = politeness_probe(openers=["Hello there."], closers=["Kind regards."])
    assert probe.vary is not None
    out = probe.vary(SAMPLES[0], random.Random(0))
    assert out is not None
    assert out.fields["explanation"] == "Hello there. The value is 0. It was checked. Kind regards."


def test_politeness_validates() -> None:
    with pytest.raises(ValueError):
        politeness_probe(openers=[], closers=[])


def test_probes_skip_missing_or_empty_fields() -> None:
    empty = Sample(id="e", fields={"evidence": "x", "explanation": "  "})
    for probe in (verbosity_probe(), politeness_probe()):
        assert probe.vary is not None
        assert probe.vary(empty, random.Random(0)) is None


# ---------------------------------------------------------------- detecting bias


def test_fair_judge_shows_no_bias(rubric: Rubric) -> None:
    probes = [position_probe(REORDERED), verbosity_probe(), politeness_probe()]
    res = run_bias(RuleJudge(rubric, fair), SAMPLES, probes, BASE)
    for p in ("position", "verbosity", "politeness"):
        s = res.stats(p)
        assert s.comparable == 30 and s.flips == 0 and s.flip_rate == 0.0


def test_position_bias_detected(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, lambda case: case.startswith("## EXPLANATION"))
    res = run_bias(judge, SAMPLES, [position_probe(REORDERED)], BASE)
    s = res.stats("position")
    assert s.flip_rate == 1.0 and s.net_shift == -1.0  # reordered cases judged more strictly
    assert res.item_flip_counts("position") == {"crit": 30}


def test_verbosity_bias_detected_as_lenient_shift(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, lambda case: len(case) < 120)  # short = fail, long = pass
    res = run_bias(judge, SAMPLES, [verbosity_probe()], BASE)
    s = res.stats("verbosity")
    assert s.to_lenient == 30 and s.to_strict == 0 and s.net_shift == 1.0


def test_politeness_bias_with_cis(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, lambda case: "care" not in case and "s1" not in case)
    probe = politeness_probe(openers=["We reviewed this with great care."], closers=[])
    res = run_bias(judge, SAMPLES, [probe], BASE)
    est = res.estimates(n_boot=300)["politeness"]
    assert est["net_shift"].value == pytest.approx(1.0)
    assert est["net_shift"].excludes(0.0)


# ---------------------------------------------------------------- frugality and bookkeeping


def test_baseline_judged_once_per_sample_and_shared(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, fair)
    probes = [position_probe(REORDERED), verbosity_probe(), politeness_probe()]
    run_bias(judge, SAMPLES[:5], probes, BASE)
    assert len(judge.seen) == estimate_bias_calls(5, 3) == 20


def test_reused_baseline_is_not_judged_again(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, fair)
    prior = {s.id: score(rubric, respond(crit="yes", maj="yes", mino="yes")) for s in SAMPLES[:4]}
    res = run_bias(judge, SAMPLES[:4], [verbosity_probe()], BASE, baseline=prior)
    assert len(judge.seen) == estimate_bias_calls(4, 1, baseline_reused=True) == 4
    assert res.cases[0].baseline == prior["s0"]


def test_estimate_calls_scales_with_panel_size() -> None:
    assert estimate_bias_calls(10, 2, calls_per_verdict=3) == 90


def test_not_applicable_counted_and_baseline_skipped(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, fair)
    empty = [Sample(id="e", fields={"evidence": "x", "explanation": "  "})]
    res = run_bias(judge, empty, [verbosity_probe()], BASE)
    assert res.cases == () and res.not_applicable == {"verbosity": 1}
    assert judge.seen == []  # nothing to compare -> no baseline call either


def test_identical_variant_is_not_applicable_except_repeat(rubric: Rubric) -> None:
    same_order = position_probe(BASE, name="same_order")
    judge = RuleJudge(rubric, fair)
    res = run_bias(judge, SAMPLES[:3], [same_order, repeat_probe()], BASE)
    assert res.not_applicable == {"same_order": 3, "repeat": 0}
    assert res.stats("repeat").comparable == 3


def test_repeat_measures_noise(rubric: Rubric) -> None:
    calls = {"n": 0}

    def flaky(_: str) -> bool:
        calls["n"] += 1
        return calls["n"] % 2 == 0  # alternates: maximal noise

    res = run_bias(RuleJudge(rubric, flaky), SAMPLES[:10], [repeat_probe()], BASE)
    assert res.stats("repeat").flip_rate == 1.0


def test_unavailable_verdicts_are_not_compared(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, fair, down="Value 0.")
    res = run_bias(judge, SAMPLES[:3], [verbosity_probe()], BASE)
    s = res.stats("verbosity")
    assert (s.n, s.comparable, s.flips) == (3, 2, 0)
    assert res.cases[0].item_flips == frozenset()


def test_rates_none_when_nothing_comparable(rubric: Rubric) -> None:
    res = run_bias(RuleJudge(rubric, fair, down="EVIDENCE"), SAMPLES[:2], [repeat_probe()], BASE)
    s = res.stats("repeat")
    assert s.flip_rate is None and s.net_shift is None
    assert "n/a" in res.to_markdown(n_boot=50)


def test_variants_are_seeded(rubric: Rubric) -> None:
    def run(seed: int) -> list[str]:
        judge = RuleJudge(rubric, fair)
        run_bias(judge, SAMPLES[:5], [verbosity_probe()], BASE, seed=seed)
        return judge.seen

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_resume_reuses_saved_cases(rubric: Rubric) -> None:
    probes = [verbosity_probe(), politeness_probe()]
    saved: list[BiasCase] = []
    run_bias(RuleJudge(rubric, fair), SAMPLES[:4], probes, BASE, on_case=saved.append)
    partial = saved[:3]  # died after s0 (2 probes) and s1's first probe
    judge = RuleJudge(rubric, fair)
    res = run_bias(judge, SAMPLES[:4], probes, BASE, resume=partial)
    # s0: nothing new; s1: one probe (baseline reused from the saved case); s2, s3: 1 + 2 each
    assert len(judge.seen) == 1 + 3 + 3
    assert len(res.cases) == 8


def test_validations(rubric: Rubric) -> None:
    judge = RuleJudge(rubric, fair)
    with pytest.raises(ValueError, match="probe names"):
        run_bias(judge, SAMPLES, [repeat_probe(), repeat_probe()], BASE)
    with pytest.raises(ValueError, match="sample ids"):
        run_bias(judge, [SAMPLES[0], SAMPLES[0]], [repeat_probe()], BASE)


def test_markdown_and_json_round_trip(rubric: Rubric) -> None:
    res = run_bias(RuleJudge(rubric, fair), SAMPLES[:5], [verbosity_probe()], BASE)
    md = res.to_markdown(n_boot=50)
    assert "| verbosity | 5 | 0 |" in md
    assert BiasResult.model_validate_json(res.model_dump_json()) == res


def test_custom_probe_with_both_vary_and_render(rubric: Rubric) -> None:
    def shout(s: Sample, r: random.Random) -> Sample | None:
        return s.updated({"explanation": s.fields["explanation"].upper()})

    probe = BiasProbe(name="caps_reordered", vary=shout, render=REORDERED)
    judge = RuleJudge(rubric, fair)
    run_bias(judge, SAMPLES[:1], [probe], BASE)
    assert judge.seen[1].startswith("## EXPLANATION\nTHE VALUE IS 0.")
