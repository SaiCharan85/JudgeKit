"""Calibration: how far can a judge be trusted?

- `cohen_kappa` / `agreement`: the judge's pass/fail against human labels, overall and per rubric
  item. Kappa corrects raw agreement for chance (0 = chance level, 1 = perfect).
- `suite_estimates`: catch and false-alarm rates of a planted-error run with bootstrap CIs.
  Resampling is by sample (a clean original and its planted copies move together).
- `compare`: paired bootstrap of the difference between two judges on the same samples
  (e.g. API judge vs local judge, single judge vs panel). A CI that excludes 0 is a real gap.
- `vote_threshold_tradeoff`: for a panel run with every member's verdict stored, catch vs
  false-alarm rate when an item fails at k of n judges (k = 1 is "any", k = n is "all").

Pure Python, seeded, no model calls. Unavailable verdicts are excluded everywhere.
"""

import random
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from judgekit.judge import Verdict
from judgekit.planted import CaseResult, SuiteResult
from judgekit.rubric import Rubric

G = TypeVar("G")
H = TypeVar("H", bound=Hashable)

DEFAULT_BOOT = 2000


class Estimate(BaseModel):
    """A point estimate with a percentile bootstrap CI; None where undefined (no data)."""

    model_config = ConfigDict(frozen=True)

    value: float | None
    low: float | None
    high: float | None
    n: int  # resampling units (samples or labeled cases)

    def excludes(self, x: float = 0.0) -> bool:
        """The CI lies entirely above or below `x` (e.g. a difference that is not noise)."""
        if self.low is None or self.high is None:
            return False
        return self.low > x or self.high < x


# ---------------------------------------------------------------- kappa


def cohen_kappa(a: Sequence[H], b: Sequence[H]) -> float | None:
    """Cohen's kappa for two raters over the same items; None when undefined (no items, or both
    raters always give the same single label, so chance agreement is already 1)."""
    if len(a) != len(b):
        raise ValueError(f"rater sequences differ in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return None
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((list(a).count(lab) / n) * (list(b).count(lab) / n) for lab in labels)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


# ---------------------------------------------------------------- bootstrap


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap(
    units: Sequence[G],
    stat: Callable[[Sequence[G]], float | None],
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap: resample `units` with replacement, recompute `stat`."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    value = stat(units)
    if value is None or not units:
        return Estimate(value=value, low=None, high=None, n=len(units))
    rng = random.Random(seed)
    k = len(units)
    draws = sorted(
        v
        for v in (stat([units[rng.randrange(k)] for _ in range(k)]) for _ in range(n_boot))
        if v is not None
    )
    if not draws:
        return Estimate(value=value, low=None, high=None, n=k)
    return Estimate(
        value=value,
        low=_quantile(draws, alpha / 2),
        high=_quantile(draws, 1 - alpha / 2),
        n=k,
    )


# ---------------------------------------------------------------- human labels


class HumanLabel(BaseModel):
    """A person's judgment of one case: overall pass/fail, optionally per rubric item."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    passed: bool
    items: dict[str, bool] = {}


class Agreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int  # labeled cases with an available verdict
    unavailable: int  # labeled cases whose verdict is unavailable (excluded)
    missing: int  # labels with no verdict at all (excluded)
    raw_agreement: float | None
    kappa: float | None
    both_fail: int
    judge_only_fail: int  # judge flagged, human passed: false alarm
    human_only_fail: int  # human flagged, judge passed: miss
    both_pass: int
    item_kappa: dict[str, float | None]
    item_n: dict[str, int]

    @property
    def judge_precision(self) -> float | None:
        flagged = self.both_fail + self.judge_only_fail
        return self.both_fail / flagged if flagged else None

    @property
    def judge_recall(self) -> float | None:
        bad = self.both_fail + self.human_only_fail
        return self.both_fail / bad if bad else None


def _paired(
    verdicts: Mapping[str, Verdict], labels: Sequence[HumanLabel]
) -> tuple[list[tuple[Verdict, HumanLabel]], int, int]:
    pairs, unavailable, missing = [], 0, 0
    for lab in labels:
        v = verdicts.get(lab.case_id)
        if v is None:
            missing += 1
        elif not v.available:
            unavailable += 1
        else:
            pairs.append((v, lab))
    return pairs, unavailable, missing


def agreement(verdicts: Mapping[str, Verdict], labels: Sequence[HumanLabel]) -> Agreement:
    """Judge vs human, matched by case id."""
    ids = [lab.case_id for lab in labels]
    if len(set(ids)) != len(ids):
        raise ValueError("one human label per case id")
    pairs, unavailable, missing = _paired(verdicts, labels)
    judge = [v.passed for v, _ in pairs]
    human = [lab.passed for _, lab in pairs]
    item_ids = sorted({i for _, lab in pairs for i in lab.items})
    item_kappa: dict[str, float | None] = {}
    item_n: dict[str, int] = {}
    for item in item_ids:
        rows = [(v.result(item).passed, lab.items[item]) for v, lab in pairs if item in lab.items]
        item_kappa[item] = cohen_kappa([j for j, _ in rows], [h for _, h in rows])
        item_n[item] = len(rows)
    n = len(pairs)
    return Agreement(
        n=n,
        unavailable=unavailable,
        missing=missing,
        raw_agreement=sum(j == h for j, h in zip(judge, human, strict=True)) / n if n else None,
        kappa=cohen_kappa(judge, human),
        both_fail=sum(not j and not h for j, h in zip(judge, human, strict=True)),
        judge_only_fail=sum(not j and h for j, h in zip(judge, human, strict=True)),
        human_only_fail=sum(j and not h for j, h in zip(judge, human, strict=True)),
        both_pass=sum(j and h for j, h in zip(judge, human, strict=True)),
        item_kappa=item_kappa,
        item_n=item_n,
    )


def kappa_ci(
    verdicts: Mapping[str, Verdict],
    labels: Sequence[HumanLabel],
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Kappa (overall pass/fail) with a bootstrap CI over labeled cases."""
    pairs, _, _ = _paired(verdicts, labels)
    units = [(v.passed, lab.passed) for v, lab in pairs]
    return bootstrap(
        units, lambda u: cohen_kappa([j for j, _ in u], [h for _, h in u]), n_boot, alpha, seed
    )


def verdicts_by_case(result: SuiteResult) -> dict[str, Verdict]:
    return {c.case_id: c.verdict for c in result.cases}


# ---------------------------------------------------------------- suite rates with CIs

# A metric is a ratio over a subset of judged cases:
#   "false_alarm"                 clean cases flagged
#   "catch" / "catch:<type>"      planted cases flagged (all types / one type)
#   "catch_expected[:<type>]"     planted cases flagged by an expected rubric item


def _in_metric(case: CaseResult, metric: str) -> tuple[bool, bool]:
    """(counts in the denominator, counts in the numerator) for one judged case."""
    if metric == "false_alarm":
        return case.error_type is None, case.flagged
    kind, _, etype = metric.partition(":")
    if kind not in ("catch", "catch_expected"):
        raise ValueError(f"unknown metric {metric!r}")
    if case.error_type is None or (etype and case.error_type != etype):
        return False, False
    return True, case.flagged_expected if kind == "catch_expected" else case.flagged


def _ratio(groups: Sequence[tuple[int, int]]) -> float | None:
    den = sum(d for _, d in groups)
    return sum(n for n, _ in groups) / den if den else None


def _by_sample(cases: Sequence[CaseResult]) -> dict[str, list[CaseResult]]:
    out: dict[str, list[CaseResult]] = {}
    for c in cases:
        if c.available:
            out.setdefault(c.sample_id, []).append(c)
    return out


def _counts(cases: Sequence[CaseResult], metric: str) -> tuple[int, int]:
    num = den = 0
    for c in cases:
        in_den, in_num = _in_metric(c, metric)
        den += in_den
        num += in_den and in_num
    return num, den


def metric_names(result: SuiteResult) -> list[str]:
    types = sorted({c.error_type for c in result.cases if c.error_type is not None})
    names = ["false_alarm", "catch", "catch_expected"]
    for t in types:
        names += [f"catch:{t}", f"catch_expected:{t}"]
    return names


def suite_estimates(
    result: SuiteResult,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Every metric of a planted-error run with a CI, resampling samples (clusters)."""
    groups = _by_sample(result.cases)
    out = {}
    for metric in metric_names(result):
        units = [_counts(cases, metric) for cases in groups.values()]
        out[metric] = bootstrap(units, _ratio, n_boot, alpha, seed)
    return out


def compare(
    a: SuiteResult,
    b: SuiteResult,
    metric: str,
    n_boot: int = DEFAULT_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Paired bootstrap of metric(b) - metric(a) over the cases both judged (same case ids)."""
    avail_a = {c.case_id: c for c in a.cases if c.available}
    avail_b = {c.case_id: c for c in b.cases if c.available}
    shared = sorted(set(avail_a) & set(avail_b))
    groups: dict[str, list[str]] = {}
    for cid in shared:
        if avail_a[cid].sample_id != avail_b[cid].sample_id:
            raise ValueError(f"case {cid!r} maps to different samples in the two runs")
        groups.setdefault(avail_a[cid].sample_id, []).append(cid)
    units = [
        (_counts([avail_a[c] for c in cids], metric), _counts([avail_b[c] for c in cids], metric))
        for cids in groups.values()
    ]

    def diff(u: Sequence[tuple[tuple[int, int], tuple[int, int]]]) -> float | None:
        ra, rb = _ratio([x for x, _ in u]), _ratio([y for _, y in u])
        return None if ra is None or rb is None else rb - ra

    return bootstrap(units, diff, n_boot, alpha, seed)


# ---------------------------------------------------------------- panel vote thresholds


class ThresholdRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: int  # an item fails when at least k of n judges fail it
    n_judges: int
    catch_rate: float | None
    catch_expected_rate: float | None
    false_alarm_rate: float | None
    n_planted: int
    n_clean: int


def vote_threshold_tradeoff(result: SuiteResult, rubric: Rubric) -> tuple[list[ThresholdRow], int]:
    """Catch vs false-alarm rates for every k-of-n vote threshold, recomputed from stored
    member verdicts. Uses only cases where all n members were available (a panel that stopped
    early stored fewer); returns the rows and how many cases were skipped."""
    sizes = {len(c.verdict.members) for c in result.cases if c.verdict.members}
    if not sizes:
        raise ValueError("no panel member verdicts stored in this result")
    n = max(sizes)
    usable = [
        c
        for c in result.cases
        if len(c.verdict.members) == n and all(m.available for m in c.verdict.members)
    ]
    blocking = [i.id for i in rubric.items if rubric.fails_verdict(i.severity)]
    rows = []
    for k in range(1, n + 1):

        def failing_items(c: CaseResult, k: int = k) -> set[str]:
            return {
                item
                for item in blocking
                if sum(not m.result(item).passed for m in c.verdict.members) >= k
            }

        planted = [c for c in usable if c.error_type is not None]
        clean = [c for c in usable if c.error_type is None]
        caught = [bool(failing_items(c)) for c in planted]
        caught_exp = [bool(failing_items(c) & c.expected_items) for c in planted]
        alarms = [bool(failing_items(c)) for c in clean]
        rows.append(
            ThresholdRow(
                k=k,
                n_judges=n,
                catch_rate=sum(caught) / len(caught) if caught else None,
                catch_expected_rate=sum(caught_exp) / len(caught_exp) if caught_exp else None,
                false_alarm_rate=sum(alarms) / len(alarms) if alarms else None,
                n_planted=len(planted),
                n_clean=len(clean),
            )
        )
    return rows, len(result.cases) - len(usable)
