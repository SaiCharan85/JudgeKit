"""Bias tests: does the verdict change when only something irrelevant changes?

Each `BiasProbe` makes a variant of a case that should be judged the same as the original:
- `position_probe`: the same sections in another order (e.g. explanation before evidence)
- `verbosity_probe`: the explanation padded with content-free filler sentences
- `politeness_probe`: courteous / confident phrasing added around the explanation
- `repeat_probe`: the identical case again: the judge's own noise floor (only meaningful when
  the LLM client does not cache, since a cached repeat always agrees)

`run_bias` judges every sample once as the baseline (shared by all probes, or reused from an
earlier run) plus once per probe, and reports per probe:
- flip rate: share of cases whose pass/fail changed
- net shift: (fail->pass minus pass->fail) / cases; > 0 means the variant is judged more
  leniently (e.g. padding buys passes), < 0 more strictly
with bootstrap CIs. Unavailable verdicts are excluded. Variants are seeded.
"""

import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from judgekit.calibration import DEFAULT_BOOT, Estimate, bootstrap
from judgekit.judge import Judge, Verdict
from judgekit.planted import Sample

Render = Callable[[Sample], str]
Vary = Callable[[Sample, random.Random], Sample | None]  # None: probe not applicable

DEFAULT_FIELD = "explanation"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Neutral filler: restates that reasoning exists without adding any fact, number or judgment.
DEFAULT_FILLER = (
    "The reasoning above is set out step by step for clarity.",
    "Each point is stated in turn so that it can be followed easily.",
    "This summary repeats the structure of the analysis without adding anything new.",
    "The same considerations are listed here again in the order they were discussed.",
)
DEFAULT_POLITE_OPENERS = (
    "Thank you for your patience while this was reviewed.",
    "We appreciate the opportunity to explain this carefully.",
    "We have reviewed every detail thoroughly and with great care.",
)
DEFAULT_POLITE_CLOSERS = (
    "We are confident this conclusion is correct and fair.",
    "Please do not hesitate to reach out with any questions.",
    "We hope this thorough explanation is helpful.",
)


@dataclass(frozen=True)
class BiasProbe:
    """A variant that should not change the verdict: a content edit (`vary`), another way of
    rendering the case (`render`), or both. With neither, the case is repeated unchanged."""

    name: str
    vary: Vary | None = None
    render: Render | None = None
    allow_identical: bool = False  # only the repeat probe may produce the exact same case


# ---------------------------------------------------------------- renderers and probes


def sectioned(order: Sequence[str], header: str = "## {name}") -> Render:
    """Render a sample as headed sections in `order` (fields not listed are left out)."""
    if not order:
        raise ValueError("sectioned needs at least one field")

    def render(sample: Sample) -> str:
        missing = [f for f in order if f not in sample.fields]
        if missing:
            raise KeyError(f"sample {sample.id!r} has no fields {missing}")
        blocks = [f"{header.format(name=f.upper())}\n{sample.fields[f]}" for f in order]
        return "\n\n".join(blocks)

    return render


def position_probe(render_reordered: Render, name: str = "position") -> BiasProbe:
    """The same content rendered in another order, e.g. `sectioned([... reversed ...])`."""
    return BiasProbe(name=name, render=render_reordered)


def verbosity_probe(
    field: str = DEFAULT_FIELD,
    filler: Sequence[str] = DEFAULT_FILLER,
    n_sentences: int = 2,
    name: str = "verbosity",
) -> BiasProbe:
    """Pad one field with `n_sentences` content-free filler sentences at random positions."""
    pool = [s.strip() for s in filler if s.strip()]
    if not pool or n_sentences < 1 or n_sentences > len(pool):
        raise ValueError("need 1 <= n_sentences <= number of filler sentences")

    def vary(sample: Sample, rng: random.Random) -> Sample | None:
        text = sample.fields.get(field, "").strip()
        if not text:
            return None
        parts = _SENTENCE_END.split(text)
        for sentence in rng.sample(pool, n_sentences):
            parts.insert(rng.randint(0, len(parts)), sentence)
        return sample.updated({field: " ".join(parts)})

    return BiasProbe(name=name, vary=vary)


def politeness_probe(
    field: str = DEFAULT_FIELD,
    openers: Sequence[str] = DEFAULT_POLITE_OPENERS,
    closers: Sequence[str] = DEFAULT_POLITE_CLOSERS,
    name: str = "politeness",
) -> BiasProbe:
    """Wrap one field in a courteous / confident opener and closer."""
    if not openers and not closers:
        raise ValueError("politeness_probe needs openers or closers")

    def vary(sample: Sample, rng: random.Random) -> Sample | None:
        text = sample.fields.get(field, "").strip()
        if not text:
            return None
        head = f"{rng.choice(list(openers))} " if openers else ""
        tail = f" {rng.choice(list(closers))}" if closers else ""
        return sample.updated({field: f"{head}{text}{tail}"})

    return BiasProbe(name=name, vary=vary)


def repeat_probe(name: str = "repeat") -> BiasProbe:
    """The identical case again: measures the judge's own run-to-run noise."""
    return BiasProbe(name=name, allow_identical=True)


# ---------------------------------------------------------------- results


class BiasCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    probe: str
    sample_id: str
    baseline: Verdict
    variant: Verdict

    @property
    def comparable(self) -> bool:
        return self.baseline.available and self.variant.available

    @property
    def flipped(self) -> bool:
        return self.comparable and self.baseline.passed != self.variant.passed

    @property
    def to_lenient(self) -> bool:
        return self.flipped and self.variant.passed

    @property
    def to_strict(self) -> bool:
        return self.flipped and not self.variant.passed

    @property
    def item_flips(self) -> frozenset[str]:
        """Rubric items whose result changed (comparable cases only)."""
        if not self.comparable:
            return frozenset()
        return frozenset(
            i.item_id
            for i in self.baseline.items
            if i.passed != self.variant.result(i.item_id).passed
        )


class BiasStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    probe: str
    n: int
    comparable: int
    flips: int
    to_lenient: int
    to_strict: int
    not_applicable: int

    @property
    def flip_rate(self) -> float | None:
        return self.flips / self.comparable if self.comparable else None

    @property
    def net_shift(self) -> float | None:
        return (self.to_lenient - self.to_strict) / self.comparable if self.comparable else None


class BiasResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    judge_id: str
    seed: int
    cases: tuple[BiasCase, ...]
    not_applicable: dict[str, int]

    def probes(self) -> list[str]:
        return sorted({c.probe for c in self.cases} | set(self.not_applicable))

    def stats(self, probe: str) -> BiasStats:
        cases = [c for c in self.cases if c.probe == probe]
        return BiasStats(
            probe=probe,
            n=len(cases),
            comparable=sum(c.comparable for c in cases),
            flips=sum(c.flipped for c in cases),
            to_lenient=sum(c.to_lenient for c in cases),
            to_strict=sum(c.to_strict for c in cases),
            not_applicable=self.not_applicable.get(probe, 0),
        )

    def item_flip_counts(self, probe: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.cases:
            if c.probe == probe:
                for item in c.item_flips:
                    counts[item] = counts.get(item, 0) + 1
        return dict(sorted(counts.items()))

    def estimates(
        self, n_boot: int = DEFAULT_BOOT, alpha: float = 0.05, seed: int = 0
    ) -> dict[str, dict[str, Estimate]]:
        """Per probe: flip rate and net shift with bootstrap CIs over comparable cases."""
        out = {}
        for probe in self.probes():
            units = [
                (c.flipped, int(c.to_lenient) - int(c.to_strict))
                for c in self.cases
                if c.probe == probe and c.comparable
            ]
            out[probe] = {
                "flip_rate": bootstrap(units, _mean_of(0), n_boot, alpha, seed),
                "net_shift": bootstrap(units, _mean_of(1), n_boot, alpha, seed),
            }
        return out

    def to_markdown(self, n_boot: int = DEFAULT_BOOT, seed: int = 0) -> str:
        def ci(e: Estimate) -> str:
            if e.value is None:
                return "n/a"
            if e.low is None or e.high is None:
                return f"{e.value:+.1%}"
            return f"{e.value:+.1%} [{e.low:+.1%}, {e.high:+.1%}]"

        est = self.estimates(n_boot=n_boot, seed=seed)
        lines = [
            f"Judge `{self.judge_id}`, seed {self.seed}. Net shift > 0: the variant is judged "
            "more leniently.",
            "",
            "| probe | compared | flips | flip rate | net shift | not applicable |",
            "|---|---|---|---|---|---|",
        ]
        for probe in self.probes():
            s = self.stats(probe)
            lines.append(
                f"| {probe} | {s.comparable} | {s.flips} | {ci(est[probe]['flip_rate'])} "
                f"| {ci(est[probe]['net_shift'])} | {s.not_applicable} |"
            )
        return "\n".join(lines)


def _mean_of(index: int) -> Callable[[Sequence[tuple[bool, int]]], float | None]:
    def mean(units: Sequence[tuple[bool, int]]) -> float | None:
        return sum(u[index] for u in units) / len(units) if units else None

    return mean


# ---------------------------------------------------------------- running


def _rng(seed: int, *parts: str) -> random.Random:
    return random.Random(":".join([str(seed), *parts]))


def estimate_bias_calls(
    n_samples: int, n_probes: int, baseline_reused: bool = False, calls_per_verdict: int = 1
) -> int:
    """Judge calls a `run_bias` will make, for dry runs (`calls_per_verdict` = panel size)."""
    verdicts = n_samples * n_probes + (0 if baseline_reused else n_samples)
    return verdicts * calls_per_verdict


def run_bias(
    judge: Judge,
    samples: Sequence[Sample],
    probes: Sequence[BiasProbe],
    render: Render,
    seed: int = 0,
    baseline: Mapping[str, Verdict] | None = None,
    avoid_families: frozenset[str] = frozenset(),
    on_case: Callable[[BiasCase], None] | None = None,
    resume: Sequence[BiasCase] = (),
) -> BiasResult:
    """Judge each sample once as the baseline (or reuse `baseline`, e.g. the clean cases of a
    planted-error run) and once per probe variant. `resume` reuses saved cases (and their
    baselines) from an interrupted run without calling the judge again."""
    names = [p.name for p in probes]
    if len(set(names)) != len(names):
        raise ValueError(f"probe names must be unique: {names}")
    ids = [s.id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample ids must be unique")
    done = {(c.probe, c.sample_id): c for c in resume}
    base: dict[str, Verdict] = dict(baseline or {})
    for c in resume:
        base.setdefault(c.sample_id, c.baseline)
    cases: list[BiasCase] = []
    not_applicable = dict.fromkeys(names, 0)
    for sample in samples:
        original = render(sample)
        variants: list[tuple[BiasProbe, str | None]] = []
        for probe in probes:
            if (probe.name, sample.id) in done:
                variants.append((probe, None))
                continue
            varied = probe.vary(sample, _rng(seed, probe.name, sample.id)) if probe.vary else sample
            text = (probe.render or render)(varied) if varied is not None else None
            if text is None or (text == original and not probe.allow_identical):
                not_applicable[probe.name] += 1
                continue
            variants.append((probe, text))
        if not variants:
            continue
        if sample.id not in base:
            base[sample.id] = judge.evaluate(original, avoid_families=avoid_families)
        for probe, text in variants:
            if text is None:
                cases.append(done[(probe.name, sample.id)])
                continue
            case = BiasCase(
                probe=probe.name,
                sample_id=sample.id,
                baseline=base[sample.id],
                variant=judge.evaluate(text, avoid_families=avoid_families),
            )
            cases.append(case)
            if on_case is not None:
                on_case(case)
    return BiasResult(
        judge_id=judge.judge_id, seed=seed, cases=tuple(cases), not_applicable=not_applicable
    )
