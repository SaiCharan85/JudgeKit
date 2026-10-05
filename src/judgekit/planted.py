"""Planted-error evaluation: does a judge catch known errors, and leave clean work alone?

1. Start from samples whose explanations are known to be correct (`Sample`: text fields).
2. Plant one known error per copy with an `ErrorType` (a seeded mutator plus the rubric items
   that should catch it). Domain projects define their own error types; `judgekit.mutators`
   has generic text mutators to build them from.
3. `run_suite` grades the clean originals and the planted copies with any `Judge` or panel.
4. `SuiteResult` reports, per error type, the catch rate (planted copies flagged) and how often
   the flag came from an expected item; on clean samples, the false-alarm rate.

Verdicts from a judge that could not run are excluded from every rate (and counted), so an
outage never looks like a catch. Planting is deterministic for a given seed.
"""

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from judgekit.judge import Judge, Verdict
from judgekit.rubric import Rubric


class Sample(BaseModel):
    """A case known to be correct, as named text fields (e.g. evidence, explanation)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    fields: dict[str, str]

    def updated(self, changes: Mapping[str, str]) -> "Sample":
        unknown = set(changes) - set(self.fields)
        if unknown:
            raise KeyError(f"sample {self.id!r} has no fields {sorted(unknown)}")
        return self.model_copy(update={"fields": {**self.fields, **changes}})


class Mutation(BaseModel):
    """What a mutator changed: the new text of the changed fields, and a description."""

    model_config = ConfigDict(frozen=True)

    changes: dict[str, str]
    description: str


Mutator = Callable[[Sample, random.Random], Mutation | None]  # None: not applicable here


@dataclass(frozen=True)
class ErrorType:
    name: str
    expected_items: frozenset[str]  # rubric items that should flag this error
    mutate: Mutator
    description: str = ""


class Planted(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    sample_id: str
    error_type: str
    expected_items: frozenset[str]
    description: str
    sample: Sample  # the mutated copy


class PlantedSuite(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int
    clean: tuple[Sample, ...]
    planted: tuple[Planted, ...]
    not_applicable: dict[str, int]  # error type -> samples it could not be planted in

    @property
    def n_cases(self) -> int:
        """Judge evaluations a run will make (use for dry-run request estimates)."""
        return len(self.clean) + len(self.planted)

    def counts(self) -> dict[str, int]:
        return dict(Counter(p.error_type for p in self.planted))


def _rng(seed: int, *parts: str) -> random.Random:
    # String seeds are hashed with SHA-512 by `random`, so this is stable across processes.
    return random.Random(":".join([str(seed), *parts]))


def plant(
    samples: Sequence[Sample],
    error_types: Sequence[ErrorType],
    seed: int = 0,
    max_per_type: int | None = None,
    max_clean: int | None = None,
) -> PlantedSuite:
    """Plant each error type into the samples (at most `max_per_type` each, for pilots)."""
    ids = [s.id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample ids must be unique")
    names = [e.name for e in error_types]
    if len(set(names)) != len(names):
        raise ValueError(f"error type names must be unique: {names}")
    for e in error_types:
        if not e.expected_items:
            raise ValueError(f"error type {e.name!r} names no expected rubric items")
    planted: list[Planted] = []
    not_applicable: dict[str, int] = {}
    for e in error_types:
        order = list(samples)
        _rng(seed, "order", e.name).shuffle(order)
        made = 0
        skipped = 0
        for s in order:
            if max_per_type is not None and made >= max_per_type:
                break
            mutation = e.mutate(s, _rng(seed, e.name, s.id))
            mutated = s.updated(mutation.changes) if mutation else s
            if mutation is None or mutated.fields == s.fields:
                skipped += 1
                continue
            planted.append(
                Planted(
                    case_id=f"{s.id}::{e.name}",
                    sample_id=s.id,
                    error_type=e.name,
                    expected_items=e.expected_items,
                    description=mutation.description,
                    sample=mutated,
                )
            )
            made += 1
        not_applicable[e.name] = skipped
    clean = list(samples)
    if max_clean is not None and max_clean < len(clean):
        position = {sid: i for i, sid in enumerate(ids)}
        clean = sorted(_rng(seed, "clean").sample(clean, max_clean), key=lambda s: position[s.id])
    return PlantedSuite(
        seed=seed, clean=tuple(clean), planted=tuple(planted), not_applicable=not_applicable
    )


# ---------------------------------------------------------------- running and scoring


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    sample_id: str
    error_type: str | None  # None for a clean sample
    expected_items: frozenset[str] = frozenset()
    verdict: Verdict

    @property
    def available(self) -> bool:
        return self.verdict.available

    @property
    def flagged(self) -> bool:
        """The judge failed this case (for a planted case: caught; for a clean one: false alarm)."""
        return self.available and not self.verdict.passed

    @property
    def flagged_expected(self) -> bool:
        """At least one expected rubric item failed: caught for the right reason."""
        return self.available and any(
            not self.verdict.result(i).passed for i in self.expected_items
        )


class TypeStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    error_type: str  # "clean" for the clean samples
    n: int
    unavailable: int
    flagged: int
    flagged_expected: int = 0

    @property
    def judged(self) -> int:
        return self.n - self.unavailable

    @property
    def rate(self) -> float | None:
        """Catch rate (planted) or false-alarm rate (clean); None if nothing was judged."""
        return self.flagged / self.judged if self.judged else None

    @property
    def expected_rate(self) -> float | None:
        return self.flagged_expected / self.judged if self.judged else None


CLEAN = "clean"


class SuiteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    rubric: str
    rubric_version: str
    judge_id: str
    seed: int
    cases: tuple[CaseResult, ...]

    def clean_stats(self) -> TypeStats:
        return _stats(CLEAN, [c for c in self.cases if c.error_type is None])

    def by_type(self) -> list[TypeStats]:
        types = sorted({c.error_type for c in self.cases if c.error_type is not None})
        return [_stats(t, [c for c in self.cases if c.error_type == t]) for t in types]

    def overall_catch(self) -> TypeStats:
        return _stats("all planted", [c for c in self.cases if c.error_type is not None])

    def item_flags(self) -> dict[str, dict[str, int]]:
        """Per error type (and clean), how often each rubric item failed (judged cases only)."""
        out: dict[str, Counter[str]] = {}
        for c in self.cases:
            if not c.available:
                continue
            key = c.error_type or CLEAN
            out.setdefault(key, Counter()).update(i.item_id for i in c.verdict.failures)
        return {k: dict(v) for k, v in sorted(out.items())}

    def to_markdown(self) -> str:
        def pct(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.1%}"

        clean = self.clean_stats()
        lines = [
            f"Judge `{self.judge_id}`, rubric `{self.rubric}` v{self.rubric_version}, "
            f"seed {self.seed}.",
            "",
            "| error type | n | judged | caught | caught by expected item |",
            "|---|---|---|---|---|",
        ]
        for s in [*self.by_type(), self.overall_catch()]:
            lines.append(
                f"| {s.error_type} | {s.n} | {s.judged} | {pct(s.rate)} | {pct(s.expected_rate)} |"
            )
        lines += [
            "",
            f"False alarms on clean samples: {pct(clean.rate)} "
            f"({clean.flagged} of {clean.judged} judged; {clean.unavailable} unavailable).",
        ]
        return "\n".join(lines)


def _stats(name: str, cases: Sequence[CaseResult]) -> TypeStats:
    return TypeStats(
        error_type=name,
        n=len(cases),
        unavailable=sum(not c.available for c in cases),
        flagged=sum(c.flagged for c in cases),
        flagged_expected=sum(c.flagged_expected for c in cases),
    )


def check_expected_items(suite: PlantedSuite, rubric: Rubric) -> None:
    """Every expected item of every planted case must exist in the rubric."""
    missing = sorted({i for p in suite.planted for i in p.expected_items} - set(rubric.ids))
    if missing:
        raise ValueError(f"expected items not in rubric {rubric.name!r}: {missing}")


def run_suite(
    judge: Judge,
    suite: PlantedSuite,
    render: Callable[[Sample], str],
    rubric: Rubric,
    avoid_families: frozenset[str] = frozenset(),
    on_case: Callable[[CaseResult], None] | None = None,
    resume: Sequence[CaseResult] = (),
) -> SuiteResult:
    """Grade every clean and planted case. `render` turns a sample into the case text the judge
    sees (domain code); `on_case` is called after each newly judged case (progress,
    checkpointing). `resume`: results saved from an interrupted run; those cases are reused, not
    judged again, so a restart never re-spends requests."""
    check_expected_items(suite, rubric)
    todo: list[tuple[str, str, str | None, frozenset[str], Sample]] = [
        (s.id, s.id, None, frozenset(), s) for s in suite.clean
    ]
    todo += [
        (p.case_id, p.sample_id, p.error_type, p.expected_items, p.sample) for p in suite.planted
    ]
    done = {r.case_id: r for r in resume}
    results = []
    for case_id, sample_id, error_type, expected, sample in todo:
        if case_id in done:
            results.append(done[case_id])
            continue
        verdict = judge.evaluate(render(sample), avoid_families=avoid_families)
        result = CaseResult(
            case_id=case_id,
            sample_id=sample_id,
            error_type=error_type,
            expected_items=expected,
            verdict=verdict,
        )
        results.append(result)
        if on_case is not None:
            on_case(result)
    return SuiteResult(
        rubric=rubric.name,
        rubric_version=rubric.version,
        judge_id=judge.judge_id,
        seed=suite.seed,
        cases=tuple(results),
    )
