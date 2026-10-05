"""Multi-judge panels: several judges grade the same case; code combines their verdicts.

Per rubric item, a rule turns the judges' votes into one result:
- `any`: the item fails if any judge fails it (strictest; most errors caught, most false alarms)
- `majority`: the item fails if at least half the judges fail it (a tie fails: fail safe)
- `all`: the item fails only if every judge fails it (most lenient; fewest false alarms)
The panel verdict then follows the rubric's severity tiers, exactly like a single judge.

Diversity: members run one after another, and each is told to avoid the model families earlier
members used, so a fallback chain cannot quietly put the same family on the panel twice.

Availability: judges that could not run do not vote. If fewer than `min_available` could run
(default: a majority of the members), the panel itself is unavailable (failed, `error` set).

Frugality: with `stop_early` (default), the panel stops calling members as soon as the remaining
ones cannot change the outcome. The item details of the skipped members are then not collected.
"""

from collections.abc import Sequence
from enum import StrEnum

from judgekit.judge import ItemResult, Judge, Verdict, unavailable
from judgekit.rubric import Rubric


class PanelRule(StrEnum):
    ANY = "any"
    MAJORITY = "majority"
    ALL = "all"


def item_fails(rule: PanelRule, fails: int, voters: int) -> bool:
    """Whether an item fails, given how many of the voting judges failed it."""
    if not 0 <= fails <= voters:
        raise ValueError(f"need 0 <= fails <= voters, got {fails} of {voters}")
    if voters == 0:
        return True  # nobody could judge: never a pass
    if rule is PanelRule.ANY:
        return fails >= 1
    if rule is PanelRule.MAJORITY:
        return 2 * fails >= voters
    return fails == voters


def model_family(verdict: Verdict) -> str:
    """The family of the model that produced a verdict (its last call that names one)."""
    return next((c.family for c in reversed(verdict.calls) if c.family), "")


class JudgePanel:
    """A `Judge` made of other judges (which may themselves be panels)."""

    def __init__(
        self,
        members: Sequence[Judge],
        rubric: Rubric,
        rule: PanelRule = PanelRule.MAJORITY,
        name: str = "panel",
        min_available: int | None = None,
        distinct_families: bool = True,
        stop_early: bool = True,
    ) -> None:
        if not members:
            raise ValueError("a panel needs at least one member")
        ids = [m.judge_id for m in members]
        if len(set(ids)) != len(ids):
            raise ValueError(f"panel member ids must be unique: {ids}")
        n = len(members)
        self.min_available = n // 2 + 1 if min_available is None else min_available
        if not 1 <= self.min_available <= n:
            raise ValueError(f"min_available must be between 1 and {n}, got {self.min_available}")
        self.members = tuple(members)
        self.rubric = rubric
        self.rule = PanelRule(rule)
        self.name = name
        self.distinct_families = distinct_families
        self.stop_early = stop_early

    @property
    def judge_id(self) -> str:
        return self.name

    def evaluate(self, case: str, *, avoid_families: frozenset[str] = frozenset()) -> Verdict:
        verdicts: list[Verdict] = []
        avoid = set(avoid_families)
        for i, member in enumerate(self.members):
            verdict = member.evaluate(case, avoid_families=frozenset(avoid))
            self._check_rubric(verdict, member.judge_id)
            verdicts.append(verdict)
            family = model_family(verdict)
            if self.distinct_families and verdict.available and family:
                avoid.add(family)
            remaining = len(self.members) - i - 1
            if self.stop_early and remaining and self._decided(verdicts, remaining):
                break
        return self.combine(verdicts)

    def combine(self, verdicts: Sequence[Verdict]) -> Verdict:
        """The panel verdict from member verdicts (exposed for replaying stored verdicts)."""
        calls = tuple(c for v in verdicts for c in v.calls)
        members = tuple(verdicts)
        voters = [v for v in verdicts if v.available]
        if len(voters) < self.min_available:
            error = (
                f"only {len(voters)} of {len(self.members)} judges available "
                f"(need {self.min_available})"
            )
            return unavailable(self.rubric, self.name, error, calls).model_copy(
                update={"members": members}
            )
        items = []
        for item in self.rubric.items:
            results = [(v.judge_id, v.result(item.id)) for v in voters]
            failing = [(jid, r) for jid, r in results if not r.passed]
            note = " | ".join(f"{jid}: {r.note}" if r.note else jid for jid, r in failing)
            items.append(
                ItemResult(
                    item_id=item.id,
                    severity=item.severity,
                    passed=not item_fails(self.rule, len(failing), len(voters)),
                    answered=any(r.answered for _, r in results),
                    note=note,
                )
            )
        passed = not any(not r.passed and self.rubric.fails_verdict(r.severity) for r in items)
        return Verdict(
            rubric=self.rubric.name,
            rubric_version=self.rubric.version,
            items=tuple(items),
            passed=passed,
            judge_id=self.name,
            calls=calls,
            members=members,
        )

    def _check_rubric(self, verdict: Verdict, member_id: str) -> None:
        if (verdict.rubric, verdict.rubric_version) != (self.rubric.name, self.rubric.version):
            raise ValueError(
                f"panel member {member_id!r} used rubric {verdict.rubric!r} "
                f"v{verdict.rubric_version}, panel uses {self.rubric.name!r} v{self.rubric.version}"
            )

    def _decided(self, verdicts: Sequence[Verdict], remaining: int) -> bool:
        """True when no outcome of the `remaining` members (pass, fail or unavailable on any
        item) could change the panel's outcome."""
        voters = [v for v in verdicts if v.available]
        if len(voters) + remaining < self.min_available:
            return True  # surely unavailable
        if len(voters) < self.min_available:
            return False  # could still go either way, or end unavailable
        possible_per_item = []
        for item in self.rubric.items:
            if not self.rubric.fails_verdict(item.severity):
                continue
            fails = sum(not v.result(item.id).passed for v in voters)
            possible = {
                item_fails(self.rule, fails + k, len(voters) + extra)
                for extra in range(remaining + 1)
                for k in range(extra + 1)
            }
            if possible == {True}:
                return True  # this item fails whatever happens: the panel fails
            possible_per_item.append(possible)
        return all(p == {False} for p in possible_per_item)
