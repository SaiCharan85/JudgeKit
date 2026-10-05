import pytest

from helpers import ALL_YES, respond
from judgekit import (
    CallInfo,
    Judge,
    JudgePanel,
    JudgeResponse,
    LLMCallError,
    LLMJudge,
    PanelRule,
    Rubric,
    item_fails,
    model_family,
    parse_rubric,
)
from judgekit.testing import ScriptedLLM

CRIT_NO = respond(crit="no", maj="yes", mino="yes")
MAJ_NO = respond(crit="yes", maj="no", mino="yes")
MINO_NO = respond(crit="yes", maj="yes", mino="no")


def member(
    rubric: Rubric, name: str, *script: JudgeResponse | Exception, family: str | None = None
) -> tuple[LLMJudge, ScriptedLLM]:
    llm = ScriptedLLM(list(script), CallInfo(model=name, family=family or f"fam_{name}"))
    return LLMJudge(llm, rubric, name=name), llm


def panel(rubric: Rubric, *scripts: JudgeResponse | Exception, **kw: object) -> JudgePanel:
    members = [member(rubric, f"j{i}", s)[0] for i, s in enumerate(scripts)]
    return JudgePanel(members, rubric, **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------- vote rules


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (PanelRule.ANY, [False, True, True, True]),
        (PanelRule.MAJORITY, [False, False, True, True]),
        (PanelRule.ALL, [False, False, False, True]),
    ],
)
def test_item_fails_by_votes_out_of_three(rule: PanelRule, expected: list[bool]) -> None:
    assert [item_fails(rule, k, 3) for k in range(4)] == expected


def test_majority_tie_fails() -> None:
    assert item_fails(PanelRule.MAJORITY, 1, 2)
    assert not item_fails(PanelRule.MAJORITY, 1, 3)


def test_no_voters_never_pass() -> None:
    for rule in PanelRule:
        assert item_fails(rule, 0, 0)


def test_invalid_vote_counts_rejected() -> None:
    with pytest.raises(ValueError):
        item_fails(PanelRule.ANY, 3, 2)
    with pytest.raises(ValueError):
        item_fails(PanelRule.ANY, -1, 2)


@pytest.mark.parametrize(
    ("rule", "passed"), [(PanelRule.ANY, False), (PanelRule.MAJORITY, True), (PanelRule.ALL, True)]
)
def test_one_dissent_out_of_three(rubric: Rubric, rule: PanelRule, passed: bool) -> None:
    v = panel(rubric, CRIT_NO, ALL_YES, ALL_YES, rule=rule, stop_early=False).evaluate("c")
    assert v.passed is passed
    assert v.result("crit").passed is passed


def test_minor_only_failures_do_not_fail_panel(rubric: Rubric) -> None:
    v = panel(rubric, MINO_NO, MINO_NO, MINO_NO, rule=PanelRule.ALL).evaluate("c")
    assert v.passed
    assert not v.result("mino").passed


def test_items_combine_independently(rubric: Rubric) -> None:
    # each judge flags a different item: under majority, no single item reaches half
    v = panel(rubric, CRIT_NO, MAJ_NO, MINO_NO, stop_early=False).evaluate("c")
    assert v.passed
    assert all(i.passed for i in v.items)


# ---------------------------------------------------------------- verdict contents


def test_notes_name_the_dissenting_judges(rubric: Rubric) -> None:
    flagged = JudgeResponse.model_validate(
        {"answers": [{"id": "crit", "answer": "no", "note": "date invented"},
                     {"id": "maj", "answer": "yes"}, {"id": "mino", "answer": "yes"}]}
    )  # fmt: skip
    v = panel(rubric, flagged, CRIT_NO, ALL_YES, stop_early=False).evaluate("c")
    assert v.result("crit").note == "j0: date invented | j1"
    assert v.result("maj").note == ""


def test_panel_verdict_keeps_members_and_all_calls(rubric: Rubric) -> None:
    v = panel(rubric, ALL_YES, ALL_YES, ALL_YES, stop_early=False, name="p").evaluate("c")
    assert v.judge_id == "p"
    assert [m.judge_id for m in v.members] == ["j0", "j1", "j2"]
    assert [c.model for c in v.calls] == ["j0", "j1", "j2"]


def test_member_with_other_rubric_is_a_config_error(rubric: Rubric) -> None:
    other = parse_rubric({"name": "other", "version": "1", "instructions": "x",
                          "items": [{"id": "crit", "question": "q?"}]})  # fmt: skip
    judge, _ = member(other, "j0", respond(crit="yes"))
    with pytest.raises(ValueError, match="used rubric 'other'"):
        JudgePanel([judge], rubric).evaluate("c")


# ---------------------------------------------------------------- availability


def test_unavailable_member_does_not_vote(rubric: Rubric) -> None:
    v = panel(rubric, LLMCallError("down"), CRIT_NO, ALL_YES, rule=PanelRule.MAJORITY,
              stop_early=False).evaluate("c")  # fmt: skip
    # 1 of 2 voters failed crit -> a tie -> fails (fail safe)
    assert v.available and not v.passed
    assert len(v.members) == 3 and not v.members[0].available


def test_too_few_available_makes_panel_unavailable(rubric: Rubric) -> None:
    v = panel(rubric, LLMCallError("a"), LLMCallError("b"), ALL_YES, stop_early=False).evaluate("c")
    assert not v.available and not v.passed
    assert "only 1 of 3 judges available (need 2)" in v.error
    assert len(v.members) == 3


def test_custom_min_available(rubric: Rubric) -> None:
    v = panel(rubric, LLMCallError("a"), LLMCallError("b"), ALL_YES, min_available=1).evaluate("c")
    assert v.available and v.passed


@pytest.mark.parametrize("bad", [0, 4])
def test_min_available_range_checked(rubric: Rubric, bad: int) -> None:
    with pytest.raises(ValueError, match="min_available"):
        panel(rubric, ALL_YES, ALL_YES, ALL_YES, min_available=bad)


def test_empty_panel_rejected(rubric: Rubric) -> None:
    with pytest.raises(ValueError, match="at least one member"):
        JudgePanel([], rubric)


def test_duplicate_member_ids_rejected(rubric: Rubric) -> None:
    a, _ = member(rubric, "same", ALL_YES)
    b, _ = member(rubric, "same", ALL_YES)
    with pytest.raises(ValueError, match="unique"):
        JudgePanel([a, b], rubric)


# ---------------------------------------------------------------- family diversity


def test_each_member_avoids_families_already_used(rubric: Rubric) -> None:
    pairs = [member(rubric, f"j{i}", ALL_YES, family=f"f{i}") for i in range(3)]
    p = JudgePanel([j for j, _ in pairs], rubric, stop_early=False)
    p.evaluate("c", avoid_families=frozenset({"writer"}))
    seen = [llm.calls[0][2] for _, llm in pairs]
    assert seen == [
        frozenset({"writer"}),
        frozenset({"writer", "f0"}),
        frozenset({"writer", "f0", "f1"}),
    ]


def test_unavailable_member_does_not_reserve_its_family(rubric: Rubric) -> None:
    a, _ = member(rubric, "a", LLMCallError("down"), family="fa")
    b, llm_b = member(rubric, "b", ALL_YES, family="fb")
    JudgePanel([a, b], rubric, min_available=1, stop_early=False).evaluate("c")
    assert llm_b.calls[0][2] == frozenset()


def test_diversity_can_be_disabled(rubric: Rubric) -> None:
    pairs = [member(rubric, f"j{i}", ALL_YES, family="same") for i in range(2)]
    JudgePanel([j for j, _ in pairs], rubric, distinct_families=False).evaluate("c")
    assert pairs[1][1].calls[0][2] == frozenset()


def test_model_family_reads_last_named_call() -> None:
    from judgekit import Verdict

    v = Verdict(rubric="r", rubric_version="1", items=(), passed=True,
                calls=(CallInfo(family="a"), CallInfo(family="b"), CallInfo()))  # fmt: skip
    assert model_family(v) == "b"
    assert model_family(v.model_copy(update={"calls": ()})) == ""


# ---------------------------------------------------------------- early stopping (frugality)


def _calls(pairs: list[tuple[LLMJudge, ScriptedLLM]]) -> list[int]:
    return [len(llm.calls) for _, llm in pairs]


def test_any_rule_stops_at_first_disqualifying_failure(rubric: Rubric) -> None:
    pairs = [member(rubric, "j0", CRIT_NO)] + [member(rubric, f"j{i}", ALL_YES) for i in (1, 2)]
    v = JudgePanel([j for j, _ in pairs], rubric, rule=PanelRule.ANY, min_available=1).evaluate("c")
    assert not v.passed
    assert _calls(pairs) == [1, 0, 0]


def test_majority_stops_once_two_of_three_agree(rubric: Rubric) -> None:
    pairs = [member(rubric, f"j{i}", ALL_YES) for i in range(3)]
    v = JudgePanel([j for j, _ in pairs], rubric).evaluate("c")
    assert v.passed
    assert _calls(pairs) == [1, 1, 0]


def test_majority_keeps_going_while_undecided(rubric: Rubric) -> None:
    pairs = [member(rubric, "j0", CRIT_NO), member(rubric, "j1", ALL_YES),
             member(rubric, "j2", ALL_YES)]  # fmt: skip
    v = JudgePanel([j for j, _ in pairs], rubric).evaluate("c")
    assert v.passed  # 1 of 3 failed crit
    assert _calls(pairs) == [1, 1, 1]


def test_all_rule_stops_at_first_clean_pass(rubric: Rubric) -> None:
    pairs = [member(rubric, f"j{i}", ALL_YES) for i in range(3)]
    v = JudgePanel([j for j, _ in pairs], rubric, rule=PanelRule.ALL, min_available=1).evaluate("c")
    assert v.passed
    assert _calls(pairs) == [1, 0, 0]


def test_stops_when_availability_is_already_lost(rubric: Rubric) -> None:
    pairs = [member(rubric, "j0", LLMCallError("a")), member(rubric, "j1", LLMCallError("b")),
             member(rubric, "j2", ALL_YES)]  # fmt: skip
    v = JudgePanel([j for j, _ in pairs], rubric, min_available=3).evaluate("c")
    assert not v.available
    assert _calls(pairs) == [1, 0, 0]  # 0 voters + 2 remaining < 3 needed after the first


def test_does_not_stop_before_min_available_is_reached(rubric: Rubric) -> None:
    pairs = [member(rubric, "j0", CRIT_NO), member(rubric, "j1", CRIT_NO)]
    JudgePanel([j for j, _ in pairs], rubric, rule=PanelRule.ANY, min_available=2).evaluate("c")
    assert _calls(pairs) == [1, 1]


def test_stop_early_off_calls_everyone(rubric: Rubric) -> None:
    pairs = [member(rubric, f"j{i}", ALL_YES) for i in range(3)]
    JudgePanel([j for j, _ in pairs], rubric, stop_early=False).evaluate("c")
    assert _calls(pairs) == [1, 1, 1]


def test_minor_items_never_keep_the_panel_going(rubric: Rubric) -> None:
    pairs = [member(rubric, "j0", MINO_NO), member(rubric, "j1", MINO_NO),
             member(rubric, "j2", MINO_NO)]  # fmt: skip
    JudgePanel([j for j, _ in pairs], rubric).evaluate("c")
    assert _calls(pairs) == [1, 1, 0]  # outcome fixed after 2 (minor never fails the panel)


# ---------------------------------------------------------------- composition


def test_panel_is_a_judge_and_can_be_nested(rubric: Rubric) -> None:
    inner = panel(rubric, CRIT_NO, CRIT_NO, rule=PanelRule.ALL, name="inner")
    outer_member, _ = member(rubric, "solo", ALL_YES)
    outer = JudgePanel([inner, outer_member], rubric, rule=PanelRule.ANY, name="outer",
                       stop_early=False)  # fmt: skip
    assert isinstance(inner, Judge) and isinstance(outer, Judge)
    v = outer.evaluate("c")
    assert not v.passed
    assert v.members[0].judge_id == "inner" and len(v.members[0].members) == 2


def test_combine_replays_stored_verdicts(rubric: Rubric) -> None:
    p = panel(rubric, ALL_YES, ALL_YES, ALL_YES, stop_early=False)
    stored = p.evaluate("c").members
    assert p.combine(stored).passed
