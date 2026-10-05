import random

import pytest

from judgekit import Sample
from judgekit.mutators import (
    drop_sentence,
    insert_sentence,
    negation_flip,
    number_swap,
    replace_phrase,
)


def s(text: str, **other: str) -> Sample:
    return Sample(id="s1", fields={"explanation": text, **other})


def rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------- number_swap


def test_number_swap_changes_one_number() -> None:
    m = number_swap()(s("The fee is 500 and the limit is 2."), rng())
    assert m is not None
    new = m.changes["explanation"]
    assert new != "The fee is 500 and the limit is 2."
    assert new.count(" is ") == 2  # only the number changed
    assert m.description.startswith("changed ")


def test_number_swap_keeps_grouping_and_decimals() -> None:
    m = number_swap(factors=(2.0,))(s("Total 1,250.50 due."), rng())
    assert m is not None and m.changes["explanation"] == "Total 2,501.00 due."


def test_number_swap_ignores_dates_ids_and_words() -> None:
    text = "Filed 2024-01-05 under ref A12 by unit7 at 3/4 time."
    assert number_swap()(s(text), rng()) is None


def test_number_swap_skips_zero() -> None:
    assert number_swap()(s("There were 0 items."), rng()) is None


def test_number_swap_handles_sentence_end_period() -> None:
    m = number_swap(factors=(2.0,))(s("The amount was 300."), rng())
    assert m is not None and m.changes["explanation"] == "The amount was 600."


def test_number_swap_not_applicable_without_numbers_or_field() -> None:
    assert number_swap()(s("No numbers here."), rng()) is None
    assert number_swap(field="missing")(s("Has 5."), rng()) is None
    assert number_swap()(s("   "), rng()) is None


def test_number_swap_tries_other_factors_when_rounding_collides() -> None:
    # 1 x 1.25 rounds back to "1"; another factor must be used
    m = number_swap(factors=(1.25, 2.0))(s("Only 1 left."), rng())
    assert m is not None and m.changes["explanation"] == "Only 2 left."


@pytest.mark.parametrize("bad", [(), (1.0,), (0.0,), (-2.0,)])
def test_number_swap_validates_factors(bad: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        number_swap(factors=bad)


# ---------------------------------------------------------------- drop_sentence


def test_drop_sentence_removes_exactly_one() -> None:
    text = "First point. Second point. Third point."
    m = drop_sentence()(s(text), rng())
    assert m is not None
    assert len(m.changes["explanation"].split(". ")) == 2
    assert "dropped:" in m.description


def test_drop_sentence_with_match_targets_that_sentence() -> None:
    text = "The cause was noted. The report came late. Everything else is fine."
    m = drop_sentence(match=r"report")(s(text), rng())
    assert m is not None
    assert "report" not in m.changes["explanation"]
    assert "cause" in m.changes["explanation"]


def test_drop_sentence_not_applicable() -> None:
    assert drop_sentence()(s("Only one sentence."), rng()) is None
    assert drop_sentence(match="absent")(s("One. Two."), rng()) is None


# ---------------------------------------------------------------- insert_sentence


def test_insert_sentence_adds_one_from_pool() -> None:
    m = insert_sentence(["A witness confirmed it."])(s("One. Two."), rng())
    assert m is not None
    assert "A witness confirmed it." in m.changes["explanation"]
    assert m.changes["explanation"].count(".") == 3


def test_insert_sentence_skips_sentences_already_present() -> None:
    text = "A witness confirmed it. Two."
    assert insert_sentence(["A witness confirmed it."])(s(text), rng()) is None


def test_insert_sentence_needs_a_pool() -> None:
    with pytest.raises(ValueError):
        insert_sentence(["  ", ""])


# ---------------------------------------------------------------- replace_phrase


def test_replace_phrase_both_directions_and_capital() -> None:
    swap = replace_phrase([("approved", "rejected")])
    m = swap(s("Approved after review."), rng())
    assert m is not None and m.changes["explanation"] == "Rejected after review."
    m = swap(s("It was rejected."), rng())
    assert m is not None and m.changes["explanation"] == "It was approved."


def test_replace_phrase_whole_words_only() -> None:
    assert replace_phrase([("cat", "dog")])(s("The category is set."), rng()) is None


def test_replace_phrase_validates_pairs() -> None:
    with pytest.raises(ValueError):
        replace_phrase([("same", "SAME"), ("", "x")])


# ---------------------------------------------------------------- negation_flip


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The item is valid.", "The item is not valid."),
        ("The item is not valid.", "The item is valid."),
        ("It cannot apply.", "It can apply."),
        ("They did respond.", "They did not respond."),
    ],
)
def test_negation_flip(text: str, expected: str) -> None:
    m = negation_flip()(s(text), rng())
    assert m is not None and m.changes["explanation"] == expected


def test_negation_flip_not_applicable() -> None:
    assert negation_flip()(s("Plain statement here."), rng()) is None


# ---------------------------------------------------------------- determinism and fields


def test_same_seed_same_mutation() -> None:
    text = "A 10. B 20. C 30. D 40."
    a = number_swap()(s(text), rng(7))
    b = number_swap()(s(text), rng(7))
    assert a == b


def test_mutators_target_their_field_only() -> None:
    m = number_swap(field="evidence")(s("Keep 5.", evidence="Value 10."), rng())
    assert m is not None and set(m.changes) == {"evidence"}
