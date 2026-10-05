from pathlib import Path

import pytest

from judgekit import DEFAULT_FAIL_ON, Rubric, RubricError, Severity, load_rubric, parse_rubric

VALID = """
name: explanation_quality
version: "1.0"
instructions: Review the reasoning, not the outcome.
items:
  - id: facts_supported
    question: Is every fact the explanation relies on present in the evidence?
    severity: critical
    guidance: An invented date counts as no.
  - id: material_points
    question: Does the explanation address every point that could change the outcome?
  - id: clear_wording
    question: Is the explanation clearly worded?
    severity: minor
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "rubric.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def _mapping(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "r",
        "version": "1",
        "instructions": "x",
        "items": [{"id": "a", "question": "q?"}],
    }
    return {**base, **over}


# ---------------------------------------------------------------- happy path


def test_load_valid_rubric(tmp_path: Path) -> None:
    r = load_rubric(_write(tmp_path, VALID))
    assert r.name == "explanation_quality"
    assert r.ids == ("facts_supported", "material_points", "clear_wording")
    assert r.item("facts_supported").severity is Severity.CRITICAL
    assert r.item("facts_supported").guidance == "An invented date counts as no."


def test_defaults_major_severity_and_fail_on_critical_major(tmp_path: Path) -> None:
    r = load_rubric(_write(tmp_path, VALID))
    assert r.item("material_points").severity is Severity.MAJOR
    assert r.fail_on == DEFAULT_FAIL_ON
    assert r.fails_verdict(Severity.CRITICAL)
    assert r.fails_verdict(Severity.MAJOR)
    assert not r.fails_verdict(Severity.MINOR)


def test_custom_fail_on() -> None:
    r = parse_rubric(_mapping(fail_on=["critical"]))
    assert r.fail_on == frozenset({Severity.CRITICAL})
    assert not r.fails_verdict(Severity.MAJOR)


def test_unquoted_numeric_version_is_kept_as_text(tmp_path: Path) -> None:
    r = load_rubric(_write(tmp_path, VALID.replace('version: "1.0"', "version: 1.0")))
    assert r.version == "1.0"
    assert parse_rubric(_mapping(version=2)).version == "2"


def test_rubric_is_immutable() -> None:
    r = parse_rubric(_mapping())
    with pytest.raises(ValueError):
        r.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------- failure paths


def test_duplicate_ids_rejected() -> None:
    items = [{"id": "a", "question": "q?"}, {"id": "a", "question": "again?"}]
    with pytest.raises(RubricError, match="duplicate rubric item ids"):
        parse_rubric(_mapping(items=items))


@pytest.mark.parametrize("bad_id", ["Facts", "1st", "has space", "dash-ed", ""])
def test_item_id_must_be_snake_case(bad_id: str) -> None:
    with pytest.raises(RubricError):
        parse_rubric(_mapping(items=[{"id": bad_id, "question": "q?"}]))


def test_unknown_severity_rejected() -> None:
    with pytest.raises(RubricError):
        parse_rubric(_mapping(items=[{"id": "a", "question": "q?", "severity": "fatal"}]))


def test_empty_items_rejected() -> None:
    with pytest.raises(RubricError):
        parse_rubric(_mapping(items=[]))


def test_empty_fail_on_rejected() -> None:
    with pytest.raises(RubricError):
        parse_rubric(_mapping(fail_on=[]))


def test_unknown_field_rejected_to_catch_typos() -> None:
    with pytest.raises(RubricError):
        parse_rubric(_mapping(fail_onn=["critical"]))
    with pytest.raises(RubricError):
        parse_rubric(_mapping(items=[{"id": "a", "question": "q?", "severty": "minor"}]))


@pytest.mark.parametrize("field", ["name", "version", "instructions"])
def test_required_text_fields(field: str) -> None:
    data = _mapping()
    del data[field]
    with pytest.raises(RubricError):
        parse_rubric(data)


def test_non_mapping_rejected() -> None:
    with pytest.raises(RubricError, match="must be a mapping"):
        parse_rubric(["not", "a", "mapping"])


def test_invalid_yaml_names_the_file(tmp_path: Path) -> None:
    p = _write(tmp_path, "name: [unclosed")
    with pytest.raises(RubricError, match="not valid YAML") as e:
        load_rubric(p)
    assert str(p) in str(e.value)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RubricError, match="cannot read rubric"):
        load_rubric(tmp_path / "nope.yaml")


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(RubricError, match="must be a mapping"):
        load_rubric(_write(tmp_path, ""))


def test_item_lookup_unknown_id() -> None:
    with pytest.raises(KeyError):
        parse_rubric(_mapping()).item("missing")


def test_rubric_error_is_value_error() -> None:
    assert issubclass(RubricError, ValueError)
    assert isinstance(parse_rubric(_mapping()), Rubric)
