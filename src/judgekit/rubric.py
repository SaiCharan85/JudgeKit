"""Rubrics: yes/no questions (yes always means "passes"), each with a severity tier.

A rubric is plain YAML so domain experts can edit it without touching code:

    name: explanation_quality
    version: "1.0"
    instructions: You review the written reasoning behind a decision ...
    fail_on: [critical, major]        # optional; these tiers fail the verdict (default)
    items:
      - id: facts_supported
        question: Is every fact the explanation relies on present in the evidence?
        severity: critical            # optional; default major
        guidance: An invented date or amount counts as "no".   # optional

Severity tiers: one "no" on a tier listed in `fail_on` fails the whole verdict; other tiers are
recorded (and counted in evals) but do not fail it.
"""

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ITEM_ID_PATTERN = r"^[a-z][a-z0-9_]*$"


class Severity(StrEnum):
    CRITICAL = "critical"  # e.g. an invented fact: always disqualifying
    MAJOR = "major"  # e.g. a material point ignored
    MINOR = "minor"  # e.g. vague wording: tracked, not disqualifying by default


DEFAULT_FAIL_ON = frozenset({Severity.CRITICAL, Severity.MAJOR})


class RubricError(ValueError):
    """A rubric file or mapping that cannot be loaded."""


class RubricItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=ITEM_ID_PATTERN, max_length=64)
    question: str = Field(min_length=1)  # phrased so that "yes" = passes
    severity: Severity = Severity.MAJOR
    guidance: str = ""  # what counts as "no", shown to the judge


class Rubric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    items: tuple[RubricItem, ...] = Field(min_length=1)
    fail_on: frozenset[Severity] = Field(default=DEFAULT_FAIL_ON, min_length=1)

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, v: Any) -> Any:
        # YAML reads an unquoted 1.0 as a float; a version is a label, so keep it as text.
        return str(v) if isinstance(v, int | float) and not isinstance(v, bool) else v

    @model_validator(mode="after")
    def _unique_ids(self) -> "Rubric":
        dupes = sorted(k for k, n in Counter(i.id for i in self.items).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate rubric item ids: {dupes}")
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(i.id for i in self.items)

    def item(self, item_id: str) -> RubricItem:
        for i in self.items:
            if i.id == item_id:
                return i
        raise KeyError(f"rubric {self.name!r} has no item {item_id!r}")

    def fails_verdict(self, severity: Severity) -> bool:
        return severity in self.fail_on


def parse_rubric(data: Any, source: str = "<mapping>") -> Rubric:
    """Validate an already-parsed mapping (e.g. from YAML) into a `Rubric`."""
    if not isinstance(data, dict):
        raise RubricError(f"{source}: a rubric must be a mapping, got {type(data).__name__}")
    try:
        return Rubric.model_validate(data)
    except ValidationError as e:
        raise RubricError(f"{source}: invalid rubric\n{e}") from e


def load_rubric(path: Path) -> Rubric:
    """Load a rubric from a YAML file; every failure is a `RubricError` naming the file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise RubricError(f"{path}: cannot read rubric ({e})") from e
    except yaml.YAMLError as e:
        raise RubricError(f"{path}: not valid YAML ({e})") from e
    return parse_rubric(data, str(path))
