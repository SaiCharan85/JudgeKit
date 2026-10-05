import pytest

from judgekit import Rubric, parse_rubric


@pytest.fixture
def rubric() -> Rubric:
    """Three items, one per severity tier: crit (critical), maj (major), mino (minor)."""
    return parse_rubric(
        {
            "name": "r",
            "version": "1",
            "instructions": "Review the reasoning, not the outcome.",
            "items": [
                {
                    "id": "crit",
                    "question": "Is every fact supported?",
                    "severity": "critical",
                    "guidance": "An invented date counts as no.",
                },
                {"id": "maj", "question": "Is every material point addressed?"},
                {"id": "mino", "question": "Is it clearly worded?", "severity": "minor"},
            ],
        }
    )
