"""JudgeKit: domain-agnostic LLM-as-judge evaluation of reasoning."""

from judgekit.judge import Answer, ItemResult, Judge, JudgeResponse, Verdict, score
from judgekit.rubric import (
    DEFAULT_FAIL_ON,
    Rubric,
    RubricError,
    RubricItem,
    Severity,
    load_rubric,
    parse_rubric,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "DEFAULT_FAIL_ON",
    "Answer",
    "ItemResult",
    "Judge",
    "JudgeResponse",
    "Rubric",
    "RubricError",
    "RubricItem",
    "Severity",
    "Verdict",
    "__version__",
    "load_rubric",
    "parse_rubric",
    "score",
]
