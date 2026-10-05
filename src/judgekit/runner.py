"""LLM judge runner: renders a rubric into a prompt, asks for a `JudgeResponse`, scores it.

The system prompt depends only on the rubric (static), and the case goes in the user message, so
providers that cache prompt prefixes reuse it across cases. Severity tiers are not shown to the
model: it answers every question on its merits, and code applies the tiers.
"""

from judgekit.judge import JudgeResponse, Verdict, score, unavailable
from judgekit.llm import LLMCallError, StructuredLLM, StructuredOutputError
from judgekit.rubric import Rubric


def render_system(rubric: Rubric) -> str:
    lines = [
        rubric.instructions.strip(),
        "",
        "Answer every question below with yes or no, exactly once per id. A yes always means the "
        "reasoning passes that check. When you answer no, give a short note saying what is wrong.",
        "",
    ]
    for item in rubric.items:
        lines.append(f"- {item.id}: {item.question}")
        if item.guidance:
            lines.append(f"  ({item.guidance})")
    return "\n".join(lines)


class LLMJudge:
    """A `Judge` backed by one structured-output LLM."""

    def __init__(self, llm: StructuredLLM, rubric: Rubric, name: str = "llm_judge") -> None:
        self.llm = llm
        self.rubric = rubric
        self.name = name
        self.system = render_system(rubric)

    @property
    def judge_id(self) -> str:
        return self.name

    def evaluate(self, case: str, *, avoid_families: frozenset[str] = frozenset()) -> Verdict:
        if not case.strip():
            raise ValueError("cannot judge an empty case")
        try:
            res = self.llm.complete(self.system, case, JudgeResponse, avoid_families=avoid_families)
        except StructuredOutputError as exc:
            return unavailable(self.rubric, self.name, f"invalid judge output: {exc}", exc.calls)
        except LLMCallError as exc:
            return unavailable(self.rubric, self.name, f"judge call failed: {exc}")
        verdict = score(self.rubric, res.value, self.name)
        return verdict.model_copy(update={"calls": res.calls})
