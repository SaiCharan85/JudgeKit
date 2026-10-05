# JudgeKit

Domain-agnostic toolkit for **LLM-as-judge** evaluation of reasoning: YAML rubrics with severity
tiers, a structured-output judge runner, multi-judge panels, planted-error evals, calibration
(Cohen's kappa, catch rate per error type) and bias tests (position, verbosity, politeness).

A judge grades the *explanation* (is it faithful, complete and consistent with the evidence), never
whether the decision is right. JudgeKit makes no LLM calls itself: you adapt your own client to its
`StructuredLLM` protocol, and bring your own domain rubrics and planted errors.

```bash
uv sync
uv run pytest
```

## What's inside

| Module | Purpose |
|---|---|
| `rubric` | YAML rubrics: yes = pass questions with `critical` / `major` / `minor` severity tiers |
| `judge` | `Judge` protocol, `Verdict`, `score()` (code, not the model, decides pass/fail) |
| `llm` | `StructuredLLM` / `TextLLM` protocols; `TextStructuredLLM` adds schema + retry |
| `runner` | `LLMJudge`: rubric prompt (static, cache-friendly) + structured verdict |
| `panel` | `JudgePanel`: any / majority / all votes, model-family diversity, early stopping |
| `planted` | Plant known errors in correct samples; catch and false-alarm rates; resumable runs |
| `mutators` | Generic text mutators: number swap, dropped / inserted sentence, phrase swap, negation |
| `calibration` | Cohen's kappa vs human labels, clustered bootstrap CIs, paired judge comparison |
| `bias` | Position, verbosity, politeness and repeat probes: flip rate and net shift |
| `testing` | Scripted fake LLMs for tests (no network, no keys) |

```python
from judgekit import LLMJudge, load_rubric

judge = LLMJudge(my_structured_llm, load_rubric(Path("rubrics/explanation.yaml")))
verdict = judge.evaluate(case_text, avoid_families=frozenset({"family_that_wrote_it"}))
verdict.passed, [f.item_id for f in verdict.failures]
```

Install from another project:

```bash
uv add --editable ../JudgeKit                                              # local development
uv add "judgekit @ git+https://github.com/SaiCharan85/JudgeKit@v0.1.0"     # pinned release
```

MIT licensed.
