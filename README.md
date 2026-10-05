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

Install from another project:

```bash
uv add --editable ../JudgeKit                                              # local development
uv add "judgekit @ git+https://github.com/SaiCharan85/JudgeKit@v0.1.0"     # pinned release
```

MIT licensed.
