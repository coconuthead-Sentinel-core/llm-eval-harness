# LLM Evaluation Harness v1.0

> **Structured LLM evaluation. Datasets × prompts × runners × graders → report.**
> Five-grader battery (exact, regex, contains, semantic, LLM-as-judge),
> multi-model dispatch, JSONL datasets, no external API keys to run.

![Status](https://img.shields.io/badge/status-public-success)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-pytest-blue)

---

## What this is

The canonical reference implementation of the standard LLM evaluation
flow:

```
Dataset (golden cases)
        │
        ▼
   PromptTemplate ── render ──> prompt string
        │
        ▼
   ModelRunner(s) ── run() ──> ModelOutput(s)
        │
        ▼
   Grader(s) ── grade() ──> GraderResult(s)
        │
        ▼
   ScoreAggregator ──> EvalReport
                       (overall + per-model + per-grader)
```

Five pluggable protocols ship with dependency-free reference backends.
The package runs end-to-end on a fresh machine **with no external API
keys**. Real adapters (OpenAI / Anthropic / local llama.cpp) plug into
the same `ModelRunner` protocol.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from llm_eval import (
    Dataset, EvalCase, PromptTemplate,
    EvalHarness, CallableRunner,
    ExactMatchGrader, ContainsGrader, RegexGrader,
)

ds = Dataset([
    EvalCase("add-1", prompt_inputs={"a": 2,  "b": 3},  expected="5"),
    EvalCase("add-2", prompt_inputs={"a": 10, "b": 7},  expected="17"),
])

def my_model(prompt: str) -> str:
    # In production this is an OpenAI / Anthropic / local LLM call
    import re
    m = re.search(r"What is (\d+) \+ (\d+)\?", prompt)
    return str(int(m.group(1)) + int(m.group(2))) if m else "?"

report = EvalHarness(
    dataset = ds,
    prompt  = PromptTemplate("What is {a} + {b}?"),
    runners = [CallableRunner(my_model, name="adder-v1")],
    graders = [ExactMatchGrader()],
).run()

print(f"Overall pass rate: {report.overall_pass_rate:.1%}")
print(f"Per-model: {report.per_model}")
```

## Multi-model bake-off

```python
report = EvalHarness(
    dataset = ds,
    prompt  = PromptTemplate("What is {a} + {b}?"),
    runners = [
        CallableRunner(model_v1, name="v1"),
        CallableRunner(model_v2, name="v2"),
        CallableRunner(model_v3, name="v3"),
    ],
    graders = [ExactMatchGrader(), RegexGrader(r"\d+")],
).run()

# report.per_model gives a per-model leaderboard you can publish
```

## Five built-in graders

| Grader | What it does | Use when |
|---|---|---|
| `ExactMatchGrader`        | Strict equality (normalized) | Single-token answers, known-good labels |
| `RegexGrader`             | Pattern match in output | Output should contain a date/number/format |
| `ContainsGrader`          | Substring set match (all/any) | Output should mention specific terms |
| `SemanticSimilarityGrader`| Jaccard token-set similarity | Open-ended answers, prose grading |
| `LLMJudgeGrader`          | Wraps another model as judge | Rubric-based grading of unstructured text |

`LLMJudgeGrader` accepts any `ModelRunner` as the judge — you can use a
stronger model to grade a weaker one's outputs (the standard 2026
production pattern). The reference parser expects the judge to emit a
line of the form `SCORE: 0.85` and `EXPLANATION: ...`.

## Why this design

- **Reproducible:** golden datasets persist as JSONL, models stub-able as `EchoRunner`, no test depends on a paid API.
- **Multi-model first-class:** the harness runs the whole dataset against EVERY runner so you can A/B (or A/B/C/D) any number of models in one shot.
- **Multi-grader first-class:** every output is graded by EVERY grader, so you can stack `ExactMatch + LLMJudge + Regex` and read all three signals.
- **Pluggable judges:** the grader that uses an LLM is just another runner — your judge can be the same model, a stronger model, or a local model.
- **Honest scores:** every `GraderResult` carries an `explanation` field; never just a number.

## Testing

```bash
pytest -v
```

## Project structure

```
LLM Evaluation Harness/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── llm_eval/
│   ├── __init__.py
│   ├── case.py                ← EvalCase, ModelOutput, GraderResult
│   ├── dataset.py             ← Dataset + JsonlDataset
│   ├── prompt.py              ← PromptTemplate
│   ├── aggregator.py          ← ScoreAggregator + EvalReport
│   ├── harness.py             ← EvalHarness
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── base.py            ← ModelRunner protocol
│   │   ├── echo.py            ← EchoRunner
│   │   └── callable_runner.py ← CallableRunner
│   └── graders/
│       ├── __init__.py
│       ├── base.py            ← Grader protocol
│       ├── exact.py           ← ExactMatchGrader
│       ├── regex.py           ← RegexGrader
│       ├── contains.py        ← ContainsGrader
│       ├── semantic.py        ← SemanticSimilarityGrader
│       └── llm_judge.py       ← LLMJudgeGrader
├── tests/
│   ├── test_case.py
│   ├── test_dataset.py
│   ├── test_prompt.py
│   ├── test_runners.py
│   ├── test_graders.py
│   ├── test_aggregator.py
│   └── test_harness.py
└── docs/
```

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Shannon Brian Kelley** — AI Orchestrator Architect.
Co-authored with Claude AI (Anthropic) under file-system-bound persona
protocol; co-creator role: **"Archivist of Wisdom"**.

Canon entry **#25** in the architect's portfolio.
