"""LLMJudgeGrader — uses ANOTHER ModelRunner as the judge.

LLM-as-judge is the dominant pattern for grading open-ended outputs in
2026 production eval pipelines. The judge is given a rubric and the
candidate output and is asked to produce a numeric score.

Reference parser expects the judge's output to contain a token like
``SCORE: 0.7`` or ``score=0.85``. Customize via the `score_re`
constructor argument.
"""
from __future__ import annotations

import re
from typing import Any

from ..case import EvalCase, GraderResult, ModelOutput
from ..runners.base import ModelRunner


_DEFAULT_PROMPT = (
    "You are an evaluation judge.\n\n"
    "Rubric:\n{rubric}\n\n"
    "Candidate output:\n{output}\n\n"
    "Expected (may be empty for open-ended cases):\n{expected}\n\n"
    "Reply with a single line in this exact form:\n"
    "SCORE: <number between 0.0 and 1.0>\n"
    "EXPLANATION: <one short sentence>\n"
)

_DEFAULT_SCORE_RE = re.compile(
    r"score\s*[:=]\s*([0-1](?:\.\d+)?|\.\d+)", re.IGNORECASE)
_DEFAULT_EXPL_RE = re.compile(
    r"explanation\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)


class LLMJudgeGrader:
    """Wraps a ModelRunner as a grading judge."""
    name = "llm_judge"

    def __init__(self, judge: ModelRunner, *,
                 threshold: float = 0.7,
                 prompt_template: str = _DEFAULT_PROMPT,
                 score_re: re.Pattern[str] | None = None,
                 explanation_re: re.Pattern[str] | None = None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        self.judge = judge
        self.threshold = float(threshold)
        self.prompt_template = prompt_template
        self.score_re = score_re or _DEFAULT_SCORE_RE
        self.explanation_re = explanation_re or _DEFAULT_EXPL_RE

    def grade(self, case: EvalCase, output: ModelOutput) -> GraderResult:
        rubric = case.rubric or "Score how well the output answers the case."
        prompt = self.prompt_template.format(
            rubric=rubric,
            output=output.output,
            expected=case.expected,
        )
        judge_text = self.judge.run(prompt)
        score, expl = self._parse(judge_text)
        passed = score >= self.threshold
        return GraderResult(
            case_id=case.case_id,
            model_name=output.model_name,
            grader_name=self.name,
            score=score,
            passed=passed,
            explanation=expl or "(no explanation parsed)",
            metadata={"judge": self.judge.name,
                      "threshold": self.threshold,
                      "raw_judge_output": judge_text[:500]},
        )

    def _parse(self, text: str) -> tuple[float, str]:
        m = self.score_re.search(text or "")
        score = 0.0
        if m:
            try:
                score = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                score = 0.0
        m2 = self.explanation_re.search(text or "")
        expl = m2.group(1).strip() if m2 else ""
        return score, expl
