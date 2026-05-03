"""ExactMatchGrader — strict equality (1.0) or zero (0.0)."""
from __future__ import annotations

from ..case import EvalCase, GraderResult, ModelOutput


class ExactMatchGrader:
    """Pass iff output (after normalization) equals expected."""
    name = "exact_match"

    def __init__(self, *, normalize: bool = True):
        self.normalize = normalize

    def _norm(self, s: str) -> str:
        if not self.normalize:
            return s
        return s.strip().lower()

    def grade(self, case: EvalCase, output: ModelOutput) -> GraderResult:
        expected = self._norm(case.expected)
        actual   = self._norm(output.output)
        passed = (expected == actual) and bool(expected)
        return GraderResult(
            case_id=case.case_id,
            model_name=output.model_name,
            grader_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation=("exact match" if passed
                         else f"expected {expected!r} got {actual!r}"),
            metadata={"normalized": self.normalize},
        )
