"""ContainsGrader — pass iff output contains all required substrings."""
from __future__ import annotations

from ..case import EvalCase, GraderResult, ModelOutput


class ContainsGrader:
    """Score = fraction of `must_contain` substrings present in output.

    With `mode="all"`, passed=True iff ALL substrings are present.
    With `mode="any"`, passed=True iff at least one is present.
    """
    name = "contains"

    def __init__(self, must_contain: list[str], *,
                 case_sensitive: bool = False, mode: str = "all"):
        if mode not in ("all", "any"):
            raise ValueError("mode must be 'all' or 'any'")
        if not must_contain:
            raise ValueError("must_contain cannot be empty")
        self.must_contain = list(must_contain)
        self.case_sensitive = case_sensitive
        self.mode = mode

    def grade(self, case: EvalCase, output: ModelOutput) -> GraderResult:
        haystack = output.output if self.case_sensitive else output.output.lower()
        needles = self.must_contain if self.case_sensitive else \
                  [n.lower() for n in self.must_contain]
        hits = [n for n in needles if n in haystack]
        score = len(hits) / len(needles)
        if self.mode == "all":
            passed = (len(hits) == len(needles))
        else:  # any
            passed = (len(hits) >= 1)
        return GraderResult(
            case_id=case.case_id,
            model_name=output.model_name,
            grader_name=self.name,
            score=score,
            passed=passed,
            explanation=f"matched {len(hits)}/{len(needles)} substrings ({self.mode})",
            metadata={"must_contain": self.must_contain,
                      "matched": hits, "mode": self.mode},
        )
