"""RegexGrader — pass iff output matches a regex pattern."""
from __future__ import annotations

import re

from ..case import EvalCase, GraderResult, ModelOutput


class RegexGrader:
    """Pass iff `pattern.search(output.output)` returns a match.

    Score is 1.0 on match, 0.0 otherwise (binary). Use the metadata
    field on EvalCase or pass `pattern` at grader construction.
    """
    name = "regex_match"

    def __init__(self, pattern: str, *,
                 flags: int = re.IGNORECASE | re.MULTILINE):
        self._pattern_str = pattern
        self._pattern = re.compile(pattern, flags=flags)

    def grade(self, case: EvalCase, output: ModelOutput) -> GraderResult:
        m = self._pattern.search(output.output)
        passed = m is not None
        return GraderResult(
            case_id=case.case_id,
            model_name=output.model_name,
            grader_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation=(f"matched {m.group(0)!r}" if m
                         else f"no match for /{self._pattern_str}/"),
            metadata={"pattern": self._pattern_str},
        )
