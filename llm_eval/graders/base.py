"""Grader protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..case import EvalCase, GraderResult, ModelOutput


@runtime_checkable
class Grader(Protocol):
    """Score one ModelOutput against the case's expected/rubric.

    Implementations return a GraderResult with a score in [0,1] and
    a passed/failed verdict per a per-grader threshold.
    """
    name: str

    def grade(self, case: EvalCase, output: ModelOutput) -> GraderResult: ...
