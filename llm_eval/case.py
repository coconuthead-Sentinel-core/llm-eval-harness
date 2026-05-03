"""Core dataclasses: EvalCase, ModelOutput, GraderResult."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    """A single golden test case in an evaluation dataset."""
    case_id:        str
    prompt_inputs:  dict[str, Any] = field(default_factory=dict)
    expected:       str = ""                 # may be empty for open-ended cases
    metadata:       dict[str, Any] = field(default_factory=dict)
    rubric:         str = ""                 # used by LLMJudgeGrader

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("EvalCase.case_id required")


@dataclass
class ModelOutput:
    """Result of running ONE model on ONE case."""
    case_id:    str
    model_name: str
    output:     str
    latency_ms: float = 0.0
    metadata:   dict[str, Any] = field(default_factory=dict)


@dataclass
class GraderResult:
    """Result of running ONE grader on ONE ModelOutput."""
    case_id:     str
    model_name:  str
    grader_name: str
    score:       float                       # in [0.0, 1.0]
    passed:      bool
    explanation: str = ""
    metadata:    dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"GraderResult.score must be in [0,1], got {self.score}")
