"""ScoreAggregator + EvalReport — collect grader results into a final report."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .case import GraderResult, ModelOutput


@dataclass
class EvalReport:
    """Final structured report from an evaluation run."""
    n_cases:    int = 0
    n_models:   int = 0
    n_graders:  int = 0
    overall_score:    float = 0.0
    overall_pass_rate: float = 0.0
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)
    per_grader: dict[str, dict[str, float]] = field(default_factory=dict)
    case_results: list[GraderResult] = field(default_factory=list)
    outputs:      list[ModelOutput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases":           self.n_cases,
            "n_models":          self.n_models,
            "n_graders":         self.n_graders,
            "overall_score":     self.overall_score,
            "overall_pass_rate": self.overall_pass_rate,
            "per_model":         self.per_model,
            "per_grader":        self.per_grader,
            "case_results": [
                {"case_id": r.case_id, "model": r.model_name,
                 "grader": r.grader_name, "score": r.score,
                 "passed": r.passed, "explanation": r.explanation}
                for r in self.case_results
            ],
        }


class ScoreAggregator:
    """Aggregates per-case grader results into per-model + per-grader stats."""

    def aggregate(self, results: list[GraderResult],
                  outputs: list[ModelOutput] | None = None) -> EvalReport:
        if not results:
            return EvalReport(case_results=[], outputs=list(outputs or []))

        models  = {r.model_name  for r in results}
        graders = {r.grader_name for r in results}
        cases   = {r.case_id     for r in results}

        # per-model: avg score + pass-rate across all (case, grader) for this model
        per_model: dict[str, dict[str, float]] = {}
        for m in models:
            ms = [r for r in results if r.model_name == m]
            per_model[m] = {
                "avg_score": sum(r.score for r in ms) / len(ms),
                "pass_rate": sum(1 for r in ms if r.passed) / len(ms),
                "n":         len(ms),
            }

        # per-grader: avg score + pass-rate across all (case, model) for this grader
        per_grader: dict[str, dict[str, float]] = {}
        for g in graders:
            gs = [r for r in results if r.grader_name == g]
            per_grader[g] = {
                "avg_score": sum(r.score for r in gs) / len(gs),
                "pass_rate": sum(1 for r in gs if r.passed) / len(gs),
                "n":         len(gs),
            }

        overall_score    = sum(r.score for r in results) / len(results)
        overall_pass     = sum(1 for r in results if r.passed) / len(results)

        return EvalReport(
            n_cases=len(cases),
            n_models=len(models),
            n_graders=len(graders),
            overall_score=overall_score,
            overall_pass_rate=overall_pass,
            per_model=per_model,
            per_grader=per_grader,
            case_results=list(results),
            outputs=list(outputs or []),
        )
