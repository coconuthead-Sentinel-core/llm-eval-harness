"""SemanticSimilarityGrader — token-overlap based proxy for semantic
similarity. Real production stacks plug in sentence-transformers or an
embedding model via the same `name`/`grade` contract.

Score = Jaccard similarity over the cleaned token sets, in [0, 1].
"""
from __future__ import annotations

import re

from ..case import EvalCase, GraderResult, ModelOutput


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "of", "to",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "for", "with", "as", "by", "this", "that",
    "it", "its",
}


def _toks(s: str) -> set[str]:
    out = set()
    for t in re.findall(r"[A-Za-z][A-Za-z0-9_]+", s.lower()):
        if t in _STOPWORDS:
            continue
        out.add(t)
    return out


class SemanticSimilarityGrader:
    """Jaccard token-set similarity with a configurable pass threshold."""
    name = "semantic_similarity"

    def __init__(self, *, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        self.threshold = float(threshold)

    def grade(self, case: EvalCase, output: ModelOutput) -> GraderResult:
        a = _toks(case.expected)
        b = _toks(output.output)
        if not a and not b:
            score = 1.0
        elif not a or not b:
            score = 0.0
        else:
            inter = len(a & b)
            union = len(a | b)
            score = inter / union if union else 0.0
        passed = score >= self.threshold
        return GraderResult(
            case_id=case.case_id,
            model_name=output.model_name,
            grader_name=self.name,
            score=score,
            passed=passed,
            explanation=f"Jaccard={score:.3f} threshold={self.threshold}",
            metadata={"threshold": self.threshold},
        )
