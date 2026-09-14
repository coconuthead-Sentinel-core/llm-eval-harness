"""
llm_eval — LLM Evaluation Harness

Structured evaluation framework for LLM outputs.

Pipeline:
  Dataset (golden cases) -> PromptTemplate -> ModelRunner (one or many)
  -> Graders (exact, contains, regex, jaccard, llm-as-judge) -> Aggregator
  -> EvalReport (per-case + overall scores)

Five built-in graders and two pluggable protocols (Grader, ModelRunner)
ship with dependency-free reference backends so the harness runs
end-to-end with no external API keys.
"""
from __future__ import annotations

from .aggregator import EvalReport, ScoreAggregator
from .case import EvalCase, GraderResult, ModelOutput
from .dataset import Dataset, JsonlDataset
from .graders.base import Grader
from .graders.contains import ContainsGrader
from .graders.exact import ExactMatchGrader
from .graders.jaccard import JaccardSimilarityGrader
from .graders.llm_judge import LLMJudgeGrader
from .graders.regex import RegexGrader
from .harness import EvalHarness
from .prompt import PromptTemplate
from .runners.base import ModelRunner
from .runners.callable_runner import CallableRunner
from .runners.echo import EchoRunner

__version__ = "0.1.0"

__all__ = [
    "EvalCase", "ModelOutput", "GraderResult",
    "Dataset", "JsonlDataset",
    "PromptTemplate",
    "ModelRunner", "EchoRunner", "CallableRunner",
    "Grader",
    "ExactMatchGrader", "RegexGrader", "ContainsGrader",
    "JaccardSimilarityGrader", "LLMJudgeGrader",
    "ScoreAggregator", "EvalReport",
    "EvalHarness",
]
