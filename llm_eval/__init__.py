"""
llm_eval — LLM Evaluation Harness

Canon entry #25. Structured evaluation framework for LLM outputs.

Pipeline:
  Dataset (golden cases) -> PromptTemplate -> ModelRunner (one or many)
  -> Graders (exact, regex, semantic, llm-as-judge) -> Aggregator
  -> EvalReport (per-case + overall scores)

Five pluggable protocols ship with dependency-free reference backends
so the harness runs end-to-end with no external API keys.
"""
from __future__ import annotations

from .case import EvalCase, ModelOutput, GraderResult
from .dataset import Dataset, JsonlDataset
from .prompt import PromptTemplate
from .runners.base import ModelRunner
from .runners.echo import EchoRunner
from .runners.callable_runner import CallableRunner
from .graders.base import Grader
from .graders.exact import ExactMatchGrader
from .graders.regex import RegexGrader
from .graders.contains import ContainsGrader
from .graders.semantic import SemanticSimilarityGrader
from .graders.llm_judge import LLMJudgeGrader
from .aggregator import ScoreAggregator, EvalReport
from .harness import EvalHarness

__version__ = "1.0.0"

__all__ = [
    "EvalCase", "ModelOutput", "GraderResult",
    "Dataset", "JsonlDataset",
    "PromptTemplate",
    "ModelRunner", "EchoRunner", "CallableRunner",
    "Grader",
    "ExactMatchGrader", "RegexGrader", "ContainsGrader",
    "SemanticSimilarityGrader", "LLMJudgeGrader",
    "ScoreAggregator", "EvalReport",
    "EvalHarness",
]
