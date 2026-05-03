"""EvalHarness — orchestrates dataset × runners × graders -> EvalReport."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from .aggregator import EvalReport, ScoreAggregator
from .case import EvalCase, GraderResult, ModelOutput
from .dataset import Dataset
from .graders.base import Grader
from .prompt import PromptTemplate
from .runners.base import ModelRunner


@dataclass
class EvalHarness:
    """End-to-end harness:

        dataset × runners × graders   ->   EvalReport

    Wire it once, call `run()` per evaluation cycle.

    Example:
        ds = Dataset([
            EvalCase(case_id="add-1", prompt_inputs={"a": 2, "b": 3},
                     expected="5"),
            EvalCase(case_id="add-2", prompt_inputs={"a": 10, "b": 7},
                     expected="17"),
        ])
        prompt = PromptTemplate("What is {a} + {b}?")
        report = EvalHarness(
            dataset=ds, prompt=prompt,
            runners=[CallableRunner(my_model, name="gpt-4o")],
            graders=[ExactMatchGrader()],
        ).run()
        print(report.overall_pass_rate)
    """
    dataset:    Dataset
    prompt:     PromptTemplate
    runners:    list[ModelRunner]
    graders:    list[Grader]
    aggregator: ScoreAggregator = field(default_factory=ScoreAggregator)

    def run(self) -> EvalReport:
        """Run every (case, runner) pair, then grade with every grader."""
        outputs: list[ModelOutput] = []
        results: list[GraderResult] = []

        for case in self.dataset:
            rendered = self.prompt.render(case.prompt_inputs)
            for runner in self.runners:
                t0 = time.perf_counter()
                raw = runner.run(rendered)
                t1 = time.perf_counter()
                output = ModelOutput(
                    case_id=case.case_id,
                    model_name=runner.name,
                    output=raw,
                    latency_ms=(t1 - t0) * 1000.0,
                )
                outputs.append(output)
                for grader in self.graders:
                    results.append(grader.grade(case, output))

        return self.aggregator.aggregate(results, outputs=outputs)
