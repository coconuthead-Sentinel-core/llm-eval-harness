"""End-to-end harness tests."""
from llm_eval import (
    Dataset, EvalCase, PromptTemplate,
    EvalHarness,
    EchoRunner, CallableRunner,
    ExactMatchGrader, ContainsGrader, RegexGrader,
)


class TestEvalHarness:
    def _build_dataset(self):
        return Dataset([
            EvalCase("add-1", prompt_inputs={"a": 2, "b": 3}, expected="5"),
            EvalCase("add-2", prompt_inputs={"a": 10, "b": 7}, expected="17"),
            EvalCase("add-3", prompt_inputs={"a": 0, "b": 0}, expected="0"),
        ])

    def _add_runner(self, name="adder"):
        # Parses "What is X + Y?" and returns the sum
        import re
        def fn(prompt: str) -> str:
            m = re.search(r"What is (\d+) \+ (\d+)\?", prompt)
            if not m:
                return "?"
            return str(int(m.group(1)) + int(m.group(2)))
        return CallableRunner(fn, name=name)

    def test_full_run_all_pass(self):
        h = EvalHarness(
            dataset=self._build_dataset(),
            prompt=PromptTemplate("What is {a} + {b}?"),
            runners=[self._add_runner()],
            graders=[ExactMatchGrader()],
        )
        report = h.run()
        assert report.n_cases == 3
        assert report.n_models == 1
        assert report.overall_pass_rate == 1.0
        assert report.per_model["adder"]["pass_rate"] == 1.0

    def test_multiple_models_in_one_run(self):
        h = EvalHarness(
            dataset=self._build_dataset(),
            prompt=PromptTemplate("What is {a} + {b}?"),
            runners=[
                self._add_runner(name="good_adder"),
                CallableRunner(lambda p: "42", name="constant_42"),
            ],
            graders=[ExactMatchGrader()],
        )
        report = h.run()
        assert report.n_models == 2
        assert report.per_model["good_adder"]["pass_rate"] == 1.0
        assert report.per_model["constant_42"]["pass_rate"] == 0.0

    def test_multiple_graders(self):
        h = EvalHarness(
            dataset=self._build_dataset(),
            prompt=PromptTemplate("What is {a} + {b}?"),
            runners=[self._add_runner()],
            graders=[
                ExactMatchGrader(),
                RegexGrader(r"\d+"),
                ContainsGrader(["5"], mode="any"),  # only case 1 contains "5"
            ],
        )
        report = h.run()
        assert report.n_graders == 3
        # 3 cases × 1 model × 3 graders = 9 results
        assert len(report.case_results) == 9

    def test_outputs_collected_with_latency(self):
        h = EvalHarness(
            dataset=self._build_dataset(),
            prompt=PromptTemplate("What is {a} + {b}?"),
            runners=[self._add_runner()],
            graders=[ExactMatchGrader()],
        )
        report = h.run()
        assert len(report.outputs) == 3
        for out in report.outputs:
            assert out.latency_ms >= 0.0

    def test_echo_runner_smoke(self):
        h = EvalHarness(
            dataset=Dataset([EvalCase("c1", prompt_inputs={"x": "hello"},
                                       expected="hello")]),
            prompt=PromptTemplate("{x}"),
            runners=[EchoRunner()],
            graders=[ExactMatchGrader()],
        )
        report = h.run()
        assert report.overall_pass_rate == 1.0

    def test_to_dict_round_trip(self):
        h = EvalHarness(
            dataset=self._build_dataset(),
            prompt=PromptTemplate("What is {a} + {b}?"),
            runners=[self._add_runner()],
            graders=[ExactMatchGrader()],
        )
        d = h.run().to_dict()
        assert d["n_cases"] == 3
        assert d["overall_pass_rate"] == 1.0
        assert "case_results" in d
