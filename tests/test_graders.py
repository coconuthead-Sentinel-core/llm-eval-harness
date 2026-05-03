"""Tests for all five graders."""
import pytest

from llm_eval import (
    EvalCase, ModelOutput, Grader,
    ExactMatchGrader, RegexGrader, ContainsGrader,
    SemanticSimilarityGrader, LLMJudgeGrader,
    EchoRunner, CallableRunner,
)


def _output(text: str = "5", *, case_id="c", model="m") -> ModelOutput:
    return ModelOutput(case_id=case_id, model_name=model, output=text)


def _case(expected: str = "5", *, case_id="c", rubric: str = "") -> EvalCase:
    return EvalCase(case_id=case_id, expected=expected, rubric=rubric)


class TestExactMatchGrader:
    def test_protocol(self):
        assert isinstance(ExactMatchGrader(), Grader)

    def test_match(self):
        r = ExactMatchGrader().grade(_case("5"), _output("5"))
        assert r.passed and r.score == 1.0

    def test_normalized_case(self):
        r = ExactMatchGrader().grade(_case("Yes"), _output("yes "))
        assert r.passed

    def test_no_normalize(self):
        g = ExactMatchGrader(normalize=False)
        r = g.grade(_case("Yes"), _output("yes"))
        assert not r.passed

    def test_empty_expected_never_passes(self):
        r = ExactMatchGrader().grade(_case(""), _output(""))
        assert not r.passed
        assert r.score == 0.0


class TestRegexGrader:
    def test_protocol(self):
        assert isinstance(RegexGrader(r"x"), Grader)

    def test_match(self):
        r = RegexGrader(r"\b\d+\b").grade(_case(), _output("the answer is 42"))
        assert r.passed and r.score == 1.0

    def test_no_match(self):
        r = RegexGrader(r"\bfoo\b").grade(_case(), _output("bar baz"))
        assert not r.passed and r.score == 0.0


class TestContainsGrader:
    def test_protocol(self):
        assert isinstance(ContainsGrader(["x"]), Grader)

    def test_all_must_pass(self):
        g = ContainsGrader(["python", "fastapi"], mode="all")
        r = g.grade(_case(), _output("python and fastapi rock"))
        assert r.passed and r.score == 1.0

    def test_all_partial(self):
        g = ContainsGrader(["python", "rust"], mode="all")
        r = g.grade(_case(), _output("python and fastapi"))
        assert not r.passed
        assert r.score == 0.5

    def test_any_passes_on_one_hit(self):
        g = ContainsGrader(["rust", "python"], mode="any")
        r = g.grade(_case(), _output("python rocks"))
        assert r.passed
        assert r.score == 0.5

    def test_case_sensitive_off_by_default(self):
        g = ContainsGrader(["PYTHON"])
        r = g.grade(_case(), _output("python rocks"))
        assert r.passed

    def test_empty_must_contain_rejected(self):
        with pytest.raises(ValueError):
            ContainsGrader([])

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            ContainsGrader(["x"], mode="some")


class TestSemanticSimilarityGrader:
    def test_protocol(self):
        assert isinstance(SemanticSimilarityGrader(), Grader)

    def test_identical_text_perfect_score(self):
        g = SemanticSimilarityGrader()
        r = g.grade(_case("python is a programming language"),
                    _output("python is a programming language"))
        assert r.score >= 0.9

    def test_orthogonal_text_low_score(self):
        g = SemanticSimilarityGrader(threshold=0.5)
        r = g.grade(_case("python programming language"),
                    _output("the cat sat on the mat colorful blanket"))
        assert r.score < 0.3
        assert not r.passed

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            SemanticSimilarityGrader(threshold=1.5)


class TestLLMJudgeGrader:
    def test_protocol(self):
        judge = EchoRunner()
        assert isinstance(LLMJudgeGrader(judge), Grader)

    def test_parses_score_format(self):
        # Build a judge that always returns SCORE: 0.85
        def judge_fn(prompt: str) -> str:
            return "SCORE: 0.85\nEXPLANATION: looks good"
        judge = CallableRunner(judge_fn, name="fake_judge")
        g = LLMJudgeGrader(judge, threshold=0.7)
        r = g.grade(_case(rubric="grade fairly"), _output("anything"))
        assert r.score == 0.85
        assert r.passed
        assert "looks good" in r.explanation

    def test_threshold_blocks_low_score(self):
        judge = CallableRunner(lambda p: "SCORE: 0.30", name="judge")
        g = LLMJudgeGrader(judge, threshold=0.7)
        r = g.grade(_case(), _output(""))
        assert r.score == 0.30
        assert not r.passed

    def test_unparseable_score_defaults_to_zero(self):
        judge = CallableRunner(lambda p: "no score here", name="judge")
        g = LLMJudgeGrader(judge)
        r = g.grade(_case(), _output(""))
        assert r.score == 0.0
        assert not r.passed

    def test_clamps_score_above_one(self):
        judge = CallableRunner(lambda p: "SCORE: 1.5", name="judge")
        g = LLMJudgeGrader(judge, threshold=0.7)
        r = g.grade(_case(), _output(""))
        assert r.score == 1.0
