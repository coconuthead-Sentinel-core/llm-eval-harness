"""Tests for EvalCase, ModelOutput, GraderResult."""
import pytest

from llm_eval import EvalCase, ModelOutput, GraderResult


class TestEvalCase:
    def test_minimal_construct(self):
        c = EvalCase(case_id="c1")
        assert c.case_id == "c1"
        assert c.prompt_inputs == {}

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError):
            EvalCase(case_id="")

    def test_all_fields(self):
        c = EvalCase(case_id="c1", prompt_inputs={"a": 1},
                     expected="x", metadata={"src": "wiki"},
                     rubric="grade fairly")
        assert c.expected == "x"
        assert c.rubric == "grade fairly"


class TestGraderResult:
    def test_score_must_be_in_range(self):
        with pytest.raises(ValueError):
            GraderResult(case_id="c", model_name="m", grader_name="g",
                         score=1.5, passed=True)
        with pytest.raises(ValueError):
            GraderResult(case_id="c", model_name="m", grader_name="g",
                         score=-0.1, passed=False)

    def test_construct_valid(self):
        r = GraderResult(case_id="c", model_name="m", grader_name="g",
                         score=0.8, passed=True, explanation="ok")
        assert r.score == 0.8
        assert r.passed
