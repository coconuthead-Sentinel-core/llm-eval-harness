"""Tests for ScoreAggregator and EvalReport."""
from llm_eval import ScoreAggregator, EvalReport, GraderResult, ModelOutput


def _r(case, model, grader, score, passed):
    return GraderResult(case_id=case, model_name=model, grader_name=grader,
                        score=score, passed=passed)


class TestScoreAggregator:
    def test_empty_input_returns_empty_report(self):
        rep = ScoreAggregator().aggregate([])
        assert rep.n_cases == 0
        assert rep.n_models == 0
        assert rep.overall_score == 0.0

    def test_aggregates_per_model(self):
        results = [
            _r("c1", "model_a", "exact", 1.0, True),
            _r("c2", "model_a", "exact", 0.0, False),
            _r("c1", "model_b", "exact", 1.0, True),
            _r("c2", "model_b", "exact", 1.0, True),
        ]
        rep = ScoreAggregator().aggregate(results)
        assert rep.n_cases == 2
        assert rep.n_models == 2
        assert rep.per_model["model_a"]["pass_rate"] == 0.5
        assert rep.per_model["model_b"]["pass_rate"] == 1.0

    def test_aggregates_per_grader(self):
        results = [
            _r("c1", "m", "exact",   1.0, True),
            _r("c1", "m", "regex",   0.0, False),
            _r("c1", "m", "contains",0.5, False),
        ]
        rep = ScoreAggregator().aggregate(results)
        assert rep.n_graders == 3
        assert rep.per_grader["exact"]["pass_rate"] == 1.0
        assert rep.per_grader["regex"]["pass_rate"] == 0.0
        assert rep.per_grader["contains"]["avg_score"] == 0.5

    def test_overall_pass_rate(self):
        results = [
            _r("c1", "m", "g", 1.0, True),
            _r("c2", "m", "g", 0.0, False),
            _r("c3", "m", "g", 1.0, True),
            _r("c4", "m", "g", 1.0, True),
        ]
        rep = ScoreAggregator().aggregate(results)
        assert rep.overall_pass_rate == 0.75

    def test_to_dict_serializes(self):
        results = [_r("c1", "m", "g", 1.0, True)]
        rep = ScoreAggregator().aggregate(results)
        d = rep.to_dict()
        assert d["n_cases"] == 1
        assert d["overall_pass_rate"] == 1.0
        assert "case_results" in d
