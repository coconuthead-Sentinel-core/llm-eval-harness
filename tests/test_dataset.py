"""Tests for Dataset and JsonlDataset."""
from llm_eval import Dataset, JsonlDataset, EvalCase


class TestDataset:
    def test_starts_empty(self):
        d = Dataset()
        assert len(d) == 0
        assert list(d) == []

    def test_add_and_iter(self):
        d = Dataset([EvalCase("a"), EvalCase("b")])
        d.add(EvalCase("c"))
        assert len(d) == 3
        ids = [c.case_id for c in d]
        assert ids == ["a", "b", "c"]

    def test_get(self):
        d = Dataset([EvalCase("a", expected="x"),
                     EvalCase("b", expected="y")])
        assert d.get("a").expected == "x"
        assert d.get("zzz") is None

    def test_jsonl_round_trip(self, tmp_path):
        d = Dataset([
            EvalCase("a", prompt_inputs={"x": 1}, expected="alpha"),
            EvalCase("b", prompt_inputs={"x": 2}, expected="bravo",
                     rubric="be brief"),
        ])
        path = tmp_path / "ds.jsonl"
        d.to_jsonl(path)
        loaded = JsonlDataset(path)
        assert len(loaded) == 2
        assert loaded.get("a").expected == "alpha"
        assert loaded.get("b").rubric == "be brief"

    def test_load_missing_file_returns_empty(self, tmp_path):
        d = JsonlDataset(tmp_path / "nope.jsonl")
        assert len(d) == 0
