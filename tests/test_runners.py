"""Tests for ModelRunner implementations."""
import pytest

from llm_eval import ModelRunner, EchoRunner, CallableRunner


class TestEchoRunner:
    def test_implements_protocol(self):
        r = EchoRunner()
        assert isinstance(r, ModelRunner)

    def test_default_echoes_verbatim(self):
        r = EchoRunner()
        assert r.run("hello") == "hello"

    def test_with_prefix_suffix(self):
        r = EchoRunner(prefix=">> ", suffix=" <<")
        assert r.run("x") == ">> x <<"

    def test_name_default(self):
        assert EchoRunner().name == "echo"


class TestCallableRunner:
    def test_implements_protocol(self):
        r = CallableRunner(lambda s: s, name="x")
        assert isinstance(r, ModelRunner)

    def test_runs_callable(self):
        r = CallableRunner(lambda s: s.upper(), name="upper")
        assert r.run("hello") == "HELLO"

    def test_non_callable_rejected(self):
        with pytest.raises(TypeError):
            CallableRunner("not-a-function", name="x")  # type: ignore[arg-type]

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            CallableRunner(lambda s: s, name="")
