"""Tests for PromptTemplate."""
import pytest

from llm_eval import PromptTemplate


class TestPromptTemplate:
    def test_no_placeholders(self):
        p = PromptTemplate("hello")
        assert p.placeholders == []
        assert p.render({}) == "hello"

    def test_render_with_placeholders(self):
        p = PromptTemplate("hello {name}, you have {n} messages")
        out = p.render({"name": "Shannon", "n": 3})
        assert out == "hello Shannon, you have 3 messages"

    def test_required_keys(self):
        p = PromptTemplate("{a}+{b}={a_plus_b}")
        assert p.required_keys == {"a", "b", "a_plus_b"}

    def test_missing_key_raises(self):
        p = PromptTemplate("hello {name}")
        with pytest.raises(KeyError):
            p.render({})

    def test_extra_keys_ignored(self):
        p = PromptTemplate("hello {name}")
        out = p.render({"name": "x", "unused": "y"})
        assert out == "hello x"

    def test_render_safe_substitutes_default(self):
        p = PromptTemplate("hello {name} from {city}")
        out = p.render_safe({"name": "Shannon"}, default="<unknown>")
        assert out == "hello Shannon from <unknown>"

    def test_dedupe_placeholders(self):
        p = PromptTemplate("{a} and {a} again")
        assert p.placeholders == ["a"]
