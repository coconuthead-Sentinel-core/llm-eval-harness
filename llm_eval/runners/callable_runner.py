"""CallableRunner — wraps any ``Callable[[str], str]`` as a ModelRunner.

Lets users plug in a custom Python function as a model with one line:

    runner = CallableRunner(my_fn, name="my_model")

Production adapters subclass or wrap this — e.g. a thin OpenAI wrapper
function gets `CallableRunner(openai_chat, name="gpt-4o")`.
"""
from __future__ import annotations

from typing import Callable


class CallableRunner:
    def __init__(self, fn: Callable[[str], str], *, name: str):
        if not callable(fn):
            raise TypeError("fn must be callable")
        if not name:
            raise ValueError("name required")
        self._fn = fn
        self.name = name

    def run(self, prompt: str) -> str:
        return self._fn(prompt)
