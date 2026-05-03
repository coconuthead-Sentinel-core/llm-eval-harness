"""EchoRunner — deterministic stub runner for tests + smoke flows.

Returns the prompt verbatim, optionally with a prefix/suffix. Useful
to validate that the harness wiring is correct independent of any
real model.
"""
from __future__ import annotations


class EchoRunner:
    """Returns the prompt back, optionally wrapped."""
    def __init__(self, *, name: str = "echo",
                 prefix: str = "", suffix: str = ""):
        self.name = name
        self.prefix = prefix
        self.suffix = suffix

    def run(self, prompt: str) -> str:
        return f"{self.prefix}{prompt}{self.suffix}"
