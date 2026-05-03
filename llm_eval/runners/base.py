"""ModelRunner protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelRunner(Protocol):
    """Wraps any LLM (or stub) into a uniform call.

    Production adapters wrap OpenAI, Anthropic, local llama.cpp, etc.
    """
    name: str

    def run(self, prompt: str) -> str:
        """Run the model on a fully-rendered prompt and return the output."""
        ...
