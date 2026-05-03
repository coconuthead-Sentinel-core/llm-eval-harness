"""PromptTemplate — safe-format string templates with named placeholders."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class PromptTemplate:
    """Named-placeholder string template.

    Renders ``"Hello {name}, you have {n} new messages"`` against
    ``{"name": "Shannon", "n": 3}`` into the final prompt string.
    Missing keys raise KeyError; extra keys are ignored.

    Use ``required_keys`` / ``placeholders`` to introspect.
    """
    template: str

    @property
    def placeholders(self) -> list[str]:
        seen: list[str] = []
        for m in _PLACEHOLDER_RE.finditer(self.template):
            k = m.group(1)
            if k not in seen:
                seen.append(k)
        return seen

    @property
    def required_keys(self) -> set[str]:
        return set(self.placeholders)

    def render(self, inputs: dict[str, Any]) -> str:
        missing = self.required_keys - set(inputs.keys())
        if missing:
            raise KeyError(f"missing prompt inputs: {sorted(missing)}")
        # Use string.Formatter so format-spec like {var:>10} could work,
        # but we keep it simple — just substitute by name.
        return _PLACEHOLDER_RE.sub(
            lambda m: str(inputs[m.group(1)]),
            self.template,
        )

    def render_safe(self, inputs: dict[str, Any], *,
                    default: str = "") -> str:
        """Like render() but substitutes ``default`` for missing keys."""
        return _PLACEHOLDER_RE.sub(
            lambda m: str(inputs.get(m.group(1), default)),
            self.template,
        )
