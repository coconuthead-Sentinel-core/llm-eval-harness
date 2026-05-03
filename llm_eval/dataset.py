"""Dataset abstractions: in-memory + JSONL on disk."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from .case import EvalCase


class Dataset:
    """In-memory dataset of EvalCases."""

    def __init__(self, cases: Iterable[EvalCase] | None = None):
        self._cases: list[EvalCase] = list(cases or [])

    def add(self, case: EvalCase) -> None:
        self._cases.append(case)

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self._cases)

    def get(self, case_id: str) -> EvalCase | None:
        for c in self._cases:
            if c.case_id == case_id:
                return c
        return None

    def to_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            for c in self._cases:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        return p


class JsonlDataset(Dataset):
    """Dataset loaded from a JSONL file. Each line is one EvalCase."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        super().__init__()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    d = json.loads(raw)
                    self._cases.append(EvalCase(
                        case_id=d["case_id"],
                        prompt_inputs=d.get("prompt_inputs", {}),
                        expected=d.get("expected", ""),
                        metadata=d.get("metadata", {}),
                        rubric=d.get("rubric", ""),
                    ))
