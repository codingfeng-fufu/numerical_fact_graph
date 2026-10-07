from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class S5PredicateEvent:
    attempt_index: int
    operator: str
    predicate: str
    evaluated: bool
    passed: bool
    unchecked_status: str
    unchecked_answer: float | str | None
    selected_token_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "operator": self.operator,
            "predicate": self.predicate,
            "evaluated": self.evaluated,
            "passed": self.passed,
            "unchecked_status": self.unchecked_status,
            "unchecked_answer": self.unchecked_answer,
            "selected_token_ids": list(self.selected_token_ids),
        }


@dataclass(frozen=True)
class S5AuditTrace:
    events: tuple[S5PredicateEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"events": [event.to_dict() for event in self.events]}
