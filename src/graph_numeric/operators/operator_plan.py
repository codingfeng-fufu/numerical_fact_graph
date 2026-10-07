from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph_numeric.operators.operator_registry import OPERATOR_SPECS


@dataclass(frozen=True)
class Slot:
    """Unified grounded slot with surface text and confidence."""

    surface: str
    grounded_value: object
    confidence: float


@dataclass(frozen=True)
class Filter:
    type: str  # "time", "entity", "entity_metadata"
    surface: str
    grounded_key: str
    grounded_value: object
    confidence: float


@dataclass(frozen=True)
class OperatorPlan:
    """Unified operator plan for all numerical operators.

    Each operator defines required and optional slots; missing required slots
    will cause Executor to reject the plan.
    """

    operator: str
    query_id: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)
    filters: list[Filter] = field(default_factory=list)
    confidence: float = 1.0
    depends_on: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None

    OPTIONAL_SLOTS = {
        name: set(spec.optional_slots)
        for name, spec in OPERATOR_SPECS.items()
    }

    REQUIRED_SLOTS = {
        name: set(spec.required_slots)
        for name, spec in OPERATOR_SPECS.items()
    }

    def missing_required_slots(self) -> set[str]:
        required = self.REQUIRED_SLOTS.get(self.operator, set())
        grounded = {
            key
            for key, value in self.slots.items()
            if isinstance(value, Slot) and value.grounded_value is not None
        }
        return required - grounded

    def is_valid(self) -> bool:
        return len(self.missing_required_slots()) == 0

    @property
    def target_field(self) -> str | None:
        """Backward compatible: returns grounded field name."""
        slot = (self.slots or {}).get("target_field")
        if isinstance(slot, Slot):
            return str(slot.grounded_value) if slot.grounded_value is not None else None
        return slot

    @property
    def selected_token_ids(self) -> tuple[str, ...]:
        """Backward compatible: selected token IDs."""
        ids = (self.slots or {}).get("selected_token_ids")
        if isinstance(ids, (list, tuple)):
            return tuple(ids)
        return ()

    @property
    def field_scores(self) -> dict[str, float]:
        """Backward compatible: field grounding scores."""
        return self.trace.get("final_scores", {}) if self.trace else {}

    @property
    def scorer_trace(self) -> dict[str, object] | None:
        """Backward compatible alias for trace."""
        return self.trace

    def to_dict(self) -> dict[str, object]:
        return {
            "operator": self.operator,
            "query_id": self.query_id,
            "slots": {
                key: (
                    {
                        "surface": value.surface,
                        "grounded_value": value.grounded_value,
                        "confidence": value.confidence,
                    }
                    if isinstance(value, Slot)
                    else value
                )
                for key, value in (self.slots or {}).items()
            },
            "filters": [
                {
                    "type": f.type,
                    "surface": f.surface,
                    "grounded_key": f.grounded_key,
                    "grounded_value": f.grounded_value,
                    "confidence": f.confidence,
                }
                for f in (self.filters or [])
            ],
            "confidence": self.confidence,
            "depends_on": self.depends_on,
            "trace": self.trace,
        }


@dataclass(frozen=True)
class CompositeOperatorPlan:
    """A minimal multi-step plan representation for composite reasoning."""

    steps: tuple[OperatorPlan, ...]
    query_id: str | None = None
    confidence: float = 1.0
    trace: dict[str, Any] | None = None

    def is_valid(self) -> bool:
        seen_step_ids: set[str] = set()
        for index, step in enumerate(self.steps):
            missing = step.missing_required_slots() - set((step.depends_on or {}).keys())
            if missing:
                return False
            if not step.depends_on:
                seen_step_ids.add(_step_id(step, index))
                continue
            refs = [
                str(value).split(".", 1)[0].lstrip("$")
                for value in step.depends_on.values()
                if isinstance(value, str) and value.startswith("$")
            ]
            if any(ref not in seen_step_ids for ref in refs):
                return False
            seen_step_ids.add(_step_id(step, index))
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "steps": [step.to_dict() for step in self.steps],
            "confidence": self.confidence,
            "trace": self.trace,
        }


# Backward compatibility alias
SumOperatorPlan = OperatorPlan


def _step_id(step: OperatorPlan, index: int) -> str:
    value = (step.trace or {}).get("step_id") if step.trace else None
    return str(value) if value else f"s{index + 1}"
