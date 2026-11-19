"""Common optimizer protocol and serializable result object.

Concrete optimizers search a ``ParameterSpace`` using an objective function and
return ``OptimizationResult`` with the best fitness, resolved physical
parameters, optional time shifts, and compact search history for later analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, Any

from ..io.parameters import ParameterSpace
from ..io.project import Project


@dataclass
class OptimizationResult:
    """Best candidate and optional history produced by an optimizer run."""
    best_score: float
    best_resolved: dict[str, float] | None = None
    best_shift_values: list[float] | None = None
    history: list[dict[str, Any]] | None = None
    variable_names: list[str] | None = None
    metadata: dict[str, Any] | None = None
    # Event ID -> synthetic fit-window start in seconds, not a relative lag.
    # All stations in an event use this start for the best_resolved model.
    best_event_window_start_s: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a compact pickle-friendly representation of the result."""
        compact_history: list[dict[str, Any]] = []
        if self.history:
            for gen_data in self.history:
                compact_individuals: list[dict[str, Any]] = []
                for ind in gen_data.get("individuals", []):
                    compact_individuals.append({
                        "score": ind["score"],
                        "resolved": ind["resolved"],
                    })
                compact_history.append({
                    "generation": gen_data["generation"],
                    "individuals": compact_individuals,
                })
        result = {
            "best_score": float(self.best_score),
            "best_resolved": dict(self.best_resolved) if self.best_resolved else {},
            "history": compact_history,
        }
        if self.metadata:
            result.update(self.metadata)
        result["best_event_window_start_s"] = dict(
            getattr(self, "best_event_window_start_s", None) or {}
        )
        return result

class Optimizer(Protocol):
    """Interface implemented by all PyFZTW search algorithms."""

    def optimize(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[list[float]], float],
    ) -> OptimizationResult:
        ...
