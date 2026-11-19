"""High-level multi-trace inversion orchestration.

The module joins shared crust parameters with event-specific source parameters,
evaluates each candidate against all requested seismograms, and returns an
``OptimizationResult`` containing the best score, resolved model values, best
time shifts, and per-generation search history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..forward.base import ForwardModel, SyntheticTrace
from ..io.earthquake import Earthquake
from ..io.parameters import ParameterSpace
from ..io.project import Project
from ..io.seismogram import Seismogram
from ..optimizers.base import Optimizer, OptimizationResult
from .objective import WindowedCorrelationFitness
from .resolve import resolve_model_parameters

@dataclass
class MultiTraceInversionProblem:
    """A validated group of seismograms to fit against one project."""
    project: Project
    seismograms: list[Seismogram]

    def __post_init__(self) -> None:
        if not self.seismograms:
            raise ValueError("seismograms cannot be empty")

    def crust_parameter_space(self) -> ParameterSpace:
        return self.project.crust.parameter_space

    def event_ids(self) -> list[str]:
        return sorted({gram.header.earthquake_id for gram in self.seismograms})

    def earthquakes(self) -> list[Earthquake]:
        return [find_earthquake_by_id(self.project, event_id) for event_id in self.event_ids()]

def _shift_label(event_id: str) -> str:
    return f"eq{event_id}_best_shift"

def make_multi_event_joint_parameter_space(
    crust_space: ParameterSpace,
    earthquake_spaces: list[ParameterSpace],
) -> ParameterSpace:
    """Build one optimizer search space from crust and event parameter spaces."""
    joint_parameters = [*crust_space.free_parameters()]
    for space in earthquake_spaces:
        joint_parameters.extend(space.free_parameters())
    return ParameterSpace(joint_parameters)


def split_multi_event_joint_vector(
    joint_vector: list[float],
    crust_space: ParameterSpace,
    earthquake_spaces: list[ParameterSpace],
) -> tuple[list[float], list[list[float]]]:
    """Split one optimizer vector into crust and per-event parameter vectors."""
    n_crust = len(crust_space.free_parameters())
    n_total_eq = sum(len(space.free_parameters()) for space in earthquake_spaces)
    expected = n_crust + n_total_eq

    if len(joint_vector) != expected:
        raise ValueError(
            f"Joint vector length mismatch: got {len(joint_vector)}, expected {expected}"
        )

    crust_vector = joint_vector[:n_crust]
    offset = n_crust
    earthquake_vectors: list[list[float]] = []
    for space in earthquake_spaces:
        n_eq = len(space.free_parameters())
        earthquake_vectors.append(joint_vector[offset:offset + n_eq])
        offset += n_eq

    return crust_vector, earthquake_vectors

@dataclass
class _MultiTraceObjectiveCallable:
    """Callable objective that scores one joint vector across all seismograms."""
    problem: MultiTraceInversionProblem
    forward_model: ForwardModel
    objective: WindowedCorrelationFitness

    def __post_init__(self) -> None:
        self.crust_space = self.problem.crust_parameter_space()
        self.event_ids = self.problem.event_ids()
        earthquakes = [
            find_earthquake_by_id(self.problem.project, event_id)
            for event_id in self.event_ids
        ]
        self.earthquake_spaces = [eq.parameter_space for eq in earthquakes]
        self.event_to_earthquake = {eq.event_id: eq for eq in earthquakes}
        self.event_to_space = {eq.event_id: eq.parameter_space for eq in earthquakes}
        self.shift_labels = [
            _shift_label(event_id)
            for event_id in self.event_ids
        ]
        self.penalty_score = 0.0
        self.last_resolved: dict[str, float] = {}
        self.last_best_shift_values = [float("nan")] * len(self.shift_labels)

    def evaluate_with_meta(
        self,
        joint_vector: list[float],
    ) -> tuple[float, dict[str, float], list[float]]:
        """Return fitness plus resolved parameters and best shifts for reporting."""
        try:
            crust_vector, earthquake_vectors = split_multi_event_joint_vector(
                joint_vector=joint_vector,
                crust_space=self.crust_space,
                earthquake_spaces=self.earthquake_spaces,
            )
            crust_values = self.crust_space.vector_to_parameters(crust_vector)
            event_to_earthquake_values = {
                event_id: self.event_to_space[event_id].vector_to_parameters(eq_vector)
                for event_id, eq_vector in zip(self.event_ids, earthquake_vectors)
            }

            weighted_score_sum = 0.0
            weight_sum = 0.0
            shift_values: list[float] = []
            all_resolved: dict[str, float] = {}
            event_observed: dict[str, list[Seismogram]] = {
                event_id: [] for event_id in self.event_ids
            }
            event_synthetic: dict[str, list[SyntheticTrace]] = {
                event_id: [] for event_id in self.event_ids
            }

            # Generate all traces first; align them with one shared shift per event.
            for gram in self.problem.seismograms:
                event_id = gram.header.earthquake_id
                earthquake = self.event_to_earthquake[event_id]
                earthquake_values = event_to_earthquake_values[event_id]
                resolved = resolve_model_parameters(
                    project=self.problem.project,
                    earthquake=earthquake,
                    crust_variable_values=crust_values,
                    earthquake_variable_values=earthquake_values,
                )
                for key, val in resolved.items():
                    if key in ("x", "y", "z"):
                        all_resolved[f"{event_id}_{key}"] = val
                    else:
                        all_resolved[key] = val
                synthetic = self.forward_model.run(
                    project=self.problem.project,
                    seismogram=gram,
                    parameters=resolved,
                )
                event_observed[event_id].append(gram)
                event_synthetic[event_id].append(synthetic)

            for event_id in self.event_ids:
                if not event_observed[event_id]:
                    continue
                event_score, best_shift, aligned_obs, _ = (
                    self.objective.evaluate_event_with_aligned_data(
                        observations=event_observed[event_id],
                        synthetics=event_synthetic[event_id],
                    )
                )
                sample_count = sum(len(data) for data in aligned_obs)
                if sample_count <= 0:
                    raise ValueError("windowed sample count must be positive")
                weight = float(sample_count)
                weighted_score_sum += event_score * weight
                weight_sum += weight
                shift_values.append(float(best_shift))

            if weight_sum <= 0:
                raise ValueError("Total weight is non-positive")

            return weighted_score_sum / weight_sum, all_resolved, shift_values
        except (ValueError, KeyError):
            return self.penalty_score, {}, [float("nan")] * len(self.shift_labels)

    def __call__(self, joint_vector: list[float]) -> float:
        score, resolved, shift_values = self.evaluate_with_meta(joint_vector)
        # Preserve optimizer-facing metadata attributes for serial evaluation.
        self.last_resolved = resolved
        self.last_best_shift_values = shift_values
        return score

def build_multi_trace_objective(
    problem: MultiTraceInversionProblem,
    forward_model: ForwardModel,
    objective: WindowedCorrelationFitness,
) -> Callable[[list[float]], float]:
    """Create the optimizer-facing objective callable for a multi-trace problem."""
    return _MultiTraceObjectiveCallable(
        problem=problem,
        forward_model=forward_model,
        objective=objective,
    )

def run_multi_trace_inversion(
    problem: MultiTraceInversionProblem,
    forward_model: ForwardModel,
    objective: WindowedCorrelationFitness,
    optimizer: Optimizer,
) -> OptimizationResult:
    """Run the optimizer and return the best fitted model with search history."""
    crust_space = problem.crust_parameter_space()
    earthquake_spaces = [eq.parameter_space for eq in problem.earthquakes()]
    joint_space = make_multi_event_joint_parameter_space(
        crust_space=crust_space,
        earthquake_spaces=earthquake_spaces,
    )
    objective_fn = build_multi_trace_objective(
        problem=problem,
        forward_model=forward_model,
        objective=objective,
    )
    result = optimizer.optimize(
        parameter_space=joint_space,
        objective_fn=objective_fn,
    )

    event_ids = problem.event_ids()
    # Attach IDs to the best candidate's shifts, not the last evaluation's.
    # This ordering is also used by the event-level objective.
    shifts = result.best_shift_values
    if shifts is not None and len(shifts) > 0:
        if len(shifts) != len(event_ids):
            raise ValueError("Best time shifts must contain one value per event")
        result.best_event_window_start_s = {
            event_id: float(start) for event_id, start in zip(event_ids, shifts)
        }
    if result.variable_names and event_ids and len(earthquake_spaces) > 0:
        n_crust = len(crust_space.free_parameters())
        renamed = list(result.variable_names[:n_crust])
        offset = n_crust
        for space, event_id in zip(earthquake_spaces, event_ids):
            n_eq = len(space.free_parameters())
            for j in range(n_eq):
                name = result.variable_names[offset + j]
                if name in ("x", "y", "z"):
                    renamed.append(f"{event_id}_{name}")
                else:
                    renamed.append(name)
            offset += n_eq
        result.variable_names = renamed

    return result


def find_earthquake_by_id(project: Project, event_id: str) -> Earthquake:
    """Return the earthquake definition referenced by a seismogram."""
    for eq in project.earthquakes:
        if eq.event_id == event_id:
            return eq
    raise KeyError(f"Earthquake id not found: {event_id}")

def run_inversion_for_seismograms(
    project: Project,
    seismograms: list[Seismogram],
    forward_model: ForwardModel,
    objective: WindowedCorrelationFitness,
    optimizer: Optimizer,
) -> OptimizationResult:
    """Convenience wrapper that builds a problem and runs multi-trace inversion."""
    problem = MultiTraceInversionProblem(
        project=project,
        seismograms=seismograms,
    )
    return run_multi_trace_inversion(
        problem=problem,
        forward_model=forward_model,
        objective=objective,
        optimizer=optimizer,
    )
