"""Inversion helpers for scoring traces and fitting shared model parameters."""

from .objective import WindowedCorrelationFitness, find_best_time_shift
from .resolve import (
    resolve_parameter_space,
    resolve_crust_parameters,
    resolve_earthquake_parameters,
    resolve_model_parameters,
)
from .inversion import (
    MultiTraceInversionProblem,
    make_multi_event_joint_parameter_space,
    split_multi_event_joint_vector,
    build_multi_trace_objective,
    run_multi_trace_inversion,
    find_earthquake_by_id,
    run_inversion_for_seismograms,
)

__all__ = [
    "WindowedCorrelationFitness",
    "find_best_time_shift",
    "resolve_parameter_space",
    "resolve_crust_parameters",
    "resolve_earthquake_parameters",
    "resolve_model_parameters",
    "MultiTraceInversionProblem",
    "make_multi_event_joint_parameter_space",
    "split_multi_event_joint_vector",
    "build_multi_trace_objective",
    "run_multi_trace_inversion",
    "find_earthquake_by_id",
    "run_inversion_for_seismograms",
]
