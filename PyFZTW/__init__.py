"""Public API for PyFZTW inversion workflows.

This package re-exports the parsers, forward models, inversion helpers, and
optimizers used to load FZTW input files, generate synthetic traces, score them
against observed seismograms, and return optimization results.
"""

from .io import (
    ParameterSpec,
    ParameterSpace,
    parse_parameter_line,
    CrustConfig,
    clean_lines,
    parse_crust_file,
    Earthquake,
    parse_earthquake_file,
    Station,
    parse_station_file,
    SeismogramHeader,
    Seismogram,
    parse_seismogram_file,
    parse_seismogram_files,
    seismogram_key,
    Project,
    load_project,
    validate_project,
    check_all_earthquakes_used,
)
from .forward import (
    ForwardModel,
    SyntheticTrace,
    HS4PDISForwardConfig,
    HS4PDISForwardModel,
)
from .inverse import (
    WindowedCorrelationFitness,
    find_best_time_shift,
    resolve_parameter_space,
    resolve_crust_parameters,
    resolve_earthquake_parameters,
    resolve_model_parameters,
    MultiTraceInversionProblem,
    make_multi_event_joint_parameter_space,
    split_multi_event_joint_vector,
    build_multi_trace_objective,
    run_multi_trace_inversion,
    find_earthquake_by_id,
    run_inversion_for_seismograms,
)
from .optimizers import (
    OptimizationResult,
    Optimizer,
    ParallelEvaluatorConfig,
    ParallelEvaluator,
    resolve_n_jobs,
)
from .cli import (
    BatchInversionConfig,
    build_deap_optimizer,
    run_single_inversion,
    run_repeated_inversion,
)

_LAZY_EXPORTS = {
    "DeapOptimizerConfig": (".optimizers", "DeapOptimizerConfig"),
    "DeapOptimizer": (".optimizers", "DeapOptimizer"),
    "PsoOptimizerConfig": (".optimizers", "PsoOptimizerConfig"),
    "PsoOptimizer": (".optimizers", "PsoOptimizer"),
}


def __getattr__(name: str):
    """Load optional optimizer classes only when they are requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "ParameterSpec",
    "ParameterSpace",
    "parse_parameter_line",
    "CrustConfig",
    "clean_lines",
    "parse_crust_file",
    "Earthquake",
    "parse_earthquake_file",
    "Station",
    "parse_station_file",
    "SeismogramHeader",
    "Seismogram",
    "parse_seismogram_file",
    "parse_seismogram_files",
    "seismogram_key",
    "Project",
    "load_project",
    "validate_project",
    "check_all_earthquakes_used",
    "ForwardModel",
    "SyntheticTrace",
    "HS4PDISForwardConfig",
    "HS4PDISForwardModel",
    "WindowedCorrelationFitness",
    "find_best_time_shift",
    "resolve_parameter_space",
    "resolve_crust_parameters",
    "resolve_earthquake_parameters",
    "resolve_model_parameters",
    "OptimizationResult",
    "Optimizer",
    "DeapOptimizerConfig",
    "DeapOptimizer",
    "PsoOptimizerConfig",
    "PsoOptimizer",
    "ParallelEvaluatorConfig",
    "ParallelEvaluator",
    "resolve_n_jobs",
    "MultiTraceInversionProblem",
    "make_multi_event_joint_parameter_space",
    "split_multi_event_joint_vector",
    "build_multi_trace_objective",
    "run_multi_trace_inversion",
    "find_earthquake_by_id",
    "run_inversion_for_seismograms",
    "BatchInversionConfig",
    "build_deap_optimizer",
    "run_single_inversion",
    "run_repeated_inversion",
]
