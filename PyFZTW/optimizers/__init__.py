"""Optimizer interfaces and lazy exports for DEAP and PSO search backends."""

from .base import OptimizationResult, Optimizer
from .parallel import ParallelEvaluatorConfig, ParallelEvaluator, resolve_n_jobs

_LAZY_EXPORTS = {
    "DeapOptimizerConfig": (".deap", "DeapOptimizerConfig"),
    "DeapOptimizer": (".deap", "DeapOptimizer"),
    "PsoOptimizerConfig": (".pso", "PsoOptimizerConfig"),
    "PsoOptimizer": (".pso", "PsoOptimizer"),
}


def __getattr__(name: str):
    """Import heavy optimizer implementations only when needed."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "OptimizationResult",
    "Optimizer",
    "DeapOptimizerConfig",
    "DeapOptimizer",
    "PsoOptimizerConfig",
    "PsoOptimizer",
    "ParallelEvaluatorConfig",
    "ParallelEvaluator",
    "resolve_n_jobs",
]
