"""Parallel objective evaluation for optimizer populations.

The evaluator batches candidate vectors and returns each score together with
optional metadata collected by the objective: resolved model parameters and
best time shifts. It can use threads or processes, falling back safely when a
process backend cannot pickle the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import multiprocessing as mp
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import ThreadPool


_PROCESS_EVAL_CALLABLE = None
_PROCESS_USE_META = False


def _init_process_worker(eval_payload: bytes, use_meta: bool) -> None:
    global _PROCESS_EVAL_CALLABLE, _PROCESS_USE_META
    _PROCESS_EVAL_CALLABLE = pickle.loads(eval_payload)
    _PROCESS_USE_META = use_meta


def _process_eval_vector(vector: list[float]) -> tuple[float, dict[str, float], list[float]]:
    if _PROCESS_EVAL_CALLABLE is None:
        raise RuntimeError("Process worker evaluator is not initialized")
    if _PROCESS_USE_META:
        score, resolved, shift_values = _PROCESS_EVAL_CALLABLE(vector)
        return float(score), dict(resolved), [float(v) for v in shift_values]
    score = float(_PROCESS_EVAL_CALLABLE(vector))
    return score, {}, []


@dataclass
class ParallelEvaluatorConfig:
    """Runtime settings for batched objective evaluation."""
    n_jobs: int = 1
    backend: str = "thread"
    process_start_method: str = "spawn"


def resolve_n_jobs(config: ParallelEvaluatorConfig) -> int:
    """Validate parallel settings and convert ``-1`` to the CPU count."""
    if config.n_jobs == 0 or config.n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    if config.backend not in {"thread", "process"}:
        raise ValueError("backend must be 'thread' or 'process'")
    if config.process_start_method not in {"spawn", "fork", "forkserver"}:
        raise ValueError("process_start_method must be 'spawn', 'fork', or 'forkserver'")
    if os.name == "nt" and config.process_start_method != "spawn":
        raise ValueError("Windows only supports process_start_method='spawn'")

    if config.n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return config.n_jobs


class ParallelEvaluator:
    """Context-managed evaluator used by optimizers to score populations."""

    def __init__(
        self,
        objective_fn: Callable[[list[float]], float],
        config: ParallelEvaluatorConfig | None = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.config = config or ParallelEvaluatorConfig()
        self.n_jobs = resolve_n_jobs(self.config)

        eval_with_meta = getattr(objective_fn, "evaluate_with_meta", None)
        self.use_meta = callable(eval_with_meta)
        self._eval_with_meta = eval_with_meta
        self._process_executor: ProcessPoolExecutor | None = None
        self._thread_pool: ThreadPool | None = None

        if self.n_jobs > 1 and self.config.backend == "process":
            eval_target = self._eval_with_meta if self.use_meta else objective_fn
            try:
                eval_payload = pickle.dumps(eval_target)
                mp_context = mp.get_context(self.config.process_start_method)
                self._process_executor = ProcessPoolExecutor(
                    max_workers=self.n_jobs,
                    initializer=_init_process_worker,
                    initargs=(eval_payload, self.use_meta),
                    mp_context=mp_context,
                )
            except Exception as exc:
                warnings.warn(
                    (
                        "Process backend is unavailable for current objective_fn "
                        f"(falling back to thread backend): {exc}"
                    ),
                    stacklevel=2,
                )

        if self.n_jobs > 1 and self._process_executor is None:
            self._thread_pool = ThreadPool(processes=self.n_jobs)

    def _evaluate_vector_with_meta(
        self,
        vector: list[float],
    ) -> tuple[float, dict[str, float], list[float]]:
        if self.use_meta:
            score, resolved, shift_values = self._eval_with_meta(vector)
            return float(score), dict(resolved), [float(v) for v in shift_values]
        score = float(self.objective_fn(vector))
        shift_values = getattr(self.objective_fn, "last_best_shift_values", [])
        resolved = getattr(self.objective_fn, "last_resolved", {})
        return score, dict(resolved), [float(v) for v in shift_values]

    def evaluate(
        self,
        vectors: list[list[float]],
    ) -> list[tuple[float, dict[str, float], list[float]]]:
        """Evaluate candidate vectors and return score/resolved/shift tuples."""
        if not vectors:
            return []
        if self._process_executor is not None:
            return list(self._process_executor.map(_process_eval_vector, vectors))
        if self._thread_pool is not None:
            return list(self._thread_pool.map(self._evaluate_vector_with_meta, vectors))
        return [self._evaluate_vector_with_meta(vec) for vec in vectors]

    def close(self) -> None:
        if self._thread_pool is not None:
            self._thread_pool.close()
            self._thread_pool.join()
            self._thread_pool = None
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=True)
            self._process_executor = None

    def __enter__(self) -> ParallelEvaluator:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
