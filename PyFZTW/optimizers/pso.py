"""Particle swarm optimizer for bounded PyFZTW parameter searches.

The optimizer moves a population of particles through the parameter bounds,
tracking personal and global best candidates. It returns the best score,
resolved model parameters, optional best shifts, and iteration history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import random

import numpy as np

from ..io.parameters import ParameterSpace
from .base import Optimizer, OptimizationResult
from .parallel import ParallelEvaluator, ParallelEvaluatorConfig


@dataclass
class PsoOptimizerConfig:
    """Particle swarm settings used by ``PsoOptimizer``."""
    population_size: int = 200
    n_iterations: int = 50
    inertia_weight: float = 0.7
    inertia_weight_end: float | None = None
    cognitive_coefficient: float = 1.5
    social_coefficient: float = 1.5
    max_velocity_ratio: float = 0.2
    random_seed: int | None = None
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 1e-6
    maximize: bool = True


class PsoOptimizer(Optimizer):
    """Maximize or minimize the objective with bounded particle swarm search."""

    def __init__(
        self,
        config: PsoOptimizerConfig | None = None,
        evaluator_config: ParallelEvaluatorConfig | None = None,
    ) -> None:
        self.config = config or PsoOptimizerConfig()
        self.evaluator_config = evaluator_config or ParallelEvaluatorConfig()

    def optimize(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[list[float]], float],
    ) -> OptimizationResult:
        """Run particle swarm search and return the best candidate and history."""
        cfg = self.config

        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)
        rng = np.random.default_rng(cfg.random_seed)

        bounds = parameter_space.bounds()
        n_dim = len(bounds)

        if n_dim == 0:
            raise ValueError("Parameter space has no free parameters")
        if cfg.population_size <= 0:
            raise ValueError("population_size must be positive")
        if cfg.n_iterations < 0:
            raise ValueError("n_iterations must be non-negative")
        if cfg.early_stop_patience is not None and cfg.early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive")
        if cfg.max_velocity_ratio < 0:
            raise ValueError("max_velocity_ratio must be non-negative")

        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        span = high - low
        if np.any(span <= 0):
            raise ValueError("Each parameter upper bound must be greater than lower bound")

        max_velocity = cfg.max_velocity_ratio * span

        def is_better(candidate: float, incumbent: float, min_delta: float = 0.0) -> bool:
            if cfg.maximize:
                return candidate > incumbent + min_delta
            return candidate < incumbent - min_delta

        def best_index(scores: np.ndarray) -> int:
            return int(np.argmax(scores) if cfg.maximize else np.argmin(scores))

        def current_inertia(iteration: int) -> float:
            if cfg.inertia_weight_end is None or cfg.n_iterations <= 1:
                return cfg.inertia_weight
            ratio = (iteration - 1) / max(cfg.n_iterations - 1, 1)
            return cfg.inertia_weight + ratio * (cfg.inertia_weight_end - cfg.inertia_weight)

        def clone_resolved(value: object) -> dict[str, float]:
            return dict(value) if isinstance(value, dict) else {}

        def clone_shift_values(value: object) -> list[float]:
            return list(value) if value is not None else []

        def make_generation_record(
            generation: int,
            positions: np.ndarray,
            velocities: np.ndarray,
            scores: np.ndarray,
            resolved_values: list[object],
            shift_values_list: list[object],
            pbest_scores: np.ndarray | None = None,
        ) -> dict:
            individuals = []
            for i in range(len(positions)):
                item = {
                    "vector": positions[i].tolist(),
                    "velocity": velocities[i].tolist(),
                    "score": float(scores[i]),
                    "resolved": clone_resolved(resolved_values[i]),
                    "shift_values": clone_shift_values(shift_values_list[i]),
                }
                if pbest_scores is not None:
                    item["pbest_score"] = float(pbest_scores[i])
                individuals.append(item)
            return {"generation": generation, "individuals": individuals}

        with ParallelEvaluator(objective_fn, self.evaluator_config) as evaluator:
            positions = rng.uniform(low, high, size=(cfg.population_size, n_dim))
            velocities = rng.uniform(-max_velocity, max_velocity, size=(cfg.population_size, n_dim))

            def evaluate_positions(pos: np.ndarray) -> tuple[np.ndarray, list[object], list[object]]:
                vectors = [row.tolist() for row in pos]
                results = evaluator.evaluate(vectors)
                scores = np.array([float(score) for score, _, _ in results], dtype=float)
                resolved_values = [resolved for _, resolved, _ in results]
                shift_values_list = [shift_values for _, _, shift_values in results]
                return scores, resolved_values, shift_values_list

            scores, resolved_values, shift_values_list = evaluate_positions(positions)

            personal_best_positions = positions.copy()
            personal_best_scores = scores.copy()
            personal_best_resolved = [clone_resolved(v) for v in resolved_values]
            personal_best_shift_values = [clone_shift_values(v) for v in shift_values_list]

            global_best_idx = best_index(personal_best_scores)
            global_best_position = personal_best_positions[global_best_idx].copy()
            global_best_score = float(personal_best_scores[global_best_idx])
            global_best_resolved = personal_best_resolved[global_best_idx]
            global_best_shift_values = personal_best_shift_values[global_best_idx]
            gbest_history = [
                {
                    "generation": 0,
                    "gbest_score": global_best_score,
                    "gbest_resolved": clone_resolved(global_best_resolved),
                }
            ]

            history = [
                make_generation_record(
                    0,
                    positions,
                    velocities,
                    scores,
                    resolved_values,
                    shift_values_list,
                    personal_best_scores,
                )
            ]

            best_score_so_far = global_best_score
            no_improve_generations = 0

            for generation in range(1, cfg.n_iterations + 1):
                w = current_inertia(generation)
                r1 = rng.random(size=(cfg.population_size, n_dim))
                r2 = rng.random(size=(cfg.population_size, n_dim))

                velocities = (
                    w * velocities
                    + cfg.cognitive_coefficient * r1 * (personal_best_positions - positions)
                    + cfg.social_coefficient * r2 * (global_best_position - positions)
                )

                if cfg.max_velocity_ratio > 0:
                    velocities = np.clip(velocities, -max_velocity, max_velocity)

                positions = positions + velocities

                below = positions < low
                above = positions > high
                hit_boundary = below | above
                positions = np.clip(positions, low, high)
                velocities[hit_boundary] = 0.0

                scores, resolved_values, shift_values_list = evaluate_positions(positions)

                if cfg.maximize:
                    improved = scores > personal_best_scores
                else:
                    improved = scores < personal_best_scores

                personal_best_positions[improved] = positions[improved]
                personal_best_scores[improved] = scores[improved]
                for i in np.where(improved)[0]:
                    personal_best_resolved[int(i)] = clone_resolved(resolved_values[int(i)])
                    personal_best_shift_values[int(i)] = clone_shift_values(shift_values_list[int(i)])

                current_best_idx = best_index(personal_best_scores)
                current_best_score = float(personal_best_scores[current_best_idx])
                if is_better(current_best_score, global_best_score):
                    global_best_score = current_best_score
                    global_best_position = personal_best_positions[current_best_idx].copy()
                    global_best_resolved = personal_best_resolved[current_best_idx]
                    global_best_shift_values = personal_best_shift_values[current_best_idx]
                gbest_history.append(
                    {
                        "generation": generation,
                        "gbest_score": float(global_best_score),
                        "gbest_resolved": clone_resolved(global_best_resolved),
                    }
                )

                history.append(
                    make_generation_record(
                        generation,
                        positions,
                        velocities,
                        scores,
                        resolved_values,
                        shift_values_list,
                        personal_best_scores,
                    )
                )

                if is_better(global_best_score, best_score_so_far, cfg.early_stop_min_delta):
                    best_score_so_far = global_best_score
                    no_improve_generations = 0
                else:
                    no_improve_generations += 1

                if (
                    cfg.early_stop_patience is not None
                    and no_improve_generations >= cfg.early_stop_patience
                ):
                    break

        variable_names = parameter_space.variable_names()
        pbest_population = []
        for i in range(cfg.population_size):
            pbest_population.append(
                {
                    "pbest_score": float(personal_best_scores[i]),
                    "pbest_resolved": clone_resolved(personal_best_resolved[i]),
                }
            )

        return OptimizationResult(
            best_score=global_best_score,
            best_resolved=global_best_resolved,
            best_shift_values=global_best_shift_values,
            history=history,
            variable_names=variable_names,
            metadata={
                "optimizer": "pso",
                "pso": {
                    "pbest_population": pbest_population,
                    "gbest_history": gbest_history,
                },
            },
        )
