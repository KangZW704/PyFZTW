"""DEAP-based genetic algorithm optimizer.

The optimizer samples bounded parameter vectors, evolves them with crossover,
mutation, selection, and elitism, then returns the best model plus population
history. Each history entry includes candidate vectors, scores, resolved model
values, and best time shifts when provided by the objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable
import random

from deap import base, creator, tools

from ..io.parameters import ParameterSpace
from .base import Optimizer, OptimizationResult
from .parallel import ParallelEvaluator, ParallelEvaluatorConfig


@dataclass
class DeapOptimizerConfig:
    """Genetic algorithm settings used by ``DeapOptimizer``."""
    population_size: int = 200
    n_generations: int = 50
    crossover_probability: float = 0.7
    mutation_probability: float = 0.2
    selection_method: str = "roulette"
    tournament_size: int = 3
    eta_crossover: float | list[float] | dict[str, float] = 20.0
    eta_mutation: float | list[float] | dict[str, float] = 20.0
    random_seed: int | None = None
    hall_of_fame_size: int = 1
    elite_size: int = 1
    mutation_indpb: float | None = None
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 1e-6


def _ensure_deap_creators() -> None:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _per_dimension_values(
    value: float | Sequence[float] | Mapping[str, float],
    n_dim: int,
    name: str,
    variable_names: Sequence[str],
    scalar_default: float,
) -> list[float]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, list, or parameter-name mapping")

    if isinstance(value, int | float):
        values = [float(value)] * n_dim
    elif isinstance(value, Mapping):
        normalized = {str(key).lower(): item for key, item in value.items()}
        default = normalized.get("default", normalized.get("*", scalar_default))
        if isinstance(default, bool):
            raise TypeError(f"{name} default must be a number")
        default_value = float(default)

        values = []
        for variable_name in variable_names:
            item = normalized.get(variable_name.lower(), default_value)
            if isinstance(item, bool):
                raise TypeError(f"{name}.{variable_name} must be a number")
            values.append(float(item))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) != n_dim:
            raise ValueError(
                f"{name} must be a scalar or have one value per parameter "
                f"({n_dim}); got {len(value)}"
            )
        values = []
        for item in value:
            if isinstance(item, bool):
                raise TypeError(f"{name} values must be numbers")
            values.append(float(item))
    else:
        raise TypeError(f"{name} must be a number, list, or parameter-name mapping")

    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} values must be non-negative")
    return values


class DeapOptimizer(Optimizer):
    """Maximize the PyFZTW objective with a bounded genetic algorithm."""

    def __init__(
        self,
        config: DeapOptimizerConfig | None = None,
        evaluator_config: ParallelEvaluatorConfig | None = None,
    ) -> None:
        self.config = config or DeapOptimizerConfig()
        self.evaluator_config = evaluator_config or ParallelEvaluatorConfig()
        _ensure_deap_creators()

    def optimize(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[list[float]], float],
    ) -> OptimizationResult:
        """Run the genetic search and return the best candidate and history."""
        cfg = self.config

        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)

        bounds = parameter_space.bounds()
        variable_names = parameter_space.variable_names()
        n_dim = len(bounds)

        if n_dim == 0:
            raise ValueError("Parameter space has no free parameters")
        if cfg.population_size <= 0:
            raise ValueError("population_size must be positive")
        if cfg.n_generations < 0:
            raise ValueError("n_generations must be non-negative")
        if cfg.selection_method not in {"roulette", "tournament"}:
            raise ValueError("selection_method must be 'roulette' or 'tournament'")
        if cfg.hall_of_fame_size <= 0:
            raise ValueError("hall_of_fame_size must be positive")
        if cfg.elite_size < 0:
            raise ValueError("elite_size must be non-negative")
        if cfg.elite_size >= cfg.population_size:
            raise ValueError("elite_size must be smaller than population_size")
        if cfg.early_stop_patience is not None and cfg.early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive")
        eta_crossover_values = _per_dimension_values(
            cfg.eta_crossover,
            n_dim,
            "eta_crossover",
            variable_names,
            DeapOptimizerConfig.eta_crossover,
        )
        eta_mutation_values = _per_dimension_values(
            cfg.eta_mutation,
            n_dim,
            "eta_mutation",
            variable_names,
            DeapOptimizerConfig.eta_mutation,
        )
        toolbox = base.Toolbox()

        def sample_dimension(low: float, high: float) -> float:
            return random.uniform(low, high)

        def create_individual() -> creator.Individual:
            values = [sample_dimension(low, high) for (low, high) in bounds]
            return creator.Individual(values)

        def bounded_simulated_binary_crossover(
            ind1: list[float],
            ind2: list[float],
        ) -> tuple[list[float], list[float]]:
            for i, ((low, up), eta) in enumerate(zip(bounds, eta_crossover_values)):
                gene1 = [ind1[i]]
                gene2 = [ind2[i]]
                tools.cxSimulatedBinaryBounded(
                    gene1,
                    gene2,
                    eta=eta,
                    low=[low],
                    up=[up],
                )
                ind1[i] = gene1[0]
                ind2[i] = gene2[0]
            return ind1, ind2

        def bounded_polynomial_mutation(
            individual: list[float],
        ) -> tuple[list[float]]:
            indpb = cfg.mutation_indpb if cfg.mutation_indpb is not None else (1.0 / n_dim)
            for i, ((low, up), eta) in enumerate(zip(bounds, eta_mutation_values)):
                gene = [individual[i]]
                tools.mutPolynomialBounded(
                    gene,
                    eta=eta,
                    low=[low],
                    up=[up],
                    indpb=indpb,
                )
                individual[i] = gene[0]
            return individual,

        def roulette_selection(
            population: list[creator.Individual],
            k: int,
        ) -> list[creator.Individual]:
            total_fitness = sum(float(ind.fitness.values[0]) for ind in population)
            if total_fitness > 0.0:
                return tools.selRoulette(population, k)
            return random.choices(population, k=k)

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", bounded_simulated_binary_crossover)
        toolbox.register("mutate", bounded_polynomial_mutation)
        if cfg.selection_method == "roulette":
            toolbox.register("select", roulette_selection)
        else:
            toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

        def evaluate_invalid_individuals(invalid_individuals: list[creator.Individual]) -> None:
            if not invalid_individuals:
                return
            vectors = [list(ind) for ind in invalid_individuals]
            results = evaluator.evaluate(vectors)

            for ind, (score, resolved, shift_values) in zip(invalid_individuals, results):
                ind.fitness.values = (float(score),)
                ind.resolved = resolved
                ind.best_shift_values = shift_values

        with ParallelEvaluator(objective_fn, self.evaluator_config) as evaluator:
            population = toolbox.population(n=cfg.population_size)
            hall_of_fame = tools.HallOfFame(cfg.hall_of_fame_size)
            history = []
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]
            evaluate_invalid_individuals(invalid_individuals)

            hall_of_fame.update(population)
            generation_data = []
            # Record the initial population so downstream analysis can compare
            # the starting search distribution with later generations.
            for ind in population:
                individual_data = {
                    "vector": list(ind),
                    "score": float(ind.fitness.values[0]),
                    "resolved": getattr(ind, "resolved", {}),
                    "shift_values": getattr(ind, "best_shift_values", [])
                }
                generation_data.append(individual_data)
            history.append({"generation": 0, "individuals": generation_data})
            best_score_so_far = float(hall_of_fame[0].fitness.values[0])
            no_improve_generations = 0
            for generation in range(1, cfg.n_generations + 1):
                n_elite = min(cfg.elite_size, cfg.population_size - 1)
                elites = [toolbox.clone(ind) for ind in tools.selBest(population, n_elite)]

                offspring = toolbox.select(population, cfg.population_size - n_elite)
                offspring = list(map(toolbox.clone, offspring))

                for i in range(1, len(offspring), 2):
                    if random.random() < cfg.crossover_probability:
                        toolbox.mate(offspring[i - 1], offspring[i])
                        if hasattr(offspring[i - 1].fitness, "values"):
                            del offspring[i - 1].fitness.values
                        if hasattr(offspring[i].fitness, "values"):
                            del offspring[i].fitness.values

                for ind in offspring:
                    if random.random() < cfg.mutation_probability:
                        toolbox.mutate(ind)
                        if hasattr(ind.fitness, "values"):
                            del ind.fitness.values

                invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
                evaluate_invalid_individuals(invalid_individuals)

                population = elites + offspring
                hall_of_fame.update(population)
                # Record each generation after selection, crossover, mutation,
                # and elite reinsertion.
                generation_data = []
                for ind in population:
                    individual_data = {
                        "vector": list(ind),
                        "score": float(ind.fitness.values[0]),
                        "resolved": getattr(ind, "resolved", {}),
                        "shift_values": getattr(ind, "best_shift_values", [])
                    }
                    generation_data.append(individual_data)
                history.append({"generation": generation, "individuals": generation_data})

                current_best = float(hall_of_fame[0].fitness.values[0])
                if current_best > best_score_so_far + cfg.early_stop_min_delta:
                    best_score_so_far = current_best
                    no_improve_generations = 0
                else:
                    no_improve_generations += 1

                if (
                    cfg.early_stop_patience is not None
                    and no_improve_generations >= cfg.early_stop_patience
                ):
                    break

        best_individual = hall_of_fame[0]
        best_vector = list(best_individual)
        best_score = float(best_individual.fitness.values[0])
        best_resolved = getattr(best_individual, "resolved", {})
        best_shift_values = getattr(best_individual, "best_shift_values", [])
        
        return OptimizationResult(
            best_score=best_score,
            best_resolved=best_resolved,
            best_shift_values=best_shift_values,
            history=history,
            variable_names=variable_names
        )
