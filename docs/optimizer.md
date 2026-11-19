# PyFZTW optimizer development

This guide covers the optimizer interface, the recommended history format, and the differences between DEAP and PSO records.

## Optimizer interface

Implement the `Optimizer` protocol defined in `PyFZTW/optimizers/base.py`.

```python
class Optimizer(Protocol):
    def optimize(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[list[float]], float],
    ) -> OptimizationResult:
        ...
```

The inputs and return value are defined as follows.

- `parameter_space.bounds()` returns the lower and upper bounds for each free parameter.
- `parameter_space.variable_names()` returns the free parameter names.
- `objective_fn(vector)` evaluates a parameter vector and returns a score.
- `optimize()` returns an `OptimizationResult`.

## OptimizationResult

The optimizer returns an `OptimizationResult` with fields such as these.

```python
@dataclass
class OptimizationResult:
    best_score: float
    best_resolved: dict[str, float] | None = None
    best_shift_values: list[float] | None = None
    history: list[dict[str, Any]] | None = None
    variable_names: list[str] | None = None
```

Field meanings

- `best_score` is the highest objective score found.
- `best_resolved` contains the physical parameters of the best model.
- `best_shift_values` records the best model's alignment values in event order.
- `history` records candidate solutions at each generation or iteration.
- `variable_names` names the dimensions of the optimization vector.

## Recommended history structure

Optimizers share a set of core history fields so postprocessing can read their candidate records.

Recommended structure

```python
history = [
    {
        "generation": 0,
        "individuals": [
            {
                "vector": [...],
                "score": 0.0,
                "resolved": {...},
                "shift_values": [...],
            }
        ],
    }
]
```

Use these fields across optimizers.

- `generation` is the generation or iteration number.
- `individuals` contains the candidates for that generation.
- `vector` is the optimizer's parameter vector.
- `score` is the candidate's objective score.
- `resolved` contains its resolved physical parameters.
- `shift_values` records its alignment values in event order.

Add algorithm-specific fields as needed.

## History in the built-in optimizers

### DEAP

The DEAP optimizer in `PyFZTW/optimizers/deap.py` records the population at each generation.

Each individual record contains fields such as these.

```python
{
    "vector": [...],
    "score": ...,
    "resolved": {...},
    "shift_values": [...],
}
```

### PSO

The PSO optimizer in `PyFZTW/optimizers/pso.py` records the swarm at each iteration.

Each particle record contains fields such as these.

```python
{
    "vector": [...],
    "velocity": [...],
    "score": ...,
    "resolved": {...},
    "shift_values": [...],
    "pbest_score": ...,
}
```

PSO records additional particle state.

- `velocity` is the current particle velocity.
- `pbest_score` is the best score the particle has found.

Keep the common fields consistent and store algorithm-specific state in additional fields.

## Saved pickle contents

The CLI writes results through `_save_run_result()` in `PyFZTW/cli.py`.

```python
def _save_run_result(
    base_output: Path,
    total_runs: int,
    item: dict,
) -> Path:
    output_path = _single_run_output_path(base_output, item["run_index"], total_runs)
    result = item["result"]
    to_save = result.to_dict() if hasattr(result, "to_dict") else result
    with output_path.open("wb") as fh:
        pickle.dump(to_save, fh)
    return output_path
```

`OptimizationResult.to_dict()` determines the fields written to the pickle.

The saved history contains compact candidate records.

```python
{
    "score": ...,
    "resolved": {...},
}
```

The saved fields support postprocessing of scores and resolved parameters. The event-to-window-start mapping for the run's best model is stored in `best_event_window_start_s`. Per-candidate vectors, shifts, or velocities would require changes to serialization.

## Add an optimizer

Place each new optimizer in its own module.

```text
PyFZTW/optimizers/my_optimizer.py
```

Example implementation

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyFZTW.io.parameters import ParameterSpace
from PyFZTW.optimizers.base import Optimizer, OptimizationResult
from PyFZTW.optimizers.parallel import ParallelEvaluator, ParallelEvaluatorConfig


@dataclass
class MyOptimizerConfig:
    population_size: int = 100
    n_iterations: int = 50
    random_seed: int | None = None


class MyOptimizer(Optimizer):
    def __init__(
        self,
        config: MyOptimizerConfig | None = None,
        evaluator_config: ParallelEvaluatorConfig | None = None,
    ) -> None:
        self.config = config or MyOptimizerConfig()
        self.evaluator_config = evaluator_config or ParallelEvaluatorConfig()

    def optimize(
        self,
        parameter_space: ParameterSpace,
        objective_fn: Callable[[list[float]], float],
    ) -> OptimizationResult:
        bounds = parameter_space.bounds()
        variable_names = parameter_space.variable_names()

        best_score = float("-inf")
        best_resolved = {}
        best_shift_values = []
        history = []

        with ParallelEvaluator(objective_fn, self.evaluator_config) as evaluator:
            for generation in range(self.config.n_iterations + 1):
                vectors = []

                # Generate or update candidate vectors here.
                # vectors.append([...])

                results = evaluator.evaluate(vectors)
                individuals = []

                for vector, (score, resolved, shift_values) in zip(vectors, results):
                    score = float(score)
                    individuals.append({
                        "vector": list(vector),
                        "score": score,
                        "resolved": dict(resolved),
                        "shift_values": list(shift_values),
                    })

                    if score > best_score:
                        best_score = score
                        best_resolved = dict(resolved)
                        best_shift_values = list(shift_values)

                history.append({
                    "generation": generation,
                    "individuals": individuals,
                })

        return OptimizationResult(
            best_score=best_score,
            best_resolved=best_resolved,
            best_shift_values=best_shift_values,
            history=history,
            variable_names=variable_names,
        )
```

## Connect an optimizer to the CLI

`PyFZTW/cli.py` accepts two sets of options.

- Shared inversion options include inputs, output, parallel workers, repeat count, and forward sampling settings.
- Optimizer-specific options are read from the file passed to `--optimizer-config`.

Shared options

```text
--crust-file
--earthquake-file
--station-file
--seismogram-files
--output
--optimizer
--optimizer-config
--jobs
--backend
--process-start-method
--n-runs
--parallel-runs
--seed-base
--npts
--lt-padding
```

`--optimizer-config` accepts JSON or flat TOML. If omitted, the optimizer uses its own configuration defaults.

## Parallel evaluation

PyFZTW supports two levels of parallel evaluation.

- Within a run, `--jobs` controls concurrent candidate evaluations. The process backend is intended for CPU and Fortran work. The thread backend has lower startup overhead and can be used for debugging or where processes are unavailable. Use `--jobs 1` for serial execution or `--jobs -1` for all available CPUs.
- Across runs, `--parallel-runs` controls how many of the `--n-runs` inversions execute concurrently. Each run creates its own optimizer and forward model and writes a separate `*_runXX.pkl`.

Start with one level of parallelism. Setting `--parallel-runs M` and `--jobs N` together can create about `M * N` candidate workers and reduce throughput if they outnumber the available CPUs.

Example commands

```bash
# Debug with serial execution.
python -m PyFZTW ... --n-runs 1 --parallel-runs 1 --jobs 1

# Evaluate candidate models in parallel within one inversion.
python -m PyFZTW ... --n-runs 1 --parallel-runs 1 --jobs 8 --backend process

# Run independent inversions in parallel.
python -m PyFZTW ... --n-runs 20 --parallel-runs 4 --jobs 1 --seed-base 1000
```

Population-based optimizers should collect candidate parameter vectors and evaluate them in one batch.

```python
with ParallelEvaluator(objective_fn, self.evaluator_config) as evaluator:
    results = evaluator.evaluate(vectors)
```

DEAP evaluates offspring whose fitness is invalid. PSO evaluates all updated positions at each iteration. Reuse `ParallelEvaluator` so the CLI worker and backend settings apply and each result includes `score`, `resolved`, and `shift_values`.

DEAP configuration example

```toml
population_size = 200
n_generations = 50
crossover_probability = 0.7
mutation_probability = 0.2
tournament_size = 3
eta_crossover = 20.0
eta_mutation = 20.0
hall_of_fame_size = 1
elite_size = 1
early_stop_patience = 10
early_stop_min_delta = 1e-6
```

PSO configuration example

```toml
population_size = 200
n_iterations = 80
inertia_weight = 0.7
inertia_weight_end = 0.4
cognitive_coefficient = 1.5
social_coefficient = 1.5
max_velocity_ratio = 0.2
early_stop_patience = 10
early_stop_min_delta = 1e-6
maximize = true
```

Example command

```bash
python -m PyFZTW \
  --optimizer pso \
  --optimizer-config configs/pso.toml \
  --crust-file inverse_crust.txt \
  --earthquake-file inverse_eqs.txt \
  --station-file inverse_stations.txt \
  --seismogram-files data/*.disp
```

### Load a factory by module path

This approach can be used without editing the CLI.

Define a factory in the new optimizer module.

```python
def build_my_optimizer(config, run_seed):
    kwargs = dict(config.optimizer_config or {})
    if run_seed is not None:
        kwargs["random_seed"] = run_seed

    return MyOptimizer(
        MyOptimizerConfig(**kwargs),
        evaluator_config=ParallelEvaluatorConfig(
            n_jobs=config.jobs,
            backend=config.parallel_backend,
            process_start_method=config.process_start_method,
        ),
    )
```

Pass the factory's module path to the CLI.

```bash
python -m PyFZTW ... --optimizer PyFZTW.optimizers.my_optimizer:build_my_optimizer
```

### Register a built-in optimizer

To expose the optimizer by a short name, register it in the CLI.

Import the optimizer and configuration in `PyFZTW/cli.py`.

```python
from .optimizers.my_optimizer import MyOptimizer, MyOptimizerConfig
```

Add a factory function.

```python
def build_my_optimizer(config: BatchInversionConfig, run_seed: int | None) -> Optimizer:
    kwargs = dict(config.optimizer_config or {})
    if run_seed is not None:
        kwargs["random_seed"] = run_seed

    return MyOptimizer(
        MyOptimizerConfig(**kwargs),
        evaluator_config=ParallelEvaluatorConfig(
            n_jobs=config.jobs,
            backend=config.parallel_backend,
            process_start_method=config.process_start_method,
        ),
    )
```

Add the following entry to `_load_optimizer_factory()`.

```python
if spec == "myopt":
    return build_my_optimizer
```

Pass the registered name to the CLI.

```bash
python -m PyFZTW ... --optimizer myopt
```

## Implementation conventions

Follow these conventions when adding an optimizer.

- Keep the `history -> generation -> individuals` structure.
- Record `vector`, `score`, `resolved`, and `shift_values` for each candidate.
- Store algorithm-specific fields as needed, such as PSO's `velocity` and `pbest_score`.
- Populate `variable_names` when returning an `OptimizationResult`.
- Reuse `ParallelEvaluator` to obtain scores, resolved parameters, and alignment values from batch evaluations.
- If postprocessing reads only the pickle, check that `OptimizationResult.to_dict()` saves the fields it needs.
