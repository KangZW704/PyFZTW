"""Command-line workflow for repeated PyFZTW inversions.

The CLI loads project input files, builds a forward model and optimizer, runs
one or more inversion experiments, and writes each run's compact result to a
versioned pickle file.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

from .forward.hs4pdis import HS4PDISForwardConfig, HS4PDISForwardModel
from .inverse.inversion import run_inversion_for_seismograms
from .inverse.objective import WindowedCorrelationFitness
from .io.project import load_project
from .optimizers.base import OptimizationResult, Optimizer
from .optimizers.parallel import ParallelEvaluatorConfig


OptimizerFactory = Callable[["BatchInversionConfig", int | None], Optimizer]


@dataclass(frozen=True)
class BatchInversionConfig:
    """All file paths and runtime options needed for batch inversion runs."""
    crust_file: Path
    earthquake_file: Path
    station_file: Path
    seismogram_files: tuple[Path, ...]
    output: Path
    optimizer: str = "deap"
    optimizer_config: dict[str, Any] | None = None
    jobs: int = 1
    parallel_backend: str = "thread"
    process_start_method: str = "spawn"
    n_runs: int = 20
    parallel_runs: int = 1
    seed_base: int | None = None
    npts: int = 4096
    lt_padding: float = 1.0


def _next_versioned_output(base_dir: Path, stem: str = "inversion_results") -> Path:
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+)\.pkl$")
    max_version = 0
    for path in base_dir.glob(f"{stem}_v*.pkl"):
        match = pattern.match(path.name)
        if match:
            max_version = max(max_version, int(match.group(1)))
    return base_dir / f"{stem}_v{max_version + 1}.pkl"


def _single_run_output_path(base_output: Path, run_index: int, total_runs: int) -> Path:
    width = max(2, len(str(total_runs)))
    suffix = base_output.suffix or ".pkl"
    return base_output.with_name(f"{base_output.stem}_run{run_index + 1:0{width}d}{suffix}")


def _parse_toml_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.startswith("{") and value.endswith("}"):
        items = value[1:-1].strip()
        result: dict[str, Any] = {}
        if not items:
            return result
        for item in items.split(","):
            if not item.strip():
                continue
            if "=" in item:
                key, nested_value = item.split("=", 1)
            elif ":" in item:
                key, nested_value = item.split(":", 1)
            else:
                raise ValueError(f"expected key = value in inline table item: {item}")
            result[key.strip()] = _parse_toml_scalar(nested_value)
        return result
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].strip()
        if not items:
            return []
        return [
            _parse_toml_scalar(item)
            for item in items.split(",")
            if item.strip()
        ]
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "none"}:
        return None
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_flat_toml_config(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_section: str | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("["):
            if not stripped.endswith("]") or stripped.startswith("[["):
                raise ValueError(f"{path} line {line_no}: unsupported TOML section syntax")
            current_section = stripped[1:-1].strip()
            if not current_section:
                raise ValueError(f"{path} line {line_no}: empty section name")
            section = config.setdefault(current_section, {})
            if not isinstance(section, dict):
                raise ValueError(
                    f"{path} line {line_no}: section conflicts with scalar key "
                    f"{current_section!r}"
                )
            continue
        if "=" not in stripped:
            raise ValueError(f"{path} line {line_no}: expected key = value")
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path} line {line_no}: empty key")
        if current_section is None:
            config[key] = _parse_toml_scalar(value)
        else:
            section = config[current_section]
            if not isinstance(section, dict):
                raise ValueError(
                    f"{path} line {line_no}: section conflicts with scalar key "
                    f"{current_section!r}"
                )
            section[key] = _parse_toml_scalar(value)
    return config


def load_optimizer_config(path: Path | None) -> dict[str, Any]:
    """Load optimizer-specific options from JSON or a flat TOML file."""
    if path is None:
        return {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("optimizer JSON config must contain an object")
        return dict(data)
    if suffix in {".toml", ".tml"}:
        try:
            import tomllib
        except ModuleNotFoundError:
            return _load_flat_toml_config(path)

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("optimizer TOML config must contain key/value pairs")
        return dict(data)
    raise ValueError("--optimizer-config must be a .json or .toml file")


def _make_dataclass_config(
    config_cls: type,
    values: dict[str, Any] | None,
    *,
    random_seed: int | None,
) -> Any:
    kwargs = dict(values or {})
    if random_seed is not None:
        kwargs["random_seed"] = random_seed

    valid_names = {field.name for field in fields(config_cls)}
    unknown = sorted(set(kwargs) - valid_names)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown optimizer config option(s) for {config_cls.__name__}: {joined}")
    return config_cls(**kwargs)


def build_deap_optimizer(config: BatchInversionConfig, run_seed: int | None) -> Optimizer:
    """Create the default DEAP optimizer for one run."""
    from .optimizers.deap import DeapOptimizer, DeapOptimizerConfig

    return DeapOptimizer(
        _make_dataclass_config(
            DeapOptimizerConfig,
            config.optimizer_config,
            random_seed=run_seed,
        ),
        evaluator_config=ParallelEvaluatorConfig(
            n_jobs=config.jobs,
            backend=config.parallel_backend,
            process_start_method=config.process_start_method,
        ),
    )


def build_pso_optimizer(config: BatchInversionConfig, run_seed: int | None) -> Optimizer:
    """Create a particle swarm optimizer for one run."""
    from .optimizers.pso import PsoOptimizer, PsoOptimizerConfig

    return PsoOptimizer(
        _make_dataclass_config(
            PsoOptimizerConfig,
            config.optimizer_config,
            random_seed=run_seed,
        ),
        evaluator_config=ParallelEvaluatorConfig(
            n_jobs=config.jobs,
            backend=config.parallel_backend,
            process_start_method=config.process_start_method,
        ),
    )


def _load_optimizer_factory(spec: str) -> OptimizerFactory:
    if spec == "deap":
        return build_deap_optimizer
    if spec == "pso":
        return build_pso_optimizer

    if ":" not in spec:
        raise ValueError(
            "optimizer must be 'deap' or an import path like 'module.submodule:factory'"
        )

    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"Optimizer factory is not callable: {spec}")
    return factory


def run_single_inversion(
    config: BatchInversionConfig,
    optimizer: Optimizer,
) -> tuple[OptimizationResult, float]:
    """Load inputs, run one inversion, and return the result plus elapsed time."""
    project = load_project(
        crust_file=config.crust_file,
        earthquake_file=config.earthquake_file,
        station_file=config.station_file,
        seismogram_files=list(config.seismogram_files),
    )
    forward_model = HS4PDISForwardModel(
        config=HS4PDISForwardConfig(
            npts=config.npts,
            lt_padding=config.lt_padding,
        )
    )
    objective = WindowedCorrelationFitness()

    t0 = time.time()
    result = run_inversion_for_seismograms(
        project=project,
        seismograms=project.seismograms,
        forward_model=forward_model,
        objective=objective,
        optimizer=optimizer,
    )
    return result, time.time() - t0


def _run_one_experiment(payload: tuple[BatchInversionConfig, int]) -> dict:
    config, run_index = payload
    run_seed = None if config.seed_base is None else config.seed_base + run_index
    optimizer_factory = _load_optimizer_factory(config.optimizer)
    optimizer = optimizer_factory(config, run_seed)
    result, elapsed = run_single_inversion(config=config, optimizer=optimizer)
    return {
        "run_index": run_index,
        "seed": run_seed,
        "elapsed_s": float(elapsed),
        "best_score": float(result.best_score),
        "best_resolved": dict(result.best_resolved or {}),
        "result": result,
    }


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


def run_repeated_inversion(config: BatchInversionConfig) -> list[Path]:
    """Run the configured experiments and save one pickle result per run."""
    saved_paths: list[Path] = []
    payloads = [(config, i) for i in range(config.n_runs)]

    if config.parallel_runs == 1:
        for payload in payloads:
            item = _run_one_experiment(payload)
            saved_paths.append(_save_run_result(config.output, config.n_runs, item))
    else:
        with ProcessPoolExecutor(max_workers=config.parallel_runs) as executor:
            futures = [executor.submit(_run_one_experiment, payload) for payload in payloads]
            for future in as_completed(futures):
                item = future.result()
                saved_paths.append(_save_run_result(config.output, config.n_runs, item))

    saved_paths.sort()
    return saved_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crust-file", type=Path, required=True)
    parser.add_argument("--earthquake-file", type=Path, required=True)
    parser.add_argument("--station-file", type=Path, required=True)
    parser.add_argument("--seismogram-files", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--optimizer",
        default="deap",
        help="optimizer choice: 'deap', 'pso', or 'module.submodule:factory'",
    )
    parser.add_argument(
        "--optimizer-config",
        type=Path,
        default=None,
        help="optimizer-specific .json or flat .toml config file",
    )
    parser.add_argument("--jobs", type=int, default=4, help="parallel jobs inside one optimizer run")
    parser.add_argument("--backend", default="process", choices=["thread", "process"])
    parser.add_argument("--process-start-method", default="spawn")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--npts", type=int, default=4096)
    parser.add_argument("--lt-padding", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="base output path; actual files are saved as *_runXX.pkl",
    )
    return parser.parse_args()


def _make_config(args: argparse.Namespace) -> BatchInversionConfig:
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive")
    if args.parallel_runs <= 0:
        raise ValueError("--parallel-runs must be positive")
    parallel_runs = min(args.parallel_runs, args.n_runs)

    output = args.output
    if output is None:
        output = _next_versioned_output(args.crust_file.parent)
    optimizer_config = load_optimizer_config(args.optimizer_config)

    return BatchInversionConfig(
        crust_file=args.crust_file,
        earthquake_file=args.earthquake_file,
        station_file=args.station_file,
        seismogram_files=tuple(args.seismogram_files),
        output=output,
        optimizer=args.optimizer,
        optimizer_config=optimizer_config,
        jobs=args.jobs,
        parallel_backend=args.backend,
        process_start_method=args.process_start_method,
        n_runs=args.n_runs,
        parallel_runs=parallel_runs,
        seed_base=args.seed_base,
        npts=args.npts,
        lt_padding=args.lt_padding,
    )


def main() -> None:
    """Parse command-line arguments and execute repeated inversion runs."""
    args = _parse_args()
    config = _make_config(args)

    print(
        f"Running repeated inversion: runs={config.n_runs}, parallel_runs={config.parallel_runs}, "
        f"optimizer={config.optimizer}, jobs={config.jobs}, backend={config.parallel_backend}"
    )

    t0 = time.time()
    payloads = [(config, i) for i in range(config.n_runs)]
    saved_paths: list[Path] = []

    if config.parallel_runs == 1:
        for run_index, payload in enumerate(payloads, start=1):
            item = _run_one_experiment(payload)
            saved_path = _save_run_result(config.output, config.n_runs, item)
            saved_paths.append(saved_path)
            print(
                f"[{run_index}/{config.n_runs}] elapsed={item['elapsed_s']:.3f}s "
                f"best_score={item['best_score']:.6f} saved={saved_path.name}"
            )
    else:
        with ProcessPoolExecutor(max_workers=config.parallel_runs) as executor:
            futures = [executor.submit(_run_one_experiment, payload) for payload in payloads]
            for done_count, future in enumerate(as_completed(futures), start=1):
                item = future.result()
                saved_path = _save_run_result(config.output, config.n_runs, item)
                saved_paths.append(saved_path)
                print(
                    f"[{done_count}/{config.n_runs}] run={item['run_index'] + 1} "
                    f"elapsed={item['elapsed_s']:.3f}s best_score={item['best_score']:.6f} "
                    f"saved={saved_path.name}"
                )

    total_elapsed = time.time() - t0
    print(f"\nAll runs completed in {total_elapsed:.3f} seconds")
    print(f"Per-run files saved as: {config.output.stem}_runXX{config.output.suffix}")


if __name__ == "__main__":
    main()
