#!/usr/bin/env python3
"""Read and explain PyFZTW inversion result pickle files.

The current CLI stores a compact dictionary produced by
OptimizationResult.to_dict():

    {
        "best_score": float,
        "best_resolved": {...},
        "history": [
            {
                "generation": int,
                "individuals": [
                    {"score": float, "resolved": {...}},
                    ...
                ],
            },
            ...
        ],
    }

This script uses only the Python standard library, so it can run on Linux
without installing the PyFZTW package.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_PKL = Path("examples/syn_test/inversion_results_v1_run01.pkl")


PARAMETER_INFO: dict[str, tuple[str, str]] = {
    "v1": ("km/s", "S-wave velocity of the left host rock."),
    "vfr": ("dimensionless", "Fault-zone to left-host S-wave velocity ratio."),
    "v3r": ("dimensionless", "Right-host to left-host S-wave velocity ratio."),
    "q1": ("dimensionless", "Quality factor of the left host rock."),
    "qf": ("dimensionless", "Quality factor of the low-velocity fault zone."),
    "q3": ("dimensionless", "Quality factor of the right host rock."),
    "density": ("g/cm^3", "Shared density in legacy model results."),
    "widthf": ("km", "Total width of the low-velocity fault zone."),
    "edge": ("km", "Fault-normal coordinate of the left fault-zone boundary."),
    "x": ("dimensionless", "Fault-normal SH source position measured from edge as a fraction of widthf."),
    "y": ("km", "Along-fault source coordinate."),
    "z": ("km", "Source depth."),
}


TOP_LEVEL_INFO: dict[str, str] = {
    "best_score": "Highest fitness found by the optimizer. Event fitness is 0.5 * (1 + correlation). Higher is better.",
    "best_resolved": "Physical parameters of the best model, including fixed and fitted parameters.",
    "history": "Compact optimization history by generation, with candidate scores and resolved parameters.",
}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def parameter_base_name(name: str) -> tuple[str | None, str]:
    """Return optional event id and base parameter name.

    Multi-event inversions store event-local parameters as '<event_id>_<name>',
    for example '1_z' or '38690767_x'. Crust parameters are stored without an
    event prefix.
    """
    if "_" not in name:
        return None, name

    prefix, suffix = name.rsplit("_", 1)
    if suffix in {"x", "y", "z"} and prefix:
        return prefix, suffix
    return None, name


def format_number(value: Any, digits: int) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.{digits}g}"
        return str(value)
    return str(value)


def iter_scores(history: list[Any]) -> list[tuple[int | None, float]]:
    scores: list[tuple[int | None, float]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        generation = entry.get("generation")
        individuals = entry.get("individuals", [])
        if not isinstance(individuals, list):
            continue
        for individual in individuals:
            if not isinstance(individual, dict):
                continue
            score = individual.get("score")
            if isinstance(score, (int, float)) and math.isfinite(float(score)):
                gen_value = int(generation) if isinstance(generation, int) else None
                scores.append((gen_value, float(score)))
    return scores


def generation_stats(history: list[Any]) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        generation = entry.get("generation")
        individuals = entry.get("individuals", [])
        if not isinstance(individuals, list):
            continue
        scores = [
            float(ind.get("score"))
            for ind in individuals
            if isinstance(ind, dict)
            and isinstance(ind.get("score"), (int, float))
            and math.isfinite(float(ind.get("score")))
        ]
        if not scores:
            rows.append(
                {
                    "generation": generation if isinstance(generation, int) else None,
                    "count": 0,
                    "min": None,
                    "mean": None,
                    "max": None,
                }
            )
            continue
        rows.append(
            {
                "generation": generation if isinstance(generation, int) else None,
                "count": len(scores),
                "min": min(scores),
                "mean": mean(scores),
                "max": max(scores),
            }
        )
    return rows


def print_heading(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_top_level(data: dict[str, Any]) -> None:
    print_heading("1. Top-level pickle fields")
    for key in data:
        meaning = TOP_LEVEL_INFO.get(key, "Unrecognized field. The script displays its value or a structural summary.")
        print(f"- {key}: {meaning}")


def print_best_resolved(best_resolved: dict[str, Any], digits: int) -> None:
    print_heading("2. Best-model parameters")
    if not best_resolved:
        print("best_resolved is empty.")
        return

    rows: list[tuple[str, str, str, str, str]] = []
    for name, value in best_resolved.items():
        event_id, base = parameter_base_name(name)
        unit, meaning = PARAMETER_INFO.get(base, ("", "Unrecognized parameter. See the input file or forward model."))
        scope = f"Event {event_id}" if event_id is not None else "Shared/crust"
        rows.append((name, scope, format_number(value, digits), unit, meaning))

    widths = [
        max(len(row[i]) for row in rows + [("Parameter", "Scope", "Value", "Unit", "Description")])
        for i in range(4)
    ]
    print(
        f"{'Parameter':<{widths[0]}}  {'Scope':<{widths[1]}}  "
        f"{'Value':>{widths[2]}}  {'Unit':<{widths[3]}}  Description"
    )
    print(
        f"{'-' * widths[0]}  {'-' * widths[1]}  "
        f"{'-' * widths[2]}  {'-' * widths[3]}  {'-' * 24}"
    )
    for name, scope, value, unit, meaning in rows:
        print(f"{name:<{widths[0]}}  {scope:<{widths[1]}}  {value:>{widths[2]}}  {unit:<{widths[3]}}  {meaning}")

    v1 = best_resolved.get("v1")
    vfr = best_resolved.get("vfr")
    v3r = best_resolved.get("v3r")
    edge = best_resolved.get("edge")
    widthf = best_resolved.get("widthf")
    if all(isinstance(v, (int, float)) for v in (v1, vfr, v3r)):
        print()
        print("Derived velocities:")
        print(f"- fault-zone velocity v2 = v1 * vfr = {format_number(float(v1) * float(vfr), digits)} km/s")
        print(f"- media 3/4 velocity v3 = v4 = v1 * v3r = {format_number(float(v1) * float(v3r), digits)} km/s")
    if isinstance(edge, (int, float)) and isinstance(widthf, (int, float)):
        print()
        print("Derived geometry:")
        print(
            "- fault-zone x range = "
            f"[edge, edge + widthf] = [{format_number(edge, digits)}, "
            f"{format_number(float(edge) + float(widthf), digits)}] km"
        )


def print_history_summary(history: list[Any], digits: int, n_last: int) -> None:
    print_heading("3. Optimization history")
    if not history:
        print("No saved history.")
        return

    scores = iter_scores(history)
    stats = generation_stats(history)
    pop_sizes = sorted({int(row["count"]) for row in stats if isinstance(row.get("count"), int)})
    print(f"- Generations: {len(history)}")
    print(f"- Total candidates: {len(scores)}")
    if pop_sizes:
        print(f"- Population size by generation: {', '.join(str(v) for v in pop_sizes)}")
    if scores:
        best_gen, best_score = max(scores, key=lambda item: item[1])
        worst_gen, worst_score = min(scores, key=lambda item: item[1])
        print(f"- Highest recorded score: {format_number(best_score, digits)} (generation={best_gen})")
        print(f"- Lowest recorded score: {format_number(worst_score, digits)} (generation={worst_gen})")

    print()
    print(f"Statistics for the last {min(n_last, len(stats))} generations:")
    print("generation  count       min      mean       max")
    print("----------  -----  --------  --------  --------")
    for row in stats[-n_last:]:
        print(
            f"{str(row['generation']):>10}  "
            f"{int(row['count']):>5}  "
            f"{format_number(row['min'], digits):>8}  "
            f"{format_number(row['mean'], digits):>8}  "
            f"{format_number(row['max'], digits):>8}"
        )


def export_best_resolved_csv(path: Path, best_resolved: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["parameter", "scope", "value", "unit", "meaning"])
        for name, value in best_resolved.items():
            event_id, base = parameter_base_name(name)
            unit, meaning = PARAMETER_INFO.get(base, ("", "Unrecognized parameter. See the input file or forward model."))
            scope = f"event:{event_id}" if event_id is not None else "shared/crust"
            writer.writerow([name, scope, value, unit, meaning])


def export_history_csv(path: Path, history: list[Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation", "individual_index", "score", "parameter", "value"])
        for entry in history:
            if not isinstance(entry, dict):
                continue
            generation = entry.get("generation")
            individuals = entry.get("individuals", [])
            if not isinstance(individuals, list):
                continue
            for index, individual in enumerate(individuals):
                if not isinstance(individual, dict):
                    continue
                score = individual.get("score")
                resolved = individual.get("resolved", {})
                if not isinstance(resolved, dict):
                    writer.writerow([generation, index, score, "", ""])
                    continue
                for name, value in resolved.items():
                    writer.writerow([generation, index, score, name, value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a PyFZTW inversion result pickle and explain best_score, best_resolved, and history."
    )
    parser.add_argument(
        "pkl",
        nargs="?",
        default=DEFAULT_PKL,
        type=Path,
        help=f"Input pickle file (default: {DEFAULT_PKL}).",
    )
    parser.add_argument("--digits", type=int, default=8, help="Significant digits to display (default: 8).")
    parser.add_argument("--last-generations", type=int, default=8, help="Number of final generations to summarize (default: 8).")
    parser.add_argument("--export-best-csv", type=Path, help="Export best_resolved as a CSV table.")
    parser.add_argument("--export-history-csv", type=Path, help="Export history in long-form CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pkl_path = args.pkl
    if not pkl_path.exists():
        raise SystemExit(f"File not found: {pkl_path}")

    data = load_pickle(pkl_path)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a top-level dictionary, got {type(data).__name__}.")

    print(f"File: {pkl_path}")
    print(f"Type: {type(data).__name__}")

    best_score = data.get("best_score")
    if isinstance(best_score, (int, float)):
        print(f"best_score: {format_number(float(best_score), args.digits)}")

    print_top_level(data)

    best_resolved = data.get("best_resolved", {})
    if not isinstance(best_resolved, dict):
        best_resolved = {}
    print_best_resolved(best_resolved, args.digits)

    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    print_history_summary(history, args.digits, max(1, args.last_generations))

    if args.export_best_csv:
        export_best_resolved_csv(args.export_best_csv, best_resolved)
        print(f"\nExported best-model parameters to {args.export_best_csv}")
    if args.export_history_csv:
        export_history_csv(args.export_history_csv, history)
        print(f"\nExported optimization history to {args.export_history_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
