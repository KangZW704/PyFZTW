"""Best-model event alignment survives GA/PSO and CLI pickle serialization."""

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PyFZTW.cli import _save_run_result
from PyFZTW.inverse.inversion import run_multi_trace_inversion
from PyFZTW.io.parameters import ParameterSpace, ParameterSpec
from PyFZTW.optimizers.base import OptimizationResult
from PyFZTW.optimizers.deap import DeapOptimizer, DeapOptimizerConfig
from PyFZTW.optimizers.pso import PsoOptimizer, PsoOptimizerConfig


class ToyObjective:
    def __init__(self, event_ids):
        self.event_ids = event_ids

    def evaluate_with_meta(self, vector):
        x = float(vector[0])
        return 1 - (x - 0.4) ** 2, {"v1": x}, [
            x + index + 1 for index in range(len(self.event_ids))
        ]


def problem_for(event_ids):
    crust = ParameterSpace([ParameterSpec("v1", 0, minimum=0, maximum=1)])
    return SimpleNamespace(
        crust_parameter_space=lambda: crust,
        earthquakes=lambda: [
            SimpleNamespace(parameter_space=ParameterSpace([])) for _ in event_ids
        ],
        event_ids=lambda: event_ids,
    )


class ResultEventAlignmentTests(unittest.TestCase):
    def test_ga_and_pso_best_alignment_saved_by_cli(self):
        for event_ids in (["1"], ["10", "2"]):
            optimizers = [
                DeapOptimizer(DeapOptimizerConfig(
                    population_size=8, n_generations=3, random_seed=17,
                    selection_method="tournament",
                )),
                PsoOptimizer(PsoOptimizerConfig(
                    population_size=8, n_iterations=3, random_seed=17,
                )),
            ]
            for optimizer in optimizers:
                with self.subTest(events=event_ids, optimizer=type(optimizer).__name__):
                    with patch("PyFZTW.inverse.inversion.build_multi_trace_objective",
                               return_value=ToyObjective(event_ids)):
                        result = run_multi_trace_inversion(
                            problem_for(event_ids), None, None, optimizer,
                        )
                    with tempfile.TemporaryDirectory() as directory:
                        path = _save_run_result(
                            Path(directory) / "result.pkl", 1,
                            {"run_index": 1, "result": result},
                        )
                        with path.open("rb") as stream:
                            saved = pickle.load(stream)
                    x = saved["best_resolved"]["v1"]
                    self.assertAlmostEqual(saved["best_score"], 1 - (x - 0.4) ** 2)
                    self.assertEqual(saved["best_event_window_start_s"], {
                        event_id: x + index + 1
                        for index, event_id in enumerate(event_ids)
                    })
                    for generation in saved["history"]:
                        for individual in generation["individuals"]:
                            self.assertEqual(set(individual), {"score", "resolved"})
                    if isinstance(optimizer, PsoOptimizer):
                        self.assertEqual(saved["optimizer"], "pso")
                        self.assertIn("pbest_population", saved["pso"])
                        self.assertIn("gbest_history", saved["pso"])

    def test_missing_shifts_remain_empty(self):
        result = OptimizationResult(best_score=0.9)
        self.assertEqual(result.to_dict()["best_event_window_start_s"], {})

    def test_mismatched_event_count_is_rejected(self):
        optimizer = SimpleNamespace(optimize=lambda **kwargs: OptimizationResult(
            best_score=0.9, best_shift_values=[1.2],
        ))
        with patch("PyFZTW.inverse.inversion.build_multi_trace_objective",
                   return_value=ToyObjective(["10", "2"])):
            with self.assertRaisesRegex(ValueError, "one value per event"):
                run_multi_trace_inversion(problem_for(["10", "2"]), None, None, optimizer)


if __name__ == "__main__":
    unittest.main()
