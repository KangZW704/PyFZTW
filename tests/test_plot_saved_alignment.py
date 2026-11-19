"""Plot alignment must preserve saved timing and interstation amplitudes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from PyFZTW.forward.base import SyntheticTrace
from PyFZTW.inverse.objective import find_best_event_time_shift
from plot_codes import reflow_ga_pso_side_by_side as synthetic_plot
from plot_codes import plot_result_rid_v3 as ridgecrest_plot


MODULES = (ridgecrest_plot, synthetic_plot)


def observation(station, time, data):
    return SimpleNamespace(station=station, windowed_data=lambda: (time, data))


class SavedPlotAlignmentTests(unittest.TestCase):
    def test_matches_objective_grids_for_different_sampling_rates(self):
        syn_time = np.arange(0, 8, 0.125)
        observations = []
        synthetics = []
        for index, dt in enumerate((0.25, 0.5)):
            values = (index + 1) * (np.sin(syn_time * 3.1) + 0.3 * np.cos(syn_time * 5.7))
            obs_time = 10 + np.arange(0, 2 + dt / 2, dt)
            obs = np.interp(1.5 + obs_time - 10, syn_time, values)
            observations.append(observation(str(index), obs_time, obs))
            synthetics.append(SyntheticTrace(time=syn_time, disp=values))
        start, _, reference_obs, reference_syn = find_best_event_time_shift(observations, synthetics)
        traces = synthetic_plot.align_saved_event(observations, synthetics, start)
        for trace, obs, syn in zip(traces, reference_obs, reference_syn):
            np.testing.assert_array_equal(trace[2], obs)
            np.testing.assert_array_equal(trace[3], syn)
            self.assertEqual(trace[1][0], 10)
            self.assertEqual(trace[4], start)

    def test_uses_supplied_start_even_when_another_shift_fits_better(self):
        obs = [observation("A", np.array([10., 11., 12.]), np.array([1., 2., 3.]))]
        syn = [SyntheticTrace(time=np.arange(6.), disp=np.array([0., 0., 1., 2., 3., 0.]))]
        for module in MODULES:
            with self.subTest(module=module.__name__):
                trace = module.align_saved_event(obs, syn, 1.)[0]
                np.testing.assert_array_equal(trace[3], [0., 1., 2.])
                np.testing.assert_array_equal(trace[1], [10., 11., 12.])
                for invalid in (float("nan"), -1., 4.):
                    with self.assertRaises(ValueError):
                        module.align_saved_event(obs, syn, invalid)

    def test_separate_event_peaks_preserve_amplitude_ratios_and_polarity(self):
        traces = [
            ("A", np.arange(2.), np.array([-2., 1.]), np.array([-3., 1.]), 0.),
            ("B", np.arange(2.), np.array([-4., 1.]), np.array([-12., 2.]), 0.),
        ]
        observed_peak, synthetic_peak = synthetic_plot.event_peak_scales(traces)
        self.assertEqual(observed_peak, 4.)
        self.assertEqual(synthetic_peak, 12.)
        self.assertEqual(ridgecrest_plot.observed_event_peak(traces), observed_peak)
        fig, ax = synthetic_plot.plt.subplots()
        try:
            synthetic_plot.draw_waveforms(ax, traces)
            np.testing.assert_array_equal(ax.lines[1].get_ydata(), traces[0][3] / 12.)
            np.testing.assert_allclose(ax.lines[3].get_ydata() - .75, traces[1][3] / 12., rtol=0, atol=1e-15)
        finally:
            synthetic_plot.plt.close(fig)

    def test_missing_saved_start_is_not_replaced_by_a_search(self):
        project = SimpleNamespace(seismograms=[SimpleNamespace(event_id="1")])
        with patch.object(synthetic_plot, "load_project", return_value=project):
            with self.assertRaisesRegex(ValueError, "best_event_window_start_s"):
                synthetic_plot.prepare_waveforms({"best_resolved": {}, "best_score": 1.})

    def test_ridgecrest_saved_field_supports_dict_and_result_object(self):
        module = ridgecrest_plot
        for result_type in (dict, SimpleNamespace):
            for start in (0., 1.5):
                result = result_type(best_event_window_start_s={module.ref_evid: start})
                self.assertEqual(module.saved_event_start(result), start)
            for starts in ({}, {"other_event": 1.}, {module.ref_evid: float("nan")}):
                with self.assertRaises(ValueError):
                    module.saved_event_start(result_type(best_event_window_start_s=starts))

    def test_ridgecrest_preparation_uses_saved_model_and_alignment(self):
        module = ridgecrest_plot
        gram = observation("B416", np.array([10., 11., 12.]), np.array([1., 2., 3.]))
        gram.event_id = module.ref_evid
        project = SimpleNamespace(seismograms=[gram])
        synthetic = SyntheticTrace(time=np.arange(6.), disp=np.array([0., 1., 2., 3., 0., 0.]))
        best = {f"{module.ref_evid}_{axis}": value for axis, value in zip("xyz", (.2, 0., 5.))}
        result = {"best_resolved": best, "best_score": 1.,
                  "best_event_window_start_s": {module.ref_evid: 1.}}
        with patch.object(module, "load_project", return_value=project), \
                patch.object(module, "HS4PDISForwardModel") as forward:
            forward.return_value.run.return_value = synthetic
            traces = module.prepare_waveforms(result)
            np.testing.assert_array_equal(traces[0][3], [1., 2., 3.])
            passed_params = forward.return_value.run.call_args.args[2]
            self.assertEqual([passed_params[axis] for axis in "xyz"], [.2, 0., 5.])
            self.assertEqual(traces[0][4], 1.)


if __name__ == "__main__":
    unittest.main()
