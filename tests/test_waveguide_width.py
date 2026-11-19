"""Regression checks for the total-width mapping; run with unittest discover."""

import unittest
from types import SimpleNamespace

import numpy as np

from PyFZTW.forward.hs4pdis import HS4PDISForwardConfig, HS4PDISForwardModel


def inputs(source=0.3, receiver=0.5, width=0.25):
    parameters = dict(v1=3.6, vfr=0.65, v3r=0.9, q1=400.0, qf=20.0,
                      q3=250.0, widthf=width, edge=-0.05,
                      x=source, y=0.0, z=5.0,
                      rou1=2.7, rou2=2.3, rou3=2.6)
    project = SimpleNamespace(stations=[SimpleNamespace(
        name="S01", x=parameters["edge"] + receiver * width, y=0.0, z=0.0)])
    trace = SimpleNamespace(header=SimpleNamespace(
        station="S01", lt=6.0, window_end=5.0, maxfreq=12.0,
        gibbs=5.0, accuracy=1e-5))
    return project, trace, parameters


class CaptureModel(HS4PDISForwardModel):
    def __init__(self):
        self.config = HS4PDISForwardConfig()

    def _call_backend(self, **kwargs):
        self.call = kwargs
        kwargs["time"][:] = np.arange(len(kwargs["time"])) * kwargs["lt"] / len(kwargs["time"])


class WidthMappingTests(unittest.TestCase):
    def test_physical_geometry_and_host_properties(self):
        for source in (0.0, 0.3, 0.989, 0.99, 0.995, 1.0):
            with self.subTest(source=source):
                model = CaptureModel()
                project, trace, p = inputs(source=source, receiver=-0.2)
                model.run(project, trace, p)
                k = model.call
                w2, w3 = k["z_source"] / k["rfz"], k["z_source"] / k["rtra"]
                self.assertAlmostEqual(w2, 0.99 * p["widthf"])
                self.assertAlmostEqual(w3, 0.01 * p["widthf"])
                self.assertAlmostEqual(w2 + w3, p["widthf"])
                for prefix in ("rc", "rou", "q"):
                    self.assertEqual(k[prefix + "2"], k[prefix + "3"])
                reflected = source >= 0.99
                self.assertEqual(k["delv"], -model.config.delv if reflected else model.config.delv)
                self.assertAlmostEqual(k["xs"] * w2,
                                       (1.0 - source if reflected else source) * p["widthf"])
                self.assertAlmostEqual(k["xr"] * w2,
                                       (1.2 if reflected else -0.2) * p["widthf"])
                self.assertLessEqual(k["xs"], 1.0)
                self.assertAlmostEqual(k["rc4"], p["v1"] if reflected else p["v1"] * p["v3r"])
                self.assertEqual(k["rou4"], p["rou1"] if reflected else p["rou3"])
                self.assertEqual(k["q4"], p["q1"] if reflected else p["q3"])

    def test_invalid_geometry(self):
        for override in ({"widthf": 0.0}, {"widthf": -1.0},
                         {"widthf": np.nan}, {"x": -0.01}, {"x": 1.01}, {"x": np.nan}):
            project, trace, p = inputs()
            p.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                CaptureModel().run(project, trace, p)

    def test_gap_uses_total_width(self):
        for source in (0.3, 0.995):
            with self.subTest(source=source), self.assertRaises(ValueError):
                CaptureModel().run(*inputs(source=source, receiver=source + 0.00499))


class CompiledWaveformTests(unittest.TestCase):
    def test_equivalence_to_previous_three_medium_mapping(self):
        try:
            model = HS4PDISForwardModel()
        except ImportError as exc:
            self.skipTest(str(exc))
        worst = 0.0
        count = 0
        for source in (0.0, 0.3, 0.98, 0.99, 0.995, 1.0):
            for receiver in (-0.2, 0.0, 0.5, 0.99, 0.995, 1.0, 1.2):
                if abs(source - receiver) <= 0.005 + 1e-12:
                    continue
                with self.subTest(source=source, receiver=receiver):
                    project, trace, p = inputs(source, receiver)
                    actual = model.run(project, trace, p)
                    # Independent baseline: the original mapping with medium 3
                    # identical to the right host and W2 equal to the total W.
                    k = dict(delv=model.config.delv, lt=trace.header.lt,
                             ifm=int(trace.header.maxfreq), gh=trace.header.gibbs,
                             rc1=p["v1"], rc2=p["v1"] * p["vfr"],
                             rc3=p["v1"] * p["v3r"], rc4=p["v1"] * p["v3r"],
                             rou1=p["rou1"], rou2=p["rou2"], rou3=p["rou3"], rou4=p["rou3"],
                             q1=p["q1"], q2=p["qf"], q3=p["q3"], q4=p["q3"],
                             z_rec=0.0, z_source=p["z"], rfz=p["z"] / p["widthf"],
                             rtra=p["z"] / (0.01 * p["widthf"]), xr=receiver, xs=source,
                             acc=trace.header.accuracy,
                             repmac=np.pi * trace.header.lt * p["v1"] / p["z"],
                             time=np.zeros(4096), disp=np.zeros(4096))
                    model._call_backend(**k)
                    self.assertTrue(np.all(np.isfinite(actual.disp)))
                    error = np.linalg.norm(actual.disp - k["disp"]) / np.linalg.norm(k["disp"])
                    worst = max(worst, error)
                    count += 1
                    self.assertLess(error, 1e-6)
                    np.testing.assert_array_equal(actual.time, k["time"])
        print(f"Compiled waveform comparisons: {count}; max relative L2 error: {worst:.6g}")


if __name__ == "__main__":
    unittest.main()
