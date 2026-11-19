"""Objective functions for aligning and scoring observed/synthetic traces.

The scoring path extracts each observed fit window, finds the synthetic time
shift with the highest normalized correlation, and converts that correlation to
a fitness score in the range used by the optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..forward.base import SyntheticTrace
from ..io.seismogram import Seismogram


_NORM_EPS = 1e-12


def _median_dt(time: np.ndarray) -> float:
    if len(time) < 2:
        raise ValueError("time array must contain at least two samples")
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError("time array must be strictly increasing")
    return dt


def _is_uniform_time(
    time: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    dt = _median_dt(time)
    return bool(np.allclose(np.diff(time), dt, rtol=rtol, atol=atol))


def _uniform_grid(start: float, end: float, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_samples = int(round((end - start) / dt)) + 1
    if n_samples < 2:
        raise ValueError("time range must contain at least two samples")
    return start + np.arange(n_samples, dtype=float) * dt


def _resample_to_uniform_grid(
    time: np.ndarray,
    data: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    if _is_uniform_time(time):
        return time, data
    grid = _uniform_grid(float(time[0]), float(time[-1]), dt)
    return grid, np.interp(grid, time, data)


def normalized_correlation(
    observed: np.ndarray,
    synthetic: np.ndarray,
) -> float:
    """Return cosine-style correlation between two equal-length waveforms."""
    observed = np.asarray(observed, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    if observed.ndim != 1 or synthetic.ndim != 1:
        raise ValueError("observed and synthetic must be 1D arrays")
    if len(observed) != len(synthetic):
        raise ValueError("observed and synthetic length mismatch")
    if len(observed) == 0:
        raise ValueError("correlation arrays cannot be empty")

    dot = float(np.dot(observed, synthetic))
    norm_obs = float(np.sqrt(np.dot(observed, observed)))
    norm_syn = float(np.sqrt(np.dot(synthetic, synthetic)))
    return dot / (norm_obs * norm_syn + _NORM_EPS)


def find_best_time_shift(
    synthetic: SyntheticTrace,
    observed_time: np.ndarray,
    observed_disp: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Find the synthetic window that best matches the observed waveform.

    Returns the synthetic-window start time, the best correlation, and the
    aligned synthetic displacement samples.
    """
    observed_time = np.asarray(observed_time, dtype=float)
    observed_disp = np.asarray(observed_disp, dtype=float)
    if observed_time.ndim != 1 or observed_disp.ndim != 1:
        raise ValueError("observed_time and observed_disp must be 1D arrays")
    if len(observed_time) != len(observed_disp):
        raise ValueError("observed_time and observed_disp length mismatch")

    obs_dt = _median_dt(observed_time)
    _, obs_data = _resample_to_uniform_grid(observed_time, observed_disp, obs_dt)
    synth_time = _uniform_grid(float(synthetic.time[0]), float(synthetic.time[-1]), obs_dt)
    synth_data = np.interp(synth_time, synthetic.time, synthetic.disp)

    n_obs = len(obs_data)
    if len(synth_data) < n_obs:
        raise ValueError("synthetic trace must be at least as long as observed window")

    synth_windows = sliding_window_view(synth_data, n_obs)
    dot = synth_windows @ obs_data
    norm_obs = float(np.sqrt(np.dot(obs_data, obs_data)))
    norm_syn = np.sqrt(np.sum(synth_windows * synth_windows, axis=1))
    correlations = dot / (norm_obs * norm_syn + _NORM_EPS)

    best_idx = int(np.argmax(correlations))
    best_corr = float(correlations[best_idx])
    best_syn_data = np.array(synth_windows[best_idx], copy=True)
    syn_shift = float(synth_time[best_idx])
    return syn_shift, best_corr, best_syn_data


def align_best_time_shift(
    synthetic: SyntheticTrace,
    observed_time: np.ndarray,
    observed_disp: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return best shift metadata plus aligned observed and synthetic samples."""
    observed_time = np.asarray(observed_time, dtype=float)
    observed_disp = np.asarray(observed_disp, dtype=float)
    obs_dt = _median_dt(observed_time)
    _, obs_data = _resample_to_uniform_grid(observed_time, observed_disp, obs_dt)
    best_shift, best_corr, best_syn_data = find_best_time_shift(
        synthetic=synthetic,
        observed_time=observed_time,
        observed_disp=observed_disp,
    )
    return best_shift, best_corr, obs_data, best_syn_data


def find_best_event_time_shift(
    observations: Sequence[Seismogram],
    synthetics: Sequence[SyntheticTrace],
) -> tuple[float, float, list[np.ndarray], list[np.ndarray]]:
    """Find one maximum-absolute-correlation shift shared by an event's traces.

    The event correlation is computed as the normalized correlation of all
    aligned station windows concatenated together. FFTs provide each station's
    linear cross-correlation curve; sliding synthetic-window energies are
    calculated with cumulative sums so the normalization is exact at each shift.
    """
    if not observations or len(observations) != len(synthetics):
        raise ValueError("observations and synthetics must be non-empty and equal in length")

    sample_intervals: list[float] = []
    for observed in observations:
        obs_time, _ = observed.windowed_data()
        sample_intervals.append(_median_dt(obs_time))

    # All event traces should share one sampling grid for a common physical
    # shift. If their rates differ slightly, use the finest observed interval
    # and interpolate every trace onto that interval.
    dt = min(sample_intervals)
    resampled_observed: list[np.ndarray] = []
    observed_energy = 0.0
    for observed in observations:
        obs_time, obs_disp = observed.windowed_data()
        n_samples = int(np.floor((obs_time[-1] - obs_time[0]) / dt + 1e-8)) + 1
        obs_grid = obs_time[0] + np.arange(n_samples, dtype=float) * dt
        obs_data = np.interp(obs_grid, obs_time, obs_disp)
        resampled_observed.append(obs_data)
        observed_energy += float(np.dot(obs_data, obs_data))

    common_start = max(float(syn.time[0]) for syn in synthetics)
    common_end = min(float(syn.time[-1]) for syn in synthetics)
    n_synthetic = int(np.floor((common_end - common_start) / dt + 1e-8)) + 1
    if n_synthetic < 2:
        raise ValueError("synthetic traces have no common time range")
    common_time = common_start + np.arange(n_synthetic, dtype=float) * dt
    resampled_synthetics = [
        np.interp(common_time, syn.time, syn.disp) for syn in synthetics
    ]

    n_shifts = min(
        len(syn_data) - len(obs_data) + 1
        for syn_data, obs_data in zip(resampled_synthetics, resampled_observed)
    )
    if n_shifts <= 0:
        raise ValueError("synthetic traces must cover every observed fit window")

    summed_numerator = np.zeros(n_shifts, dtype=float)
    summed_synthetic_energy = np.zeros(n_shifts, dtype=float)
    for syn_data, obs_data in zip(resampled_synthetics, resampled_observed):
        n_full = len(syn_data) + len(obs_data) - 1
        n_fft = 1 << (n_full - 1).bit_length()
        cross_spectrum = (
            np.fft.rfft(syn_data, n_fft)
            * np.fft.rfft(obs_data[::-1], n_fft)
        )
        convolution = np.fft.irfft(cross_spectrum, n_fft)
        numerator = convolution[len(obs_data) - 1 : len(obs_data) - 1 + n_shifts]
        squared_prefix = np.concatenate(([0.0], np.cumsum(syn_data * syn_data)))
        energy = squared_prefix[len(obs_data) : len(obs_data) + n_shifts] - squared_prefix[:n_shifts]
        summed_numerator += numerator
        summed_synthetic_energy += energy

    denominator = np.sqrt(observed_energy * summed_synthetic_energy)
    correlations = summed_numerator / (denominator + _NORM_EPS)
    best_idx = int(np.argmax(np.abs(correlations)))
    # The absolute value selects the shift only; preserve the signed coefficient
    # so an opposite-polarity match receives low fitness under (1 + C*) / 2.
    best_corr = float(correlations[best_idx])
    best_shift = float(common_time[best_idx])
    aligned_observed = [data.copy() for data in resampled_observed]
    aligned_synthetic = [
        data[best_idx : best_idx + len(obs_data)].copy()
        for data, obs_data in zip(resampled_synthetics, resampled_observed)
    ]
    return best_shift, best_corr, aligned_observed, aligned_synthetic


@dataclass
class WindowedCorrelationFitness:
    """Fitness evaluator for aligned observed and synthetic seismograms.

    The score is ``0.5 * (1 + correlation)`` so larger values are better. Helper
    methods expose either a trace-level shift or one shared event-level shift.
    """
    last_best_shift: float | None = None

    def evaluate_with_aligned_data(
        self,
        observed: Seismogram,
        synthetic: SyntheticTrace,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        obs_t, obs_y = observed.windowed_data()
        best_shift, best_corr, obs_data, best_syn_data = align_best_time_shift(
            synthetic=synthetic,
            observed_time=obs_t,
            observed_disp=obs_y,
        )
        shift = float(best_shift)
        self.last_best_shift = shift
        score = 0.5 * (1.0 + best_corr)
        return float(score), shift, obs_data, best_syn_data

    def evaluate_event_with_aligned_data(
        self,
        observations: Sequence[Seismogram],
        synthetics: Sequence[SyntheticTrace],
    ) -> tuple[float, float, list[np.ndarray], list[np.ndarray]]:
        """Score all traces from one event using one shared time shift."""
        best_shift, best_corr, obs_data, syn_data = find_best_event_time_shift(
            observations=observations,
            synthetics=synthetics,
        )
        self.last_best_shift = float(best_shift)
        score = 0.5 * (1.0 + best_corr)
        return float(score), float(best_shift), obs_data, syn_data

    def evaluate_with_shift(
        self,
        observed: Seismogram,
        synthetic: SyntheticTrace,
    ) -> tuple[float, float]:
        score, shift, _, _ = self.evaluate_with_aligned_data(
            observed=observed,
            synthetic=synthetic,
        )
        return score, shift

    def evaluate(
        self,
        observed: Seismogram,
        synthetic: SyntheticTrace,
    ) -> float:
        """Return only the optimizer-facing scalar fitness score."""
        score, _ = self.evaluate_with_shift(
            observed=observed,
            synthetic=synthetic,
        )
        return score
