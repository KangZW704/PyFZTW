"""Draw GA/PSO results side by side directly from saved inversion pickles.

GA uses all candidates from its final 20 saved generations; PSO uses particle
personal bests. Waveforms use the saved event window start and current project
inputs; reproduced fitness must agree with the saved value. Black parameter
curves are normalized fitness sums per bin, not posterior densities.
No source PDFs or other plotting scripts are required.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from PyFZTW.forward.hs4pdis import HS4PDISForwardModel
from PyFZTW.io.project import load_project
from PyFZTW.inverse.objective import normalized_correlation

SYN_DIR = ROOT / "examples" / "syn_test"
GA_RESULT = SYN_DIR / "ga" / "inversion_results_v1_run086.pkl"
PSO_RESULT = SYN_DIR / "pso" / "inversion_results_v1_run04.pkl"
OUTPUT = ROOT / "synthetic_ga_pso.pdf"
WIDTH, HEIGHT = 468, 342  # pt; 6.5 x 4.75 inches
TICK_SIZE = LABEL_SIZE = 8
PANEL_SIZE = 12
SCATTER_DPI = 300
FINAL_GENERATIONS = 20
HISTOGRAM_BINS = 50
BEST_VALUE_DECIMALS = {"v1": 1, "vfr": 1, "v3r": 1, "qf": 0, "widthf": 2, "1_z": 1}
# Expressions and parameter order are defined here, with no label helper.
PARAMETERS = (
    ("v1", r"$v_1\ (\mathrm{km/s})$", 1),
    ("vfr", r"$v_{\mathrm{fr}}\ (\%)$", 100),
    ("v3r", r"$v_{\mathrm{3r}}\ (\%)$", 100),
    ("qf", r"$Q_f$", 1),
    ("widthf", r"$W_f\ (\mathrm{km})$", 1),
    ("1_z", r"$\mathrm{z}_s\ (\mathrm{km})$", 1),
)


def get_field(result, name, default=None):
    return result.get(name, default) if isinstance(result, dict) else getattr(result, name, default)


def load_result(path):
    with Path(path).open("rb") as handle:
        loaded = pickle.load(handle)
    result = loaded.get("result", loaded) if isinstance(loaded, dict) else loaded
    if not get_field(result, "best_resolved"):
        raise ValueError(f"No best_resolved model in {path}")
    if not np.isfinite(get_field(result, "best_score", np.nan)):
        raise ValueError(f"No finite best_score in {path}")
    return result


def extract_final_ga_data(result, n_generations=FINAL_GENERATIONS):
    """Select the final consecutive generations actually recorded in the file."""
    history = get_field(result, "history", []) or []
    generations = sorted({int(get_field(item, "generation")) for item in history})
    if len(generations) < n_generations:
        raise ValueError(f"GA result has only {len(generations)} saved generations; need {n_generations}")
    selected = generations[-n_generations:]
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("Final GA generations are not consecutive in the saved history")
    df = extract_history_data(result, generation_range=(selected[0], selected[-1]))
    if df.empty or set(df["gen"]) != set(selected):
        raise ValueError("Missing candidate models in the final GA generations")
    return df, (selected[0], selected[-1])


def _find_resolved_key(keys, variable):
    if variable in keys:
        return variable
    return next((key for key in keys if key.endswith(f"_{variable}")), None)


def extract_history_data(result, generation_range=None):
    rows = []
    for item in get_field(result, "history", []) or []:
        generation = get_field(item, "generation")
        if generation_range and not generation_range[0] <= generation <= generation_range[1]:
            continue
        for individual in get_field(item, "individuals", []):
            resolved = get_field(individual, "resolved") or {}
            if resolved:
                rows.append({**resolved, "gen": generation,
                             "score": float(get_field(individual, "score"))})
    return pd.DataFrame(rows)


def extract_pbest_data(result):
    metadata = result if isinstance(result, dict) else (get_field(result, "metadata") or {})
    rows = []
    for item in metadata.get("pso", {}).get("pbest_population", []):
        if item.get("pbest_resolved") and item.get("pbest_score") is not None:
            rows.append({**item["pbest_resolved"], "score": float(item["pbest_score"])})
    return pd.DataFrame(rows)


def align_saved_event(observations, synthetics, start_s):
    """Reproduce objective.py's grids at a saved synthetic-window start.

    No correlation search is performed. The returned times remain on each
    observation's time axis; start_s is an absolute synthetic time, not a lag.
    """
    if not observations or len(observations) != len(synthetics):
        raise ValueError("Need matching non-empty observed and synthetic traces")
    windows = [gram.windowed_data() for gram in observations]
    if any(len(time) < 2 for time, _ in windows):
        raise ValueError("Observed windows must contain at least two samples")
    dt = min(float(np.median(np.diff(time))) for time, _ in windows)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(start_s):
        raise ValueError("Invalid sampling interval or saved event window start")
    common_start = max(float(syn.time[0]) for syn in synthetics)
    common_end = min(float(syn.time[-1]) for syn in synthetics)
    n_synthetic = int(np.floor((common_end - common_start) / dt + 1e-8)) + 1
    index = int(round((start_s - common_start) / dt))
    if n_synthetic < 2 or index < 0 or index >= n_synthetic:
        raise ValueError("Saved event window start is outside the synthetic grid")
    common_time = common_start + np.arange(n_synthetic, dtype=float) * dt
    if not np.isclose(common_time[index], start_s, rtol=0, atol=dt * 1e-6):
        raise ValueError("Saved event window start does not lie on the objective sampling grid")
    traces = []
    for gram, (time, data), syn in zip(observations, windows, synthetics):
        if not all(np.all(np.isfinite(a)) for a in (time, data, syn.time, syn.disp)):
            raise ValueError(f"Non-finite waveform data at {gram.station}")
        n_samples = int(np.floor((time[-1] - time[0]) / dt + 1e-8)) + 1
        if index + n_samples > n_synthetic:
            raise ValueError(f"Saved synthetic window does not cover {gram.station}")
        obs_time = time[0] + np.arange(n_samples, dtype=float) * dt
        obs = np.interp(obs_time, time, data)
        resampled_syn = np.interp(common_time, syn.time, syn.disp)
        aligned_syn = resampled_syn[index:index + n_samples].copy()
        traces.append((gram.station, obs_time, obs, aligned_syn, float(start_s)))
    return traces


def event_peak_scales(traces):
    """Separate positive global peaks over the aligned event fit windows."""
    observed_peak = max(float(np.max(np.abs(trace[2]))) for trace in traces)
    synthetic_peak = max(float(np.max(np.abs(trace[3]))) for trace in traces)
    if not np.isfinite(observed_peak + synthetic_peak) or min(observed_peak, synthetic_peak) <= 0:
        raise ValueError("Event wavefields must have finite, nonzero global peaks")
    return observed_peak, synthetic_peak


def prepare_waveforms(result, syn_dir=SYN_DIR):
    project = load_project(
        syn_dir / "syn_crust.txt", syn_dir / "syn_eq.txt", syn_dir / "syn_stations.txt",
        [syn_dir / "data" / f"{station}_new.disp" for station in ("S01", "S02", "S03")],
    )
    event_ids = {gram.event_id for gram in project.seismograms}
    if event_ids != {"1"}:
        raise ValueError("This synthetic figure expects the single reference event '1'")
    starts = get_field(result, "best_event_window_start_s") or {}
    if "1" not in starts:
        raise ValueError("Missing best_event_window_start_s['1']; saved alignment is required")
    best = dict(get_field(result, "best_resolved"))
    best.update({axis: best[f"1_{axis}"] for axis in ("x", "y", "z")})
    forward = HS4PDISForwardModel()
    synthetics = [forward.run(project, gram, best) for gram in project.seismograms]
    traces = align_saved_event(project.seismograms, synthetics, float(starts["1"]))
    observed = np.concatenate([trace[2] for trace in traces])
    synthetic = np.concatenate([trace[3] for trace in traces])
    score = 0.5 * (1 + normalized_correlation(observed, synthetic))
    saved_score = float(get_field(result, "best_score"))
    difference = score - saved_score
    print(f"Saved fitness={saved_score:.12f}; reproduced={score:.12f}; delta={difference:+.3e}")
    if not np.isfinite(saved_score) or not np.isclose(score, saved_score, rtol=0, atol=1e-8):
        raise ValueError("Reproduced fitness differs from pickle; check input data and forward configuration")
    event_peak_scales(traces)
    return traces


def add_axes(figure, bounds):
    """Keep the existing layout coordinates in PDF points."""
    x, y, width, height = bounds
    return figure.add_axes((x / WIDTH, y / HEIGHT, width / WIDTH, height / HEIGHT))


def draw_waveforms(axis, traces):
    observed_peak, synthetic_peak = event_peak_scales(traces)
    for index, (station, time, observed, synthetic, _) in enumerate(traces):
        axis.plot(time, observed / observed_peak + index * .75, color="black", lw=.7)
        axis.plot(time, synthetic / synthetic_peak + index * .75,
                  color="red", ls="--", lw=.7)
    axis.set_yticks(np.arange(len(traces)) * .75, [trace[0] for trace in traces])
    axis.set_xlim(min(trace[1][0] for trace in traces), max(trace[1][-1] for trace in traces))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    axis.set_xlabel("Time (s)", labelpad=2)
    axis.set_ylabel("Station", labelpad=2)


def draw_parameters(figure, xbase, result, samples, algorithm, vector_scatter=False):
    best = get_field(result, "best_resolved")
    best_score = float(get_field(result, "best_score"))
    prepared = []
    for key, label, scale in PARAMETERS:
        column = _find_resolved_key(samples.columns, key)
        if column is None:
            raise ValueError(f"Missing samples for {key}")
        values = samples[column].to_numpy(dtype=float) * scale
        scores = samples["score"].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(scores)):
            raise ValueError(f"Non-finite samples for {key}")
        if np.any(scores < 0) or scores.sum() <= 0:
            raise ValueError(f"Invalid fitness weights for {key}")
        prepared.append((values, scores, float(best[key]) * scale))
    minimum = min(float(scores.min()) for _, scores, _ in prepared)
    maximum = max(best_score, max(float(scores.max()) for _, scores, _ in prepared))
    padding = max(.04 * (maximum - minimum), .0002)
    ylim = minimum - padding, maximum + padding
    if algorithm == "PSO":
        ylim = min(ylim[0], .979), max(ylim[1], 1.001)
    for index, ((key, label, _), (values, scores, best_value)) in enumerate(zip(PARAMETERS, prepared)):
        row, col = divmod(index, 3)
        axis = add_axes(figure, (xbase + 61 * col, 123 - 91 * row, 42, 44))
        density_axis = axis.twinx()
        # Keep dots above the black distribution without hiding its axes.
        density_axis.set_zorder(1)
        axis.set_zorder(2)
        axis.patch.set_visible(False)
        axis.scatter(values, scores, s=.7, color="green", alpha=.5,
                     linewidths=0, rasterized=not vector_scatter)
        axis.scatter([best_value], [best_score], s=4, color="black", linewidths=0, zorder=4)
        weighted, edges = np.histogram(values, bins=HISTOGRAM_BINS, weights=scores)
        fractions = weighted / weighted.sum()
        density_axis.plot((edges[:-1] + edges[1:]) / 2, fractions, color="black", lw=.7)
        density_axis.set_ylim(bottom=0)
        density_axis.yaxis.set_major_locator(MaxNLocator(nbins=2))
        density_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
        lo, hi = min(float(values.min()), best_value), max(float(values.max()), best_value)
        xpad = max((hi - lo) * .08, abs(best_value) * .001, 1e-6)
        axis.set_xlim(lo - xpad, hi + xpad)
        ticks = MaxNLocator(nbins=2, min_n_ticks=2, steps=[1, 2, 2.5, 5, 10]).tick_values(lo-xpad, hi+xpad)
        ticks = ticks[(ticks >= lo-xpad) & (ticks <= hi+xpad)]
        axis.set_xticks(ticks if len(ticks) <= 2 else ticks[[0, -1]])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}".rstrip("0").rstrip(".")))
        axis.set_ylim(*ylim)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=2))
        if algorithm == "PSO" and ylim == (.979, 1.001):
            axis.set_yticks([.98, .99, 1.0])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4f}".rstrip("0").rstrip(".")))
        axis.tick_params(axis="y", labelleft=col == 0)
        axis.set_xlabel(label, labelpad=2, fontweight="semibold")
        axis.text(.5, 1.08, f"{best_value:.{BEST_VALUE_DECIMALS[key]}f}", transform=axis.transAxes,
                  ha="center", va="bottom", fontsize=LABEL_SIZE)
        print(f"{key}: best={best_value:.6g}; samples={len(values)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-result", type=Path, default=GA_RESULT)
    parser.add_argument("--pso-result", type=Path, default=PSO_RESULT)
    parser.add_argument("--syn-dir", type=Path, default=SYN_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--final-generations", type=int, default=FINAL_GENERATIONS)
    parser.add_argument("--vector-scatter", action="store_true")
    args = parser.parse_args()
    if args.final_generations <= 0:
        parser.error("--final-generations must be positive")
    ga, pso = load_result(args.ga_result), load_result(args.pso_result)
    ga_samples, generations = extract_final_ga_data(ga, args.final_generations)
    pso_samples = extract_pbest_data(pso)
    if pso_samples.empty:
        raise ValueError("PSO result contains no particle personal-best models")
    # Validate waveforms before creating or overwriting the output.
    ga_traces = prepare_waveforms(ga, args.syn_dir)
    pso_traces = prepare_waveforms(pso, args.syn_dir)
    print(f"GA generations {generations[0]}-{generations[1]}: {len(ga_samples)} candidates")
    print(f"PSO: {len(pso_samples)} personal-best models")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "font.size": LABEL_SIZE, "axes.labelsize": LABEL_SIZE, "axes.linewidth": .55,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "xtick.major.size": 2.3, "ytick.major.size": 2.3,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
    })
    figure = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
    wave_axes = []
    for col, (algorithm, result, samples, traces) in enumerate([
        ("GA", ga, ga_samples, ga_traces), ("PSO", pso, pso_samples, pso_traces),
    ]):
        xbase = 39 + 234 * col
        axis = add_axes(figure, (xbase, 210, 172, 86))
        draw_waveforms(axis, traces)
        wave_axes.append(axis)
        draw_parameters(figure, xbase, result, samples, algorithm, args.vector_scatter)
        figure.text((xbase+86)/WIDTH, 327/HEIGHT, algorithm,
                    ha="center", fontsize=LABEL_SIZE, fontweight="bold")
        for y, panel in [(300, "a" if col == 0 else "b"), (178, "c" if col == 0 else "d")]:
            figure.text((xbase-15)/WIDTH, y/HEIGHT, f"({panel})",
                        fontsize=PANEL_SIZE, fontweight="bold", va="bottom")
    limits = min(ax.get_ylim()[0] for ax in wave_axes), max(ax.get_ylim()[1] for ax in wave_axes)
    for axis in wave_axes:
        axis.set_ylim(*limits)
    figure.text(16/WIDTH, 97/HEIGHT, "Fitness", rotation=90, ha="center", va="center")
    # Preserve the existing title; values are normalized fitness sums per bin.
    figure.text(458/WIDTH, 97/HEIGHT, "Probability density", rotation=90, ha="center", va="center")
    figure.legend(handles=[Line2D([], [], color="black", lw=.7, label="Input"),
                           Line2D([], [], color="red", ls="--", lw=.7, label="Best-fitting model")],
                  loc="upper center", bbox_to_anchor=(.5, .985), ncol=2,
                  frameon=False, borderaxespad=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(args.output, dpi=SCATTER_DPI)
    finally:
        plt.close(figure)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
