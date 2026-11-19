"""Create a Ridgecrest GA summary PDF from a saved inversion result.

The figure contains the best resolved waveform model and all individuals from
the last 10 saved generations. Waveforms use the saved event-wide start by
default; --alignment per-station finds a separate cross-correlation shift for
each station. Synthetic traces are scaled to their observed station peaks.
Parameter axes include every plotted point.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# All project paths are defined here. Keep them relative to the repository root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyFZTW.forward.hs4pdis import HS4PDISForwardModel
from PyFZTW.io.project import load_project
from PyFZTW.inverse.objective import align_best_time_shift

ref_evid = "38690767"

STATION_NAMES = ["B416", "B417", "B418", "B419", "B420", "B421", "B422", "B423"]
DEFAULT_INPUT_DIR = REPO_ROOT / "examples" / "ridgecrest_test"
RESULT_PATH = (
    REPO_ROOT
    / "examples"
    / "ridgecrest_test"
    / "ga"
    / "ridgecrest_38690767_deap_run022.pkl"
)
FALLBACK_RESULT_PATH = RESULT_PATH.with_name("inversion_results_v1_run022.pkl")
LAST_GENERATIONS = 10
OUTPUT_FIGURE = REPO_ROOT / "ridgecrest_run022.pdf"
AUTO_XLIM_PADDING_FRACTION = 0.02
LABEL_SIZE = 14
TICK_SIZE = 12
PANEL_SIZE = 14
PARAMETER_LABELS = {
    "v1": r"$\mathbfit{v}_{\mathbf{1}}\ (\mathbf{km/s})$",
    "vfr": r"$\mathbfit{v}_{\mathbf{fr}}\ (\mathbf{\%})$",
    "v3r": r"$\mathbfit{v}_{\mathbf{3r}}\ (\mathbf{\%})$",
    "qf": r"$\mathbfit{Q}_{\mathbfit{f}}$",
    "widthf": r"$\mathbfit{W}_{\mathbfit{f}}\ (\mathbf{km})$",
    ref_evid + "_z": r"$\mathbf{z}_{\mathbfit{s}}\ (\mathbf{km})$",
}


def _unwrap_result(loaded):
    if isinstance(loaded, dict) and "result" in loaded:
        return loaded["result"]
    return loaded


def _result_field(result, name, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def load_result_file(result_file):
    with Path(result_file).open("rb") as stream:
        result = _unwrap_result(pickle.load(stream))
    best_params = _result_field(result, "best_resolved", {}) or {}
    best_score = float(_result_field(result, "best_score", np.nan))
    if not best_params or not np.isfinite(best_score):
        raise ValueError(f"Missing best result in {result_file}")
    history_data = extract_history_data(result, last_generations=LAST_GENERATIONS)
    if history_data.empty:
        raise ValueError(f"No history rows in {result_file}")
    return result, history_data


def saved_event_start(result):
    starts = _result_field(result, "best_event_window_start_s") or {}
    if ref_evid not in starts:
        raise ValueError(
            f"Missing best_event_window_start_s['{ref_evid}']; saved alignment is required"
        )
    start_s = float(starts[ref_evid])
    if not np.isfinite(start_s):
        raise ValueError("Saved event window start must be finite")
    return start_s


def align_saved_event(observations, synthetics, start_s):
    """Sample every synthetic trace at the saved event-wide window start.

    No correlation search is performed. The returned times remain on each
    observation's time axis; start_s is an absolute synthetic time, not a lag.
    """
    if not observations or len(observations) != len(synthetics):
        raise ValueError("Need matching non-empty observed and synthetic traces")
    windows = [gram.windowed_data() for gram in observations]
    if any(len(time) < 2 for time, _ in windows):
        raise ValueError("Observed windows must contain at least two samples")
    traces = []
    for gram, (time, data), syn in zip(observations, windows, synthetics):
        if not all(np.all(np.isfinite(a)) for a in (time, data, syn.time, syn.disp)):
            raise ValueError(f"Non-finite waveform data at {gram.station}")
        if np.any(np.diff(time) <= 0) or np.any(np.diff(syn.time) <= 0):
            raise ValueError(f"Waveform time axis must increase at {gram.station}")
        synthetic_time = start_s + (time - time[0])
        if synthetic_time[0] < syn.time[0] or synthetic_time[-1] > syn.time[-1]:
            raise ValueError(f"Saved synthetic window does not cover {gram.station}")
        aligned_syn = np.interp(synthetic_time, syn.time, syn.disp)
        traces.append((gram.station, time, data, aligned_syn, float(start_s)))
    return traces


def align_per_station(observations, synthetics):
    """Find each station's best synthetic window by normalized correlation."""
    if not observations or len(observations) != len(synthetics):
        raise ValueError("Need matching non-empty observed and synthetic traces")
    traces = []
    for gram, syn in zip(observations, synthetics):
        time, data = gram.windowed_data()
        if not all(np.all(np.isfinite(a)) for a in (time, data, syn.time, syn.disp)):
            raise ValueError(f"Non-finite waveform data at {gram.station}")
        shift, _, observed, aligned_syn = align_best_time_shift(syn, time, data)
        dt = float(np.median(np.diff(time)))
        obs_time = time[0] + np.arange(len(observed), dtype=float) * dt
        traces.append((gram.station, obs_time, observed, aligned_syn, shift))
    return traces


def observed_event_peak(traces):
    """Global observed peak over the aligned event fit windows."""
    observed_peak = max(float(np.max(np.abs(trace[2]))) for trace in traces)
    if not np.isfinite(observed_peak) or observed_peak <= 0:
        raise ValueError("Observed event wavefield must have a finite, nonzero peak")
    return observed_peak


def observation_files(input_dir):
    """Find a complete set of reviewer inputs or Step 3 outputs."""
    reviewer_files = [
        input_dir / "Disp" / f"{station}_disp_ref_{ref_evid}.txt"
        for station in STATION_NAMES
    ]
    if all(path.is_file() for path in reviewer_files):
        return reviewer_files
    if any(path.is_file() for path in reviewer_files):
        missing = [path.name for path in reviewer_files if not path.is_file()]
        raise FileNotFoundError(f"Incomplete Ridgecrest inputs in {input_dir / 'Disp'}: {', '.join(missing)}")

    step3_files = [
        input_dir / "DISP" / f"{station}_ref_{ref_evid}.disp"
        for station in STATION_NAMES
    ]
    if all(path.is_file() for path in step3_files):
        return step3_files
    raise FileNotFoundError(
        f"Expected all eight Ridgecrest waveforms in {input_dir / 'Disp'} "
        f"(reviewer .txt files) or {input_dir / 'DISP'} (Step 3 .disp files)"
    )


def prepare_waveforms(result, input_dir=None, alignment="saved"):
    input_dir = Path(input_dir) if input_dir is not None else DEFAULT_INPUT_DIR
    project = load_project(
        input_dir / "crust.txt", input_dir / "eqs.txt",
        input_dir / "stations.txt", observation_files(input_dir),
    )
    if {gram.event_id for gram in project.seismograms} != {ref_evid}:
        raise ValueError(f"Expected only reference event {ref_evid}")
    best = dict(_result_field(result, "best_resolved"))
    best.update({axis: best[f"{ref_evid}_{axis}"] for axis in ("x", "y", "z")})
    forward = HS4PDISForwardModel()
    synthetics = [forward.run(project, gram, best) for gram in project.seismograms]
    if alignment == "saved":
        start_s = saved_event_start(result)
        traces = align_saved_event(project.seismograms, synthetics, start_s)
        print(f"Saved event window start: {start_s:.9f} s (all stations)")
    elif alignment == "per-station":
        traces = align_per_station(project.seismograms, synthetics)
        print("Per-station synthetic window starts from cross-correlation (s):")
        shifts = {name: shift for name, _, _, _, shift in traces}
        for name in STATION_NAMES:
            print(f"  {name}: {shifts[name]:.9f}")
    else:
        raise ValueError(f"Unknown alignment mode: {alignment}")
    return traces


def _find_resolved_key(keys, var_name):
    if var_name in keys:
        return var_name
    for k in keys:
        if k.endswith(f'_{var_name}'):
            return k
    return None


def auto_xlim(values, best_value=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    best_is_valid = best_value is not None and np.isfinite(best_value)

    if len(values) == 0:
        if not best_is_valid:
            return None
        pad = max(abs(float(best_value)) * AUTO_XLIM_PADDING_FRACTION, 1e-6)
        return float(best_value) - pad, float(best_value) + pad

    lo, hi = float(np.min(values)), float(np.max(values))
    if best_is_valid:
        lo = min(lo, float(best_value))
        hi = max(hi, float(best_value))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        pad = max(abs(float(lo)) * AUTO_XLIM_PADDING_FRACTION, 1e-6)
    else:
        pad = (hi - lo) * AUTO_XLIM_PADDING_FRACTION
    return float(lo - pad), float(hi + pad)


def extract_history_data(result, last_generations=LAST_GENERATIONS):
    history = _result_field(result, "history", []) or []
    if not history:
        return pd.DataFrame()

    generation_numbers = sorted(
        {
            gen_data['generation'] if isinstance(gen_data, dict) else gen_data.generation
            for gen_data in history
        }
    )
    selected_generations = set(generation_numbers[-last_generations:])
    data_list = []
    for gen_data in history:
        gen = gen_data['generation'] if isinstance(gen_data, dict) else gen_data.generation
        if gen not in selected_generations:
            continue
        individuals = gen_data['individuals'] if isinstance(gen_data, dict) else gen_data.individuals
        for ind_id, ind in enumerate(individuals):
            score = ind['score'] if isinstance(ind, dict) else ind.score
            resolved = ind['resolved'] if isinstance(ind, dict) else getattr(ind, 'resolved', None)
            row = {'gen': gen, 'ind_id': ind_id, 'score': float(score)}
            if resolved:
                for k, v in resolved.items():
                    row[k] = v
            data_list.append(row)
    return pd.DataFrame(data_list)


def plot_single_result(best_params, best_score_val, df, traces, output_figure,
                       figure_note=None, reference_values=None):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    })
    observed_by_station = {name: (time, obs) for name, time, obs, syn, start in traces}
    synthetic_by_station = {name: (time, syn) for name, time, obs, syn, start in traces}

    fig = plt.figure(figsize=(12, 14))
    gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1.0], hspace=0.12)

    ax_wave = fig.add_subplot(gs_main[0, 0])
    global_xmin, global_xmax = np.inf, -np.inf
    global_obs_max = observed_event_peak(traces)
    for idx, name in enumerate(STATION_NAMES):
        obs_time, obs_data = observed_by_station[name]
        syn_time, syn = synthetic_by_station[name]
        obs_norm = obs_data / global_obs_max
        obs_factor = np.max(np.abs(obs_data)) / global_obs_max
        syn_peak = np.max(np.abs(syn))
        if not np.isfinite(syn_peak) or syn_peak <= 0:
            raise ValueError(f"Synthetic trace at {name} must have a finite, nonzero peak")
        syn_norm = syn / syn_peak * obs_factor
        ax_wave.plot(
            obs_time,
            obs_norm + idx * 0.75,
            'k-',
            lw=1.0,
            zorder=3,
            label='Observed' if idx == 0 else '',
        )
        ax_wave.plot(
            syn_time,
            syn_norm + idx * 0.75,
            color='red',
            lw=1.0,
            zorder=4,
            label='Synthetic' if idx == 0 else '',
        )
        global_xmin = min(global_xmin, obs_time.min())
        global_xmax = max(global_xmax, obs_time.max())

    ax_wave.set_xlabel('Time (s)', labelpad=2, fontweight='bold')
    ax_wave.set_ylabel('Station', labelpad=2, fontweight='bold')
    ax_wave.set_yticks([i * 0.75 for i in range(len(STATION_NAMES))])
    ax_wave.set_yticklabels(STATION_NAMES)
    ax_wave.set_xlim(global_xmin, global_xmax)
    ax_wave.tick_params(labelsize=TICK_SIZE, direction='out')
    ax_wave.grid(axis='x', linestyle=':', alpha=0.4)
    ax_wave.text(-0.025, 1.005, '(a)', transform=ax_wave.transAxes, fontsize=PANEL_SIZE, fontweight='bold', va='bottom', ha='left')

    handles, labels = ax_wave.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_wave.legend(unique.values(), unique.keys(), loc='upper right',
                   fontsize=LABEL_SIZE, frameon=True, framealpha=0.35,
                   facecolor='white', edgecolor='none', borderpad=0.3,
                   labelspacing=0.25, handlelength=1.5)

    plot_vars = ['v1', 'vfr', 'v3r', 'qf', 'widthf', ref_evid + '_z']
    n_vars = len(plot_vars)
    n_cols_bottom = 3
    n_rows_bottom = (n_vars + n_cols_bottom - 1) // n_cols_bottom

    var_units = {'v1': 'km/s', 'vfr': '%', 'v3r': '%', 'qf': '',
                 'widthf': 'km', ref_evid + '_z': 'km'}
    value_decimals = {'v1': 1, 'vfr': 2, 'v3r': 2, 'qf': 0,
                      'widthf': 2, ref_evid + '_z': 1}

    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        n_rows_bottom, n_cols_bottom,
        subplot_spec=gs_main[1, 0],
        wspace=0.35, hspace=0.45
    )

    for i, variable in enumerate(plot_vars):
        row, col = i // n_cols_bottom, i % n_cols_bottom
        ax = fig.add_subplot(gs_bottom[row, col])
        col_name = _find_resolved_key(df.columns, variable)
        if col_name is None:
            ax.set_title(variable)
            continue
        valid = df.dropna(subset=[col_name, 'score'])
        if valid.empty:
            ax.set_title(variable)
            continue

        scale = 100 if variable in ('vfr', 'v3r') else 1
        x_vals = valid[col_name].values * scale

        x_best_val = best_params.get(variable, None)
        x_best_val = x_best_val * scale if x_best_val is not None else None
        x_reference = None if reference_values is None else reference_values.get(variable)
        if x_reference is not None:
            x_reference = float(x_reference) * scale
            if not np.isfinite(x_reference):
                raise ValueError(f"Non-finite reference value for {variable}")

        ax.scatter(
            x_vals,
            valid['score'],
            color='green',
            s=1,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        unit = var_units.get(variable, '')
        unit_label = f' {unit}' if unit else ''
        if x_best_val is not None:
            ax.scatter(x_best_val, best_score_val, color='black', s=25, marker='o', zorder=5)
            label_y = 1.02 if reference_values is not None else 0.98
            label_va = 'bottom' if reference_values is not None else 'top'
            ax.text(0.02, label_y,
                    f'{x_best_val:.{value_decimals[variable]}f}{unit_label}',
                    transform=ax.transAxes,
                    fontsize=LABEL_SIZE, fontweight='bold', va=label_va)
        if col == 0:
            ax.set_ylabel('Fitness', color='green', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='green', labelsize=TICK_SIZE, direction='out')
        else:
            ax.tick_params(axis='y', labelleft=False)

        ax2 = ax.twinx()
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        ax2.set_zorder(2)
        ax2.patch.set_visible(False)
        bins = 50
        counts, bin_edges = np.histogram(x_vals, bins=bins)
        weighted_sum, _ = np.histogram(x_vals, bins=bin_edges, weights=valid['score'])
        total = np.sum(weighted_sum)
        density = weighted_sum / total if total > 0 else weighted_sum
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.plot(bin_centers, density, color='black', lw=1.5, zorder=3)
        if x_reference is not None:
            ax2.scatter(x_reference, best_score_val, transform=ax.transData,
                        s=64, facecolors='none', edgecolors='red', linewidths=1.7,
                        zorder=10)

        if col == n_cols_bottom - 1:
            ax2.set_ylabel('Probability Density', color='black', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black', labelsize=TICK_SIZE, direction='out')
        xlim_values = x_vals if x_reference is None else np.append(x_vals, x_reference)
        xlim = auto_xlim(xlim_values, x_best_val)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(PARAMETER_LABELS[variable], labelpad=2, fontweight='bold')

        if row == 0 and col == 0:
            ax.text(-0.08, 1.02, '(b)', transform=ax.transAxes, fontsize=PANEL_SIZE, fontweight='bold', va='bottom', ha='left')

    plt.subplots_adjust(left=0.07, right=0.95, top=0.97, bottom=0.05)
    if figure_note:
        fig.text(0.07, 0.995, figure_note, ha='left', va='top', fontsize=LABEL_SIZE,
                 color='darkred')
        fig.subplots_adjust(top=0.94)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_figure, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_FIGURE)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory with crust.txt, eqs.txt, stations.txt and waveforms in Disp/ or DISP/")
    parser.add_argument("--alignment", choices=("saved", "per-station"), default="saved",
                        help="Use the saved common start or find a cross-correlation start for each station")
    args = parser.parse_args()
    result_path = args.result
    if result_path == RESULT_PATH and not result_path.exists() and FALLBACK_RESULT_PATH.exists():
        result_path = FALLBACK_RESULT_PATH
        print(f"Requested result is absent; using existing run022 result: {result_path}")
    result, history_data = load_result_file(result_path)
    traces = prepare_waveforms(result, args.input_dir, args.alignment)
    plot_single_result(
        best_params=dict(_result_field(result, "best_resolved")),
        best_score_val=float(_result_field(result, "best_score")),
        df=history_data,
        traces=traces,
        output_figure=args.output,
    )
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
