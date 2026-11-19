"""Detect FZTW candidates from Step 1 fault-parallel waveform pickles.

The workflow reads ``*_fpara.pkl`` files from ``step1_load_sac.py`` and processes
each event in three stages:

1. Preprocess the fault-parallel waveforms, including integration, bandpass
   filtering, and the 1 / sqrt(t) convolution correction.
2. Measure station-wise S-wave amplitude features within the picked S-wave
   analysis window.
3. Correlate each event's amplitude pattern with the median amplitude pattern
   across all valid events to detect FZTW candidates.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal


AMP_DISTRIBUTION_FILE = "Amp_distribution.npz"
FZTW_THRESHOLD = 0.2
MEDIAN_FILTER_KERNEL = 3
REQUIRED_PICKLE_KEYS = {
    "data_fpara",
    "stalst",
    "arrdist",
    "indsort",
    "Spicks",
    "Pick_info",
    "time_axis",
}


@dataclass(frozen=True)
class PickleEvent:
    """One Step 1 event pickle after schema validation."""

    event_id: str
    path: Path
    time_axis: np.ndarray
    data_fpara: list[np.ndarray]
    stations: np.ndarray
    arrdist: np.ndarray
    spicks: np.ndarray
    pick_info: np.ndarray


@dataclass(frozen=True)
class CandidateRow:
    """Amplitude-pattern screening result for one valid event."""

    event_id: str
    amp_cc: float
    selected: bool
    reason: str


def _event_id_from_path(path: Path) -> str:
    name = path.stem
    suffix = "_fpara"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _safe_normalize(data: np.ndarray) -> np.ndarray:
    max_abs = float(np.nanmax(np.abs(data))) if data.size else 0.0
    if max_abs <= 0.0 or not np.isfinite(max_abs):
        return np.zeros_like(data, dtype=float)
    return np.asarray(data, dtype=float) / max_abs


def discover_event_pickles(input_dir: str | Path) -> list[Path]:
    """Return sorted Step 1 pickle paths from an input directory."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input pickle directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")
    paths = sorted(input_dir.glob("*_fpara.pkl"))
    if not paths:
        raise ValueError(f"No *_fpara.pkl files found in: {input_dir}")
    return paths


def load_event_pickle(path: str | Path) -> PickleEvent:
    """Load and validate one Step 1 pickle file."""
    path = Path(path)
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a top-level dictionary")

    missing = REQUIRED_PICKLE_KEYS - set(payload)
    if missing:
        raise ValueError(f"{path} is missing required Step 2 field(s): {sorted(missing)}")

    time_axis = np.asarray(payload["time_axis"], dtype=float)
    data_fpara = [np.asarray(trace, dtype=float) for trace in payload["data_fpara"]]
    stations = np.asarray(payload["stalst"])
    arrdist = np.asarray(payload["arrdist"], dtype=float)
    spicks = np.asarray(payload["Spicks"], dtype=float)
    pick_info = np.asarray(payload["Pick_info"], dtype=float)

    if time_axis.ndim != 1 or len(time_axis) < 2:
        raise ValueError(f"{path}: time_axis must be a 1-D array with at least two samples")
    nstation = len(data_fpara)
    if nstation == 0:
        raise ValueError(f"{path}: data_fpara is empty")
    for index, trace in enumerate(data_fpara):
        if trace.ndim != 1:
            raise ValueError(f"{path}: data_fpara[{index}] must be a 1-D array")
        if len(trace) != len(time_axis):
            raise ValueError(
                f"{path}: data_fpara[{index}] length {len(trace)} does not match time_axis {len(time_axis)}"
            )
    if stations.shape != (nstation,):
        raise ValueError(f"{path}: stalst shape must be ({nstation},), got {stations.shape}")
    if arrdist.shape != (nstation,):
        raise ValueError(f"{path}: arrdist shape must be ({nstation},), got {arrdist.shape}")
    if spicks.shape != (nstation,):
        raise ValueError(f"{path}: Spicks shape must be ({nstation},), got {spicks.shape}")
    if pick_info.shape != (nstation, 2):
        raise ValueError(f"{path}: Pick_info shape must be ({nstation}, 2), got {pick_info.shape}")

    return PickleEvent(
        event_id=_event_id_from_path(path),
        path=path,
        time_axis=time_axis,
        data_fpara=data_fpara,
        stations=stations,
        arrdist=arrdist,
        spicks=spicks,
        pick_info=pick_info,
    )


def validate_common_station_axis(events: list[PickleEvent]) -> tuple[np.ndarray, np.ndarray]:
    """Ensure all valid input events use the same station and distance axis."""
    if not events:
        raise ValueError("No event pickles were loaded")
    stations = events[0].stations
    arrdist = events[0].arrdist
    for event in events[1:]:
        if len(event.stations) != len(stations) or not np.array_equal(event.stations, stations):
            raise ValueError(
                f"{event.path}: station order differs from first event {events[0].path}"
            )
        if len(event.arrdist) != len(arrdist) or not np.allclose(
            event.arrdist,
            arrdist,
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        ):
            raise ValueError(
                f"{event.path}: arrdist differs from first event {events[0].path}"
            )
    return stations, arrdist


def event_s_window(event: PickleEvent, *, min_snr: float) -> tuple[float, float]:
    """Estimate an event-level S window from high-SNR station picks."""
    snr = event.pick_info[:, 1]
    mask = np.isfinite(event.spicks) & np.isfinite(snr) & (snr > min_snr)
    picks = event.spicks[mask]
    if picks.size < 2:
        raise ValueError(
            f"Event {event.event_id} has fewer than two S picks with SNR > {min_snr:g}"
        )
    win_min, win_max = np.percentile(picks, [10, 90])
    if not np.isfinite(win_min) or not np.isfinite(win_max) or win_max <= win_min:
        raise ValueError(f"Event {event.event_id} has invalid S window")
    return float(win_min), float(win_max)


def preprocess_trace(time: np.ndarray, data: np.ndarray, *, freqmin: float, freqmax: float):
    """Prepare a trace for amplitude analysis using the detection preprocessing chain."""
    from obspy.core.trace import Trace
    from scipy.signal import fftconvolve

    time = np.asarray(time, dtype=float)
    trace = Trace(data=np.asarray(data, dtype=float))
    trace.stats.delta = float(np.median(np.diff(time)))
    trace.detrend(type="constant")
    trace.detrend(type="linear")
    trace.taper(max_percentage=0.05)
    trace.integrate(method="cumtrapz")

    ind_conv = time > 1.0
    trace.detrend(type="constant")
    trace.detrend(type="linear")
    trace.taper(max_percentage=0.05)
    trace.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)

    data_raw = np.zeros_like(trace.data)
    if np.any(ind_conv):
        conv_t = 1.0 / np.sqrt(time[ind_conv])
        data_conv = fftconvolve(trace.data[ind_conv], conv_t, mode="full")[: np.sum(ind_conv)]
        data_raw[ind_conv] = data_conv * np.sqrt(time[ind_conv])
    trace.data = data_raw
    return trace


def preprocess_event_traces(
    time: np.ndarray,
    traces: list[np.ndarray],
    *,
    freqmin: float,
    freqmax: float,
) -> list[np.ndarray]:
    """Preprocess event traces, replacing invalid station traces with zeros."""
    processed: list[np.ndarray] = []
    for trace in traces:
        trace = np.asarray(trace, dtype=float)
        if trace.shape != time.shape or not np.all(np.isfinite(trace)):
            processed.append(np.zeros_like(time, dtype=float))
            continue
        try:
            processed_trace = preprocess_trace(
                time,
                trace,
                freqmin=freqmin,
                freqmax=freqmax,
            ).data
        except Exception:
            processed.append(np.zeros_like(time, dtype=float))
            continue
        processed.append(np.asarray(processed_trace, dtype=float))
    return processed


def amp_analysis(
    time: np.ndarray,
    s_window: tuple[float, float],
    traces: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Measure normalized station amplitude features for preprocessed traces."""
    win_min, win_max = s_window
    ind_s = (time >= win_min - 0.5) & (time <= win_max + 1.5)
    if int(np.sum(ind_s)) < 2:
        raise ValueError(f"S amplitude window [{win_min:.6g}, {win_max:.6g}] is empty")

    stack = np.array([np.asarray(trace, dtype=float) for trace in traces], dtype=float)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    max_p = float(np.max(np.abs(stack[:, ind_s])))
    amp_dist = np.zeros((len(traces), 3), dtype=float)
    if max_p <= 0.0 or not np.isfinite(max_p):
        return np.zeros(len(traces), dtype=float), amp_dist

    for j, trace in enumerate(stack):
        env = np.abs(trace[ind_s])
        duration = np.sum(env >= 0.2 * max_p) / float(np.sum(ind_s))
        amp_dist[j, :] = (
            float(np.max(env)),
            float(duration),
            float(np.sqrt(np.mean(env * env))),
        )

    combined = amp_dist[:, 0] * amp_dist[:, 2]
    return _safe_normalize(combined), amp_dist


def representative_amplitude_profile(res: np.ndarray) -> np.ndarray:
    """Return the median finite, nonzero amplitude profile across events."""
    valid = np.isfinite(res).all(axis=1) & (np.sqrt(np.mean(res * res, axis=1)) > 0.0)
    if not np.any(valid):
        raise ValueError("Amp_distribution contains no finite nonzero amplitude rows")
    return np.median(res[valid, :], axis=0)


def amplitude_correlations(res: np.ndarray, mean_amp: np.ndarray) -> np.ndarray:
    """Correlate each event's amplitude profile with the representative profile."""
    coef = np.zeros(res.shape[0], dtype=float)
    norm_mean = float(np.sqrt(np.mean(mean_amp * mean_amp)))
    if norm_mean <= 0.0 or not np.isfinite(norm_mean):
        return np.full(res.shape[0], np.nan, dtype=float)

    for i in range(res.shape[0]):
        row = np.asarray(res[i, :], dtype=float)
        norm_row = float(np.sqrt(np.mean(row * row)))
        if norm_row <= 0.0 or not np.isfinite(norm_row):
            coef[i] = float("nan")
        else:
            coef[i] = float(np.mean(mean_amp * row) / (norm_mean * norm_row))
    return coef


def plot_all_events_amp_analysis(
    path: str | Path,
    res: np.ndarray,
    arrdist: np.ndarray,
) -> None:
    """Save the all-event amplitude distribution summary figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if res.ndim != 2 or res.shape[0] == 0 or res.shape[1] == 0:
        raise ValueError("No valid event amplitude distributions to plot")

    median_amp = np.median(res, axis=0)
    std_amp = np.std(res, axis=0)
    nsta = res.shape[1]

    bins = np.linspace(0.0, 1.0, 30)
    histogram = np.zeros((len(bins), nsta), dtype=float)
    for i in range(nsta):
        counts, _ = np.histogram(res[:, i], bins=bins)
        max_count = float(np.max(counts)) if counts.size else 0.0
        if max_count > 0.0:
            histogram[: len(counts), i] = counts / max_count

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.3))
    x_grid, y_grid = np.meshgrid(arrdist - np.gradient(arrdist) / 2.0, bins)
    im = ax.pcolormesh(x_grid, y_grid, histogram, cmap="gray_r", shading="auto")

    ymax = max(1.08, float(np.max(median_amp + std_amp / 1.5)) + 0.05)
    ax.errorbar(
        arrdist,
        median_amp,
        yerr=std_amp / 2.0,
        fmt="-o",
        color="r",
        ecolor="r",
        capsize=5,
        markersize=3,
        linewidth=1.2,
    )

    ax.set_ylim(-0.05, ymax)
    xmin, xmax = np.percentile(arrdist, [0, 100])
    ax.set_xlim(xmin * 1.01 - 0.01 * xmax, 0.99 * xmax + 0.01 * xmin)
    ax.set_xlabel("Distance to array center (km)", fontsize=16)
    ax.set_ylabel("Normalized likelihood", fontsize=16)
    ax.text(0.0, 1.02, "SW", transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    ax.text(1.0, 1.02, "NE", transform=ax.transAxes, ha="right", va="bottom", fontsize=12)
    ax.tick_params(labelsize=14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("Normalized PDF", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_fztw_station_summary(
    res: np.ndarray,
    stations: np.ndarray,
    arrdist: np.ndarray,
) -> None:
    """Print candidate stations whose median-filtered amplitudes exceed threshold."""
    median_amp = np.median(res, axis=0)
    median_filtered = signal.medfilt(
        median_amp,
        kernel_size=MEDIAN_FILTER_KERNEL,
    )
    ind_fz = np.where(median_filtered > FZTW_THRESHOLD)[0]

    print("FZTW candidate stations:")
    if ind_fz.size == 0:
        print("  No FZTW candidate stations found with median-filtered amplitude > 0.2.")
        return
    print("  index  station  arrdist_km  median_amp  median_filtered")
    for index in ind_fz:
        print(
            f"  {index:5d}  {str(stations[index]):>7s}  "
            f"{float(arrdist[index]):10.6f}  "
            f"{float(median_amp[index]):10.6f}  "
            f"{float(median_filtered[index]):15.6f}"
        )


def write_amp_candidates(path: str | Path, rows: list[CandidateRow]) -> None:
    """Write events that pass amplitude-correlation screening."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["evid\tamp_cc"]
    for row in rows:
        if not row.selected:
            continue
        amp_cc = "nan" if not np.isfinite(row.amp_cc) else f"{row.amp_cc:.6f}".rstrip("0").rstrip(".")
        lines.append("\t".join([row.event_id, amp_cc]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step2_detect_fztw(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    min_snr: float = 20.0,
    freqmin: float = 1.0,
    freqmax: float = 15.0,
    amp_cc_threshold: float = 0.95,
) -> dict[str, object]:
    """Run Step 2 amplitude analysis and median-pattern screening."""
    if freqmax <= freqmin or freqmin <= 0.0:
        raise ValueError("--freqmax must be greater than positive --freqmin")

    pickle_paths = discover_event_pickles(input_dir)
    events = [load_event_pickle(path) for path in pickle_paths]
    stations, arrdist = validate_common_station_axis(events)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_distribution_path = output_dir / AMP_DISTRIBUTION_FILE
    summary_figure_path = output_dir / "All_events_amp_analysis.png"
    candidates_path = output_dir / "amp_candidates.txt"

    evids: list[str] = []
    res_rows: list[np.ndarray] = []
    skipped: list[tuple[str, str]] = []

    for event in events:
        try:
            s_window = event_s_window(event, min_snr=min_snr)
            processed_traces = preprocess_event_traces(
                event.time_axis,
                event.data_fpara,
                freqmin=freqmin,
                freqmax=freqmax,
            )
            combined, _ = amp_analysis(event.time_axis, s_window, processed_traces)
            if float(np.max(combined)) <= 0.0:
                raise ValueError("combined amplitude is zero")
            evids.append(event.event_id)
            res_rows.append(np.asarray(combined, dtype=float))
        except ValueError as exc:
            skipped.append((event.event_id, str(exc)))

    if not res_rows:
        raise ValueError("No valid events were available for amplitude analysis")

    res = np.array(res_rows, dtype=float)
    global_max = float(np.max(res)) if res.size else 0.0
    if global_max > 0.0 and np.isfinite(global_max):
        res /= global_max

    mean_amp = representative_amplitude_profile(res)
    amp_cc = amplitude_correlations(res, mean_amp)
    amp_selected = np.isfinite(amp_cc) & (amp_cc >= amp_cc_threshold)
    candidate_rows = [
        CandidateRow(
            event_id,
            float(coef),
            bool(selected),
            "selected" if selected else ("invalid_amp_cc" if not np.isfinite(coef) else "amp_cc_below_threshold"),
        )
        for event_id, coef, selected in zip(evids, amp_cc, amp_selected)
    ]

    np.savez_compressed(
        amp_distribution_path,
        res=res,
        evids=np.array(evids),
        stations=np.asarray(stations),
        arrdist=np.asarray(arrdist, dtype=float),
        mean_amp=mean_amp,
        amp_cc=amp_cc,
        amp_selected=amp_selected,
        min_snr=np.array(min_snr),
        freqmin=np.array(freqmin),
        freqmax=np.array(freqmax),
        amp_cc_threshold=np.array(amp_cc_threshold)
    )

    plot_all_events_amp_analysis(summary_figure_path, res, np.asarray(arrdist, dtype=float))
    write_amp_candidates(candidates_path, candidate_rows)
    print_fztw_station_summary(res, np.asarray(stations), np.asarray(arrdist, dtype=float))

    return {
        "amp_distribution": amp_distribution_path,
        "summary_figure": summary_figure_path,
        "candidates": candidates_path,
    }


def main() -> None:
    """Parse CLI options, run Step 2 amplitude analysis, and print outputs."""
    parser = argparse.ArgumentParser(
        description="Compute FZTW amplitude distributions from Step 1 *_fpara.pkl files"
    )
    parser.add_argument("input_dir", help="Directory containing Step 1 *_fpara.pkl files")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--min-snr", type=float, default=20.0)
    parser.add_argument("--freqmin", type=float, default=1.0)
    parser.add_argument("--freqmax", type=float, default=15.0)
    parser.add_argument("--amp-cc-threshold", type=float, default=0.95)
    args = parser.parse_args()

    result = run_step2_detect_fztw(
        args.input_dir,
        args.output_dir,
        min_snr=args.min_snr,
        freqmin=args.freqmin,
        freqmax=args.freqmax,
        amp_cc_threshold=args.amp_cc_threshold,
    )
    print("Step2 FZTW detection done:")
    for key, value in result.items():
        if value is not None:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
