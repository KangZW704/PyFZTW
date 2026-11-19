"""Prepare PyFZTW inversion inputs from Step 1 pickles and Step 2 candidates.


1. Load the reference event and candidate fault-parallel waveforms from Step 1
   pickle files, using the Step 2 amplitude-correlation candidates as the event
   list.
2. Extract the selected station range and identify a template S-wave window for
   the reference event.
3. If ``stage == 0``, write inversion files directly from the reference event
   template window. If ``stage == 1``, screen candidates by template waveform
   correlation, align selected events by their best lags, average the aligned
   waveforms, and write inversion-ready ``.disp`` files and metadata.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

try:
    from .step2_detect_fztw import PickleEvent, load_event_pickle, preprocess_trace
except ImportError:  # pragma: no cover - supports direct script execution
    from step2_detect_fztw import PickleEvent, load_event_pickle, preprocess_trace  # type: ignore


DEFAULT_CRUST_LINES = [
    "v1 0 3 5",
    "vfr 0 0.1 0.9",
    "v3r 0 0.8 1.2",
    "q1 1 200",
    "qf 0 10 100",
    "q3 1 200",
    "density 1 2.5",
    "widthf 0 0.1 0.5",
    "edge 0 -0.15 0.15",
]

DEFAULT_EQ_LINES = [
    "x 0 0.01 0.9",
    "y 1 0",
    "z 0 1 15",
]


@dataclass(frozen=True)
class CandidateEvent:
    """One amplitude-screened candidate event from ``amp_candidates.txt``."""

    event_id: str
    amp_cc: float


@dataclass(frozen=True)
class DetectionRow:
    """Selection metadata for one candidate event."""

    event_id: str
    amp_cc: float
    template_cc: float
    selected: bool
    reason: str


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def parse_amp_candidates(path: str | Path) -> list[CandidateEvent]:
    """Read Step 2 candidate rows, accepting two-column and four-column formats."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"amp_candidates file not found: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    if not lines:
        raise ValueError(f"amp_candidates file is empty: {path}")

    header = lines[0].split()
    index = {name: i for i, name in enumerate(header)}
    if "evid" not in index:
        raise ValueError(f"{path}: amp_candidates header must include 'evid'")
    amp_index = index.get("amp_cc")
    selected_index = index.get("selected")

    rows: list[CandidateEvent] = []
    for line_no, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) < len(header):
            raise ValueError(
                f"{path} line {line_no}: expected at least {len(header)} columns, got {len(parts)}"
            )

        if selected_index is not None:
            selected_text = parts[selected_index].strip().lower()
            if selected_text not in {"1", "true", "yes", "y"}:
                continue

        amp_cc = float("nan")
        if amp_index is not None:
            amp_cc = float(parts[amp_index])
        rows.append(CandidateEvent(parts[index["evid"]], amp_cc))

    if not rows:
        raise ValueError(f"No candidate events found in: {path}")
    return rows


def load_pickle_event(data_dir: str | Path, event_id: str) -> PickleEvent:
    """Load ``{event_id}_fpara.pkl`` from a Step 1 data directory."""
    path = Path(data_dir) / f"{event_id}_fpara.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Step 1 pickle not found for event {event_id}: {path}")
    return load_event_pickle(path)


def station_name_range(
    stations: list[str],
    start: str,
    end: str,
) -> tuple[int, int, list[str]]:
    """Resolve an inclusive station-name range into indices and station names."""
    index = {station: i for i, station in enumerate(stations)}
    missing = [station for station in (start, end) if station not in index]
    if missing:
        raise ValueError(f"Station range endpoint(s) not found: {missing}")

    nmin = index[start]
    nmax = index[end]
    if nmin > nmax:
        nmin, nmax = nmax, nmin
    return nmin, nmax, stations[nmin : nmax + 1]


def _event_trace_by_station(event: PickleEvent) -> dict[str, np.ndarray]:
    return {
        str(station): np.asarray(trace, dtype=float)
        for station, trace in zip(event.stations, event.data_fpara)
    }


def extract_template_window(
    event: PickleEvent,
    selected_stations: list[str],
    *,
    freqmin: float,
    freqmax: float,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """Extract aligned S-window data for selected stations of one pickle event."""
    event_by_station = _event_trace_by_station(event)
    tref = np.asarray(event.time_axis, dtype=float)
    if tref.ndim != 1 or len(tref) < 4:
        raise ValueError(f"Event {event.event_id} has too few samples")

    data_2d: list[np.ndarray] = []
    norm_2d = -1.0
    for station_name in selected_stations:
        if station_name not in event_by_station:
            raise ValueError(f"Event {event.event_id} is missing station {station_name}")
        trace = np.asarray(event_by_station[station_name], dtype=float)
        if trace.shape != tref.shape or not np.all(np.isfinite(trace)):
            raise ValueError(f"Event {event.event_id} station {station_name} has invalid waveform")
        processed = preprocess_trace(
            tref,
            trace,
            freqmin=freqmin,
            freqmax=freqmax,
        ).data
        data_2d.append(np.asarray(processed, dtype=float))
        norm_2d = max(norm_2d, float(np.max(np.abs(processed))))

    if norm_2d <= 0.0 or not np.isfinite(norm_2d):
        raise ValueError(f"Event {event.event_id} has zero template-window amplitude")

    normalized = [data / norm_2d for data in data_2d]
    env_mean = np.zeros_like(normalized[0], dtype=float)
    for data in normalized:
        env_mean += np.abs(signal.hilbert(data))
    max_env = float(np.max(env_mean))
    if max_env <= 0.0 or not np.isfinite(max_env):
        raise ValueError(f"Event {event.event_id} has zero template envelope")
    env_mean /= max_env

    ind_search = tref > 0.0
    if int(np.sum(ind_search)) < 4:
        ind_search = np.ones_like(tref, dtype=bool)

    peaks, properties = signal.find_peaks(
        env_mean[ind_search],
        height=0.5,
        width=20,
        rel_height=0.8,
    )
    if len(peaks) == 0:
        raise ValueError(f"Event {event.event_id} has no template-window envelope peak")

    search_indices = np.where(ind_search)[0]
    f_time = interp1d(np.arange(len(tref), dtype=float), tref)
    left_ips = search_indices[0] + properties["left_ips"]
    t_s = float(np.min(f_time(left_ips)))
    ind_ref = np.where((tref >= t_s - 0.5) & (tref <= t_s + 1.5))[0]
    if len(ind_ref) < 2:
        raise ValueError(f"Event {event.event_id} has too few samples near template S")

    taxis_s0 = np.linspace(t_s - 0.5, t_s + 1.5, len(ind_ref))
    data_int: list[np.ndarray] = []
    for data in data_2d:
        f_data = interp1d(tref, data, kind="linear", fill_value="extrapolate")
        data_int.append(np.asarray(f_data(taxis_s0), dtype=float))

    data_array = np.array(data_int, dtype=float)
    if float(np.max(np.abs(data_array))) <= 0.0:
        raise ValueError(f"Event {event.event_id} has zero S-window amplitude")
    return data_array, taxis_s0, t_s, selected_stations


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Return normalized full-correlation peak and its index."""
    corr = np.correlate(a, b, mode="full")
    norm = math.sqrt(float(np.sum(a * a))) * math.sqrt(float(np.sum(b * b)))
    if norm <= 0.0 or not np.isfinite(norm):
        return float("nan"), 0
    idx = int(np.argmax(corr))
    return float(corr[idx] / norm), idx


def stack_reference_event(
    reference: PickleEvent,
    selected_stations: list[str],
    *,
    freqmin: float,
    freqmax: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[DetectionRow], float]:
    """Return the reference event template window as the stage-0 stack."""
    ref_data, ref_time, ref_s, station_names = extract_template_window(
        reference,
        selected_stations,
        freqmin=freqmin,
        freqmax=freqmax,
    )
    return (
        ref_data,
        ref_time - ref_s,
        station_names,
        [DetectionRow(reference.event_id, 1.0, 1.0, True, "reference_event")],
        ref_s,
    )


def stack_template_selected_events(
    events: dict[str, PickleEvent],
    candidate_rows: list[CandidateEvent],
    reference_event: str,
    selected_stations: list[str],
    *,
    template_cc_threshold: float,
    freqmin: float,
    freqmax: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[DetectionRow], float]:
    """Select template-matching events and return their stacked S-window traces."""
    if reference_event not in events:
        raise ValueError(f"Reference event not found in loaded pickles: {reference_event}")

    ref_data, ref_time, ref_s, station_names = extract_template_window(
        events[reference_event],
        selected_stations,
        freqmin=freqmin,
        freqmax=freqmax,
    )
    data_sum = np.array(ref_data, copy=True)
    selected_count = 1
    detection: list[DetectionRow] = [
        DetectionRow(reference_event, 1.0, 1.0, True, "reference_event")
    ]
    ref_delta = float(ref_time[1] - ref_time[0])
    ref_relative_time = ref_time - ref_s

    seen = {reference_event}
    for candidate in candidate_rows:
        event_id = candidate.event_id
        if event_id in seen:
            continue
        seen.add(event_id)
        if event_id not in events:
            detection.append(
                DetectionRow(event_id, candidate.amp_cc, float("nan"), False, "missing_pickle")
            )
            continue

        try:
            data, time, event_s, _ = extract_template_window(
                events[event_id],
                selected_stations,
                freqmin=freqmin,
                freqmax=freqmax,
            )
        except ValueError as exc:
            detection.append(
                DetectionRow(event_id, candidate.amp_cc, float("nan"), False, str(exc))
            )
            continue

        ratios: list[float] = []
        shifts: list[float] = []
        for station_idx in range(len(selected_stations)):
            ratio, shift_idx = normalized_correlation(data[station_idx], ref_data[station_idx])
            lag = shift_idx - (len(ref_data[station_idx]) - 1)
            ratios.append(ratio)
            shifts.append(float(lag * ref_delta))

        template_cc = float(np.nanmean(ratios))
        if not np.isfinite(template_cc) or template_cc <= template_cc_threshold:
            detection.append(
                DetectionRow(
                    event_id,
                    candidate.amp_cc,
                    template_cc,
                    False,
                    "template_cc_below_threshold",
                )
            )
            continue

        for station_idx, shift in enumerate(shifts):
            f_data = interp1d(time, data[station_idx], kind="linear", fill_value="extrapolate")
            data_sum[station_idx] += f_data(event_s + ref_relative_time + shift)
        selected_count += 1
        detection.append(DetectionRow(event_id, candidate.amp_cc, template_cc, True, "selected"))

    return data_sum / float(selected_count), ref_relative_time, station_names, detection, ref_s


def reference_mean_s_pick(event: PickleEvent, selected_stations: list[str]) -> float:
    """Average the reference event S picks over selected stations."""
    station_to_pick = {
        str(station): float(s_pick) for station, s_pick in zip(event.stations, event.spicks)
    }
    picks = np.array(
        [
            station_to_pick[station]
            for station in selected_stations
            if station in station_to_pick and np.isfinite(station_to_pick[station])
        ],
        dtype=float,
    )
    if picks.size == 0:
        raise ValueError(
            f"Reference event {event.event_id} has no finite S picks in selected stations"
        )
    return float(np.mean(picks))


def relative_station_distances(arrdist: np.ndarray) -> np.ndarray:
    """Convert selected station distances to start at zero."""
    arrdist = np.asarray(arrdist, dtype=float)
    if arrdist.size == 0:
        raise ValueError("No station distances available for selected station range")
    return arrdist - float(arrdist[0])


def write_selected_events_table(path: str | Path, rows: list[DetectionRow]) -> None:
    """Write only events selected for the final stack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if row.selected]
    lines = ["stack_order\tevid\tamp_cc\ttemplate_cc\treason"]
    for i, row in enumerate(selected, start=1):
        lines.append(
            "\t".join(
                [
                    str(i),
                    row.event_id,
                    _fmt(row.amp_cc),
                    _fmt(row.template_cc),
                    row.reason,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_station_table(path: str | Path, stations: list[str], arrdist: np.ndarray) -> None:
    """Write the station geometry table consumed by PyFZTW inversion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for station, distance in zip(stations, arrdist):
        lines.append("\t".join([station, _fmt(float(distance)), "0", "0"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_crust_template(path: str | Path) -> Path:
    """Write a starter crust parameter file for inversion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(DEFAULT_CRUST_LINES) + "\n", encoding="utf-8")
    return path


def write_eqs_template(path: str | Path, event_id: str) -> Path:
    """Write a starter earthquake parameter file for the reference event."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([str(event_id), *DEFAULT_EQ_LINES]) + "\n", encoding="utf-8")
    return path


def write_disp_files(
    output_dir: str | Path,
    reference_event: str,
    stations: list[str],
    time: np.ndarray,
    data: np.ndarray,
    *,
    maxfreq: float,
    accuracy: float,
    gibbs: float,
) -> list[Path]:
    """Write stacked displacement windows, one file per selected station."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for station, trace in zip(stations, data):
        path = output_dir / f"{station}_ref_{reference_event}.disp"
        lines = [
            f"eq {reference_event}",
            f"station {station}",
            f"window {_fmt(float(time[0]))} {_fmt(float(time[-1]))}",
            f"maxfreq {_fmt(maxfreq)}",
            f"accuracy {_fmt(accuracy)}",
            f"gibbs {_fmt(gibbs)}",
        ]
        lines.extend(f"{_fmt(float(t))}   {_fmt(float(y))}" for t, y in zip(time, trace))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def plot_fztw_data(
    path: str | Path,
    time: np.ndarray,
    reference_data: np.ndarray,
    stacked_data: np.ndarray,
    stations: list[str],
    *,
    reference_event: str,
    offset: float = 1.25,
) -> None:
    """Save a comparison plot of the reference and stacked FZTW waveforms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    x = np.asarray(time, dtype=float)
    original_norm = float(np.max(np.abs(reference_data)))
    stacked_norm = float(np.max(np.abs(stacked_data)))
    for i, station in enumerate(stations):
        original = np.asarray(reference_data[i], dtype=float)
        stacked = np.asarray(stacked_data[i], dtype=float)
        if original_norm > 0.0 and np.isfinite(original_norm):
            original = original / original_norm
        if stacked_norm > 0.0 and np.isfinite(stacked_norm):
            stacked = stacked / stacked_norm
        ax.plot(x, original + i * offset, "k", linewidth=1.0, label="Original" if i == 0 else "")
        ax.plot(x, stacked + i * offset, "r", linewidth=1.0, label="Stacked" if i == 0 else "")

    ax.set_yticks(np.arange(len(stations)) * offset)
    ax.set_yticklabels(stations)
    ax.set_xlabel("Time to S pick (s)", fontsize=16)
    ax.set_ylabel("Station", fontsize=16)
    ax.set_title(f"FZTW stacked vs original - {reference_event}", fontsize=16)
    ax.legend(loc=1, prop={"size": 14})
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_step3_prepare_inversion_model(
    amp_candidates_file: str | Path | None = None,
    *,
    data_dir: str | Path,
    ref_event: str,
    station_range: tuple[str, str],
    stage: int = 1,
    output_dir: str | Path = ".",
    template_cc_threshold: float = 0.85,
    freqmin: float = 1.0,
    freqmax: float = 15.0,
    accuracy: float = 1e-5,
    gibbs: float = 5.0,
    plot_offset: float = 1.25,
) -> dict[str, object]:
    """Run Step 3 and return generated inversion input paths."""
    if stage not in {0, 1}:
        raise ValueError("--stage must be 0 or 1")
    if freqmax <= freqmin or freqmin <= 0.0:
        raise ValueError("--freqmax must be greater than positive --freqmin")

    ref_event = str(ref_event)
    if stage == 1:
        if amp_candidates_file is None:
            raise ValueError("amp_candidates_file is required when --stage 1")
        candidate_rows = parse_amp_candidates(amp_candidates_file)
        candidate_by_event = {row.event_id: row for row in candidate_rows}
        if ref_event not in candidate_by_event:
            candidate_rows.insert(0, CandidateEvent(ref_event, 1.0))
            candidate_by_event[ref_event] = candidate_rows[0]
    else:
        candidate_rows = []

    reference = load_pickle_event(data_dir, ref_event)
    stations_all = [str(station) for station in reference.stations]
    nmin, nmax, selected_stations = station_name_range(
        stations_all,
        station_range[0],
        station_range[1],
    )
    selected_arrdist = reference.arrdist[nmin : nmax + 1]

    if stage == 0:
        stacked, disp_time, disp_stations, detection_rows, _reference_s = stack_reference_event(
            reference,
            selected_stations,
            freqmin=freqmin,
            freqmax=freqmax,
        )
    else:
        events: dict[str, PickleEvent] = {ref_event: reference}
        for row in candidate_rows:
            if row.event_id == ref_event or row.event_id in events:
                continue
            try:
                events[row.event_id] = load_pickle_event(data_dir, row.event_id)
            except FileNotFoundError:
                continue
        stacked, disp_time, disp_stations, detection_rows, _reference_s = stack_template_selected_events(
            events,
            candidate_rows,
            ref_event,
            selected_stations,
            template_cc_threshold=template_cc_threshold,
            freqmin=freqmin,
            freqmax=freqmax,
        )
        ref_data, _, _, _ = extract_template_window(
            reference,
            selected_stations,
            freqmin=freqmin,
            freqmax=freqmax,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stations_path = output_dir / "stations.txt"
    crust_path = output_dir / "crust.txt"
    eqs_path = output_dir / "eqs.txt"

    mean_s_pick = reference_mean_s_pick(reference, selected_stations)
    disp_output_time = np.asarray(disp_time, dtype=float) + mean_s_pick
    station_output_dist = relative_station_distances(selected_arrdist)

    write_station_table(stations_path, disp_stations, station_output_dist)
    write_crust_template(crust_path)
    write_eqs_template(eqs_path, ref_event)
    disp_paths = write_disp_files(
        output_dir / "DISP",
        ref_event,
        disp_stations,
        disp_output_time,
        stacked,
        maxfreq=freqmax,
        accuracy=accuracy,
        gibbs=gibbs,
    )

    result: dict[str, object] = {
        "stations": stations_path,
        "crust": crust_path,
        "eqs": eqs_path,
        "disp_files": disp_paths,
        "selected_event_count": sum(1 for row in detection_rows if row.selected),
        "station_range": f"{disp_stations[0]}-{disp_stations[-1]}",
        "reference_mean_s_pick": mean_s_pick,
        "stage": stage,
    }
    if stage == 1:
        selected_events_path = output_dir / "fztw_selected_events.txt"
        figure_path = output_dir / f"fztw_data_{ref_event}.png"
        write_selected_events_table(selected_events_path, detection_rows)
        plot_fztw_data(
            figure_path,
            disp_time,
            ref_data,
            stacked,
            disp_stations,
            reference_event=ref_event,
            offset=plot_offset,
        )
        result["selected_events"] = selected_events_path
        result["figure"] = figure_path
        result["candidate_event_count"] = len(candidate_rows)
    return result


def main() -> None:
    """Parse CLI options, run Step 3, and print generated outputs."""
    parser = argparse.ArgumentParser(
        description="Prepare PyFZTW inversion inputs from Step 1 pickles and Step 2 candidates"
    )
    parser.add_argument(
        "amp_candidates_file",
        nargs="?",
        default=None,
        help="Path to Step 2 amp_candidates.txt; required when --stage 1",
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing Step 1 *_fpara.pkl files")
    parser.add_argument("--ref-event", required=True, help="Reference event id")
    parser.add_argument(
        "--station-range",
        nargs=2,
        required=True,
        metavar=("START", "END"),
        help="Inclusive station-name range, for example: B416 B423",
    )
    parser.add_argument("--stage", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--template-cc-threshold", type=float, default=0.85)
    parser.add_argument("--freqmin", type=float, default=1.0)
    parser.add_argument("--freqmax", type=float, default=15.0)
    parser.add_argument("--accuracy", type=float, default=1e-5)
    parser.add_argument("--gibbs", type=float, default=5.0)
    parser.add_argument("--plot-offset", type=float, default=1.25)
    args = parser.parse_args()

    result = run_step3_prepare_inversion_model(
        args.amp_candidates_file,
        data_dir=args.data_dir,
        ref_event=args.ref_event,
        station_range=(args.station_range[0], args.station_range[1]),
        stage=args.stage,
        output_dir=args.output_dir,
        template_cc_threshold=args.template_cc_threshold,
        freqmin=args.freqmin,
        freqmax=args.freqmax,
        accuracy=args.accuracy,
        gibbs=args.gibbs,
        plot_offset=args.plot_offset,
    )
    print("Step3 inversion input preparation done:")
    for key, value in result.items():
        if key == "disp_files":
            print(f"  {key}: {len(value)} files")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
