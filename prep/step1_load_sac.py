"""Load SAC pairs, rotate to fault-parallel waveforms, and save event pickles.

This step reads paired N/E SAC files, rotates them into fault coordinates,
picks S arrivals, and writes one ``*_fpara.pkl`` file per readable event.
"""

from __future__ import annotations

import argparse
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

try:
    from obspy import read as obspy_read
except Exception as exc:  # pragma: no cover
    obspy_read = None
    _OBSPY_IMPORT_ERROR = exc
else:
    _OBSPY_IMPORT_ERROR = None

try:
    import utm
except Exception as exc:  # pragma: no cover
    utm = None
    _UTM_IMPORT_ERROR = exc
else:
    _UTM_IMPORT_ERROR = None


PICK_BANDPASS_FREQMIN = 0.5
PICK_BANDPASS_FREQMAX = 20.0


@dataclass(frozen=True)
class InputRow:
    """One input table row pointing to paired north/east SAC files."""

    event_id: str
    station_name: str
    north_sac: Path
    east_sac: Path


@dataclass(frozen=True)
class SacMetadata:
    """SAC header metadata needed for geometry checks and station projection."""

    station_name: str
    stla: float
    stlo: float
    station_depth_km: float
    evla: float
    evlo: float
    evdp_km: float
    component: str


@dataclass(frozen=True)
class StationInfo:
    """Station coordinates used to define the array axis."""

    name: str
    lon: float
    lat: float
    elev_m: float


@dataclass
class FaultComponents:
    """Rotated horizontal components and metadata for one SAC pair."""

    row: InputRow
    metadata: SacMetadata
    time: np.ndarray
    fault_parallel: np.ndarray
    fault_normal: np.ndarray


@dataclass
class LoadedStation:
    """Loaded waveform and pick data for one event/station row."""

    row: InputRow
    time: np.ndarray | None
    fault_parallel: np.ndarray | None
    s_pick: float
    s_snr: float
    vs_est: float


def _require_obspy() -> None:
    if obspy_read is None:
        raise ImportError(f"obspy is required: {_OBSPY_IMPORT_ERROR}")


def _require_utm() -> None:
    if utm is None:
        raise ImportError(f"utm is required for station projection: {_UTM_IMPORT_ERROR}")


def parse_input_file(path: str | Path) -> list[InputRow]:
    """Read the raw SAC-pair input table used by the Step1 workflow."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    base_dir = path.parent
    rows: list[InputRow] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"{path} line {line_no}: expected 4 columns: evid station north_sac east_sac"
            )
        event_id, station_name, north_sac, east_sac = parts
        event_id = event_id.lstrip("\ufeff")
        station_name = station_name.lstrip("\ufeff")
        north_path = Path(north_sac)
        east_path = Path(east_sac)
        if not north_path.is_absolute():
            north_path = base_dir / north_path
        if not east_path.is_absolute():
            east_path = base_dir / east_path
        rows.append(InputRow(event_id, station_name, north_path, east_path))

    if not rows:
        raise ValueError(f"No valid data rows found in input file: {path}")
    return rows


def _read_trace(sac_path: Path):
    if not sac_path.exists():
        raise FileNotFoundError(f"SAC file not found: {sac_path}")
    _require_obspy()
    assert obspy_read is not None
    trace = obspy_read(str(sac_path))[0]
    sac = getattr(trace.stats, "sac", None)
    if sac is None:
        raise ValueError(f"SAC header missing in file: {sac_path}")
    return trace, sac


def _get_sac_float(sac, key: str, file_path: Path) -> float:
    value = getattr(sac, key, None)
    if value is None:
        raise ValueError(f"SAC header '{key}' missing in file: {file_path}")
    return float(value)


def _get_station_depth_km(sac, file_path: Path) -> float:
    stdp = getattr(sac, "stdp", None)
    if stdp is not None:
        return float(stdp) / 1000.0
    stel = getattr(sac, "stel", None)
    if stel is not None:
        return -float(stel) / 1000.0
    raise ValueError(f"SAC header 'stel/stdp' missing in file: {file_path}")


def _get_component(trace, sac) -> str:
    kcmpnm = str(getattr(sac, "kcmpnm", "")).strip()
    if kcmpnm:
        return kcmpnm.upper()
    channel = str(getattr(trace.stats, "channel", "")).strip()
    return channel.upper()


def _extract_metadata(trace, sac, file_path: Path) -> SacMetadata:
    station = str(getattr(sac, "kstnm", "")).strip()
    if not station:
        station = str(getattr(trace.stats, "station", "")).strip()

    return SacMetadata(
        station_name=station,
        stla=_get_sac_float(sac, "stla", file_path),
        stlo=_get_sac_float(sac, "stlo", file_path),
        station_depth_km=_get_station_depth_km(sac, file_path),
        evla=_get_sac_float(sac, "evla", file_path),
        evlo=_get_sac_float(sac, "evlo", file_path),
        evdp_km=_get_sac_float(sac, "evdp", file_path),
        component=_get_component(trace, sac),
    )


def _angles_close(a: float, b: float, tol: float = 20.0) -> bool:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff <= tol


def _warn_component_if_suspicious(trace, sac, file_path: Path, *, expected: str) -> None:
    component = _get_component(trace, sac)
    cmpaz = getattr(sac, "cmpaz", None)
    cmpinc = getattr(sac, "cmpinc", None)

    label_ok = bool(component) and expected in component
    az_expected = 0.0 if expected == "N" else 90.0
    az_ok = cmpaz is not None and _angles_close(float(cmpaz), az_expected)
    inc_ok = cmpinc is None or abs(float(cmpinc) - 90.0) <= 20.0

    if component and label_ok and inc_ok:
        return
    if cmpaz is not None and az_ok and inc_ok:
        return

    warnings.warn(
        (
            f"{file_path}: component header does not clearly match expected {expected}; "
            f"kcmpnm/channel='{component}', cmpaz={cmpaz}, cmpinc={cmpinc}. "
            "The fixed input column order is still used."
        ),
        stacklevel=2,
    )


def _assert_close(
    left: float,
    right: float,
    *,
    label: str,
    left_path: Path,
    right_path: Path,
    atol: float,
) -> None:
    if abs(left - right) > atol:
        raise ValueError(
            f"N/E SAC metadata mismatch for {label}: {left_path}={left}, {right_path}={right}"
        )


def _validate_pair(
    row: InputRow,
    north_trace,
    north_sac,
    east_trace,
    east_sac,
) -> tuple[SacMetadata, SacMetadata]:
    north_meta = _extract_metadata(north_trace, north_sac, row.north_sac)
    east_meta = _extract_metadata(east_trace, east_sac, row.east_sac)

    for meta, file_path in ((north_meta, row.north_sac), (east_meta, row.east_sac)):
        if meta.station_name and meta.station_name != row.station_name:
            warnings.warn(
                (
                    f"{file_path}: SAC station '{meta.station_name}' differs from input "
                    f"station '{row.station_name}'; using input station name."
                ),
                stacklevel=2,
            )

    _warn_component_if_suspicious(north_trace, north_sac, row.north_sac, expected="N")
    _warn_component_if_suspicious(east_trace, east_sac, row.east_sac, expected="E")

    if int(north_trace.stats.npts) != int(east_trace.stats.npts):
        raise ValueError(
            f"N/E SAC npts mismatch: {row.north_sac}={north_trace.stats.npts}, "
            f"{row.east_sac}={east_trace.stats.npts}"
        )
    _assert_close(
        float(north_trace.stats.delta),
        float(east_trace.stats.delta),
        label="delta",
        left_path=row.north_sac,
        right_path=row.east_sac,
        atol=1e-9,
    )

    for key, atol in (
        ("stla", 1e-5),
        ("stlo", 1e-5),
        ("station_depth_km", 1e-4),
        ("evla", 1e-5),
        ("evlo", 1e-5),
        ("evdp_km", 1e-4),
    ):
        _assert_close(
            float(getattr(north_meta, key)),
            float(getattr(east_meta, key)),
            label=key,
            left_path=row.north_sac,
            right_path=row.east_sac,
            atol=atol,
        )

    return north_meta, east_meta


def _hypocentral_distance_km(meta: SacMetadata) -> float:
    earth_radius_km = 6371.0
    phi1 = math.radians(meta.evla)
    phi2 = math.radians(meta.stla)
    dphi = math.radians(meta.stla - meta.evla)
    dlambda = math.radians(meta.stlo - meta.evlo)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    horizontal = 2.0 * earth_radius_km * math.atan2(
        math.sqrt(a),
        math.sqrt(max(0.0, 1.0 - a)),
    )
    vertical = meta.evdp_km + meta.station_depth_km
    return float(math.sqrt(horizontal * horizontal + vertical * vertical))


def rotate_ne_to_fault(
    north: np.ndarray,
    east: np.ndarray,
    *,
    fault_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate north/east traces into fault-parallel and fault-normal components."""
    theta = math.radians(fault_angle + 180.0)
    fault_parallel = east * math.sin(theta) + north * math.cos(theta)
    fault_normal = east * math.cos(theta) - north * math.sin(theta)
    return np.asarray(fault_parallel, dtype=float), np.asarray(fault_normal, dtype=float)


def _time_axis(npts: int, dt: float, time_start: float, time_end: float, source: Path) -> np.ndarray:
    actual_end = time_start + (npts - 1) * dt
    tolerance = max(2.0 * dt, 0.05)
    if abs(actual_end - time_end) > tolerance:
        warnings.warn(
            (
                f"{source}: trace time span [{time_start}, {actual_end:.6f}] differs "
                f"from requested [{time_start}, {time_end}]"
            ),
            stacklevel=2,
        )
    return time_start + np.arange(npts, dtype=float) * dt


def load_fault_components(
    row: InputRow,
    *,
    fault_angle: float,
    time_start: float,
    time_end: float,
) -> FaultComponents:
    """Load a SAC pair, validate metadata, rotate it, and return waveform arrays."""
    north_trace, north_sac = _read_trace(row.north_sac)
    east_trace, east_sac = _read_trace(row.east_sac)
    north_meta, _ = _validate_pair(row, north_trace, north_sac, east_trace, east_sac)

    dt = float(north_trace.stats.delta)
    npts = int(north_trace.stats.npts)
    time = _time_axis(npts, dt, time_start, time_end, row.north_sac)
    fault_parallel, fault_normal = rotate_ne_to_fault(
        np.asarray(north_trace.data, dtype=float),
        np.asarray(east_trace.data, dtype=float),
        fault_angle=fault_angle,
    )
    return FaultComponents(
        row=row,
        metadata=north_meta,
        time=time,
        fault_parallel=fault_parallel,
        fault_normal=fault_normal,
    )


def lonlat_to_utm_km(lon: float, lat: float) -> tuple[float, float]:
    """Project longitude/latitude to UTM coordinates in kilometers."""
    _require_utm()
    assert utm is not None
    x_m, y_m, *_ = utm.from_latlon(lat, lon)
    return float(x_m) / 1000.0, float(y_m) / 1000.0


def compute_array_geometry(
    station_infos: dict[str, StationInfo],
) -> tuple[list[str], np.ndarray, float, float, float, float]:
    """Order stations along the array axis and return signed distances."""
    if len(station_infos) < 2:
        raise ValueError("At least two stations are required for array geometry")

    by_lon = sorted(station_infos.values(), key=lambda item: item.lon)
    x1, y1 = lonlat_to_utm_km(by_lon[0].lon, by_lon[0].lat)
    x2, y2 = lonlat_to_utm_km(by_lon[-1].lon, by_lon[-1].lat)
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist <= 0.0:
        raise ValueError("Cannot compute array strike from coincident endpoint stations")

    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist

    distances: list[tuple[str, float]] = []
    for name, info in station_infos.items():
        x0, y0 = lonlat_to_utm_km(info.lon, info.lat)
        distances.append((name, (x0 - xc) * dx + (y0 - yc) * dy))

    distances.sort(key=lambda item: item[1])
    return (
        [name for name, _ in distances],
        np.array([distance for _, distance in distances], dtype=float),
        xc,
        yc,
        dx,
        dy,
    )


def _station_info_from_components(row: InputRow, components) -> StationInfo:
    meta = components.metadata
    return StationInfo(
        name=row.station_name,
        lon=float(meta.stlo),
        lat=float(meta.stla),
        elev_m=-1000.0 * float(meta.station_depth_km),
    )


def _s_velocity_window(hypo_dist_km: float, trace_end: float) -> tuple[float, float]:
    """Estimate a physically reasonable S-arrival search window.

    The initial window is centered on a reference S velocity of 3.2 km/s, with an
    empirical tolerance of about -2 s for early arrivals and +3 s for late arrivals.
    The implied velocity bounds are clipped to 2.0-4.0 km/s, and the final window is
    limited by both 50 s and the available trace duration.

    Returns the start and end time of the S-arrival search window, in seconds.
"""
    if hypo_dist_km <= 0:
        raise ValueError(
            f"Cannot build S velocity window for hypocentral distance {hypo_dist_km:.6g} km"
        )

    denom_fast = hypo_dist_km / 3.2 - 2.0
    denom_slow = hypo_dist_km / 3.2 + 3.0
    if denom_slow <= 0:
        raise ValueError(
            f"Cannot build S velocity window for hypocentral distance {hypo_dist_km:.6g} km"
        )

    vsmax_candidate = float("inf") if denom_fast <= 0 else hypo_dist_km / denom_fast
    vsmax = min(4.0, vsmax_candidate)
    vsmin = max(2.0, hypo_dist_km / denom_slow)
    if vsmax <= 0 or vsmin <= 0 or vsmin > vsmax:
        raise ValueError(
            f"Invalid S velocity bounds for hypocentral distance {hypo_dist_km:.6g} km"
        )

    tmin_s = hypo_dist_km / vsmax
    tmax_s = min(hypo_dist_km / vsmin, 50.0, trace_end)
    if tmax_s <= tmin_s:
        raise ValueError(
            f"Invalid S search window [{tmin_s:.6g}, {tmax_s:.6g}] for distance {hypo_dist_km:.6g} km"
        )
    return float(tmin_s), float(tmax_s)


def _normalize(data: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(data))) if data.size else 0.0
    if max_abs <= 0 or not np.isfinite(max_abs):
        return np.zeros_like(data, dtype=float)
    return np.asarray(data, dtype=float) / max_abs


def _lowpass_envelope(envelope: np.ndarray, dt: float, *, freq: float = 3.0) -> np.ndarray:
    nyquist = 0.5 / dt
    if freq >= nyquist:
        warnings.warn(
            f"Envelope lowpass {freq:g} Hz is not below Nyquist {nyquist:g} Hz; skipping lowpass",
            stacklevel=2,
        )
        return _normalize(envelope)

    sos = signal.butter(4, freq, btype="lowpass", fs=1.0 / dt, output="sos")
    try:
        filtered = signal.sosfiltfilt(sos, envelope)
    except ValueError:
        warnings.warn(
            "Trace is too short for zero-phase envelope lowpass; using unfiltered envelope",
            stacklevel=2,
        )
        filtered = envelope
    return _normalize(np.asarray(filtered, dtype=float))


def _tapered_signal_window(mask: np.ndarray) -> np.ndarray:
    """Return the old workflow's tapered signal window with a low outside floor."""
    mask = np.asarray(mask, dtype=bool)
    window = np.zeros(mask.shape, dtype=float) + 0.01
    if not np.any(mask):
        return window

    segment = np.ones(int(np.sum(mask)), dtype=float)
    try:
        from obspy.core.trace import Trace

        trace = Trace(data=segment)
        trace.taper(max_percentage=0.05)
        segment = np.asarray(trace.data, dtype=float)
    except Exception:
        # A rectangular window is still better than losing an otherwise usable pick.
        segment = np.ones(int(np.sum(mask)), dtype=float)
    window[mask] = segment
    return window


def preprocess_pick_trace(time: np.ndarray, data: np.ndarray, *, freqmin: float, freqmax: float):
    """Detrend, taper, and bandpass one trace before S picking."""
    from obspy.core.trace import Trace

    time = np.asarray(time, dtype=float)
    trace = Trace(data=np.asarray(data, dtype=float))
    trace.stats.delta = float(np.median(np.diff(time)))
    trace.detrend(type="constant")
    trace.detrend(type="linear")
    trace.taper(max_percentage=0.05)
    trace.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
    return trace


def pick_s_arrival(
    time: np.ndarray,
    fault_parallel: np.ndarray,
    fault_normal: np.ndarray,
    *,
    hypo_dist_km: float,
) -> tuple[float, float, int]:
    """Pick the S arrival and return pick time, SNR, and quality flag."""
    dt = float(np.median(np.diff(time)))
    tmin_s, tmax_s = _s_velocity_window(hypo_dist_km, float(time[-1]))
    s_window = (time >= tmin_s) & (time <= tmax_s)
    if int(np.sum(s_window)) < 2:
        raise ValueError(
            f"S search window [{tmin_s:.6g}, {tmax_s:.6g}] has fewer than 2 samples"
        )

    pick_parallel = preprocess_pick_trace(
        time,
        fault_parallel,
        freqmin=PICK_BANDPASS_FREQMIN,
        freqmax=PICK_BANDPASS_FREQMAX,
    ).data
    pick_normal = preprocess_pick_trace(
        time,
        fault_normal,
        freqmin=PICK_BANDPASS_FREQMIN,
        freqmax=PICK_BANDPASS_FREQMAX,
    ).data
    signal_window = _tapered_signal_window(s_window)
    pick_parallel = np.asarray(pick_parallel, dtype=float) * signal_window
    pick_normal = np.asarray(pick_normal, dtype=float) * signal_window
    env_parallel = np.abs(signal.hilbert(pick_parallel))
    env_normal = np.abs(signal.hilbert(pick_normal))
    horizontal_env = np.sqrt(env_parallel * env_parallel + env_normal * env_normal)
    horizontal_env = _lowpass_envelope(_normalize(horizontal_env), dt, freq=3.0)

    env_win = horizontal_env[s_window]
    win_times = time[s_window]
    peaks, properties = signal.find_peaks(
        env_win,
        height=0.5,
        width=20,
        rel_height=0.8,
    )

    if len(peaks) == 0:
        s_pick = hypo_dist_km / 3.2
        s_flag = -1
    else:
        peak_heights = horizontal_env[s_window][peaks]
        best_peak = int(np.argmax(peak_heights))
        left_ip = float(properties["left_ips"][best_peak])
        s_pick = float(np.interp(left_ip, np.arange(len(win_times)), win_times))
        s_flag = 1

    after = horizontal_env[time > s_pick]
    before = horizontal_env[time < s_pick - 0.1]
    if after.size == 0 or before.size < 2:
        warnings.warn(
            f"Cannot compute stable S SNR for s_pick={s_pick:.6g}; using nan",
            stacklevel=2,
        )
        s_snr = float("nan")
    else:
        noise = float(np.std(before))
        if noise <= 0 or not np.isfinite(noise):
            warnings.warn(
                f"Noise level before s_pick={s_pick:.6g} is zero or invalid; using nan SNR",
                stacklevel=2,
            )
            s_snr = float("nan")
        else:
            s_snr = float(s_flag * np.max(after) / noise)

    return float(s_pick), s_snr, int(s_flag)


def _group_rows_by_event(rows: list[InputRow]) -> dict[str, list[InputRow]]:
    grouped: dict[str, list[InputRow]] = {}
    for row in rows:
        grouped.setdefault(row.event_id, []).append(row)
    return grouped


def _collect_station_infos(
    rows: list[InputRow],
    *,
    fault_angle: float,
    time_start: float,
    time_end: float,
) -> dict[str, StationInfo]:
    station_infos: dict[str, StationInfo] = {}
    station_names = {row.station_name for row in rows}

    for row in rows:
        if row.station_name in station_infos:
            continue
        try:
            components = load_fault_components(
                row,
                fault_angle=fault_angle,
                time_start=time_start,
                time_end=time_end,
            )
        except Exception:
            continue
        station_infos[row.station_name] = _station_info_from_components(row, components)
        if len(station_infos) == len(station_names):
            break

    return station_infos


def _load_station(
    row: InputRow,
    *,
    fault_angle: float,
    time_start: float,
    time_end: float,
) -> tuple[LoadedStation, StationInfo | None]:
    try:
        components = load_fault_components(
            row,
            fault_angle=fault_angle,
            time_start=time_start,
            time_end=time_end,
        )
    except Exception:
        return (
            LoadedStation(
                row=row,
                time=None,
                fault_parallel=None,
                s_pick=float("nan"),
                s_snr=float("nan"),
                vs_est=float("nan"),
            ),
            None,
        )

    station_info = _station_info_from_components(row, components)
    hypo_dist = _hypocentral_distance_km(components.metadata)
    try:
        s_pick, s_snr, s_flag = pick_s_arrival(
            components.time,
            components.fault_parallel,
            components.fault_normal,
            hypo_dist_km=hypo_dist,
        )
    except ValueError:
        return (
            LoadedStation(
                row=row,
                time=np.asarray(components.time, dtype=float),
                fault_parallel=np.asarray(components.fault_parallel, dtype=np.float64),
                s_pick=float("nan"),
                s_snr=float("nan"),
                vs_est=float("nan"),
            ),
            station_info,
        )

    if s_flag < 0:
        s_snr = -1.0
    vs_est = float("nan")
    if np.isfinite(s_pick) and s_pick > 0.0:
        vs_est = float(hypo_dist / s_pick)

    return (
        LoadedStation(
            row=row,
            time=np.asarray(components.time, dtype=float),
            fault_parallel=np.asarray(components.fault_parallel, dtype=np.float64),
            s_pick=float(s_pick),
            s_snr=float(s_snr),
            vs_est=vs_est,
        ),
        station_info,
    )


def _reference_time(loads: list[LoadedStation]) -> np.ndarray | None:
    for item in loads:
        if item.time is not None and item.fault_parallel is not None:
            return np.asarray(item.time, dtype=float)
    return None


def _sort_indices(rows: list[InputRow], arrdist_by_station: dict[str, float]) -> np.ndarray:
    def sort_key(index: int) -> tuple[bool, float, str, int]:
        station = rows[index].station_name
        distance = arrdist_by_station.get(station, float("nan"))
        return (not np.isfinite(distance), float(distance), station, index)

    return np.array(sorted(range(len(rows)), key=sort_key), dtype=int)


def _aligned_fault_parallel(item: LoadedStation, reference_time: np.ndarray) -> np.ndarray:
    if item.time is None or item.fault_parallel is None:
        return np.full(reference_time.shape, np.nan, dtype=np.float64)

    data = np.asarray(item.fault_parallel, dtype=np.float64)
    time = np.asarray(item.time, dtype=float)
    if len(time) == len(reference_time) and np.allclose(
        time,
        reference_time,
        rtol=1e-5,
        atol=max(1e-8, 0.01 * float(np.median(np.diff(reference_time)))),
    ):
        return data
    return np.asarray(np.interp(reference_time, time, data), dtype=np.float64)


def _build_payload(
    event_rows: list[InputRow],
    loads_by_station: dict[str, LoadedStation],
    arrdist_by_station: dict[str, float]
) -> dict[str, object] | None:
    loads = [loads_by_station[row.station_name] for row in event_rows]
    reference_time = _reference_time(loads)
    if reference_time is None:
        return None

    indsort = _sort_indices(event_rows, arrdist_by_station)
    sorted_rows = [event_rows[int(index)] for index in indsort]
    sorted_loads = [loads_by_station[row.station_name] for row in sorted_rows]

    data_fpara = [_aligned_fault_parallel(item, reference_time) for item in sorted_loads]
    stalst = np.array([row.station_name for row in sorted_rows])
    arrdist = np.array(
        [arrdist_by_station.get(row.station_name, float("nan")) for row in sorted_rows],
        dtype=float,
    )
    spicks_raw = np.array([item.s_pick for item in sorted_loads], dtype=float)
    spicks = signal.medfilt(spicks_raw, kernel_size=5)
    pick_info = np.array(
        [[item.vs_est, item.s_snr] for item in sorted_loads],
        dtype=float,
    )

    return {
        "data_fpara": data_fpara,
        "stalst": stalst,
        "arrdist": arrdist,
        "indsort": indsort,
        "Spicks": spicks,
        "Pick_info": pick_info,
        "time_axis": reference_time,
    }


def run_step1_load_sac(
    input_file: str | Path,
    output: str | Path = ".",
    *,
    fault_angle: float = 300.0,
    time_window: tuple[float, float] = (-5.0, 55.0)
) -> dict[str, object]:
    """Run Step 1 and write one fault-parallel pickle per readable event."""
    time_start, time_end = time_window
    if time_end <= time_start:
        raise ValueError("--time-window END must be greater than START")

    _require_obspy()
    rows = parse_input_file(input_file)
    grouped = _group_rows_by_event(rows)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_infos = _collect_station_infos(
        rows,
        fault_angle=fault_angle,
        time_start=time_start,
        time_end=time_end,
    )
    if len(station_infos) < 2:
        raise ValueError("At least two stations with readable coordinates are required")
    station_order, arrdist, *_ = compute_array_geometry(station_infos)
    arrdist_by_station = {
        station: float(distance) for station, distance in zip(station_order, arrdist)
    }

    written_paths: list[Path] = []
    skipped_events = 0
    for event_id, event_rows in grouped.items():
        loads_by_station: dict[str, LoadedStation] = {}
        for row in event_rows:
            item, station_info = _load_station(
                row,
                fault_angle=fault_angle,
                time_start=time_start,
                time_end=time_end,
            )
            loads_by_station[row.station_name] = item
            if station_info is not None and row.station_name not in station_infos:
                station_infos[row.station_name] = station_info

        payload = _build_payload(
            event_rows,
            loads_by_station,
            arrdist_by_station        )
        if payload is None:
            skipped_events += 1
            continue

        path = output_dir / f"{event_id}_fpara.pkl"
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        written_paths.append(path)

    return {
        "output": output_dir,
        "pickle_files": written_paths,
        "event_count": len(grouped),
        "written_event_count": len(written_paths),
        "skipped_event_count": skipped_events,
    }


def main() -> None:
    """Parse CLI options, run Step 1, and print generated outputs."""
    parser = argparse.ArgumentParser(
        description="Load paired N/E SAC files, rotate to fault-parallel, and save event pickles"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="input_file.txt",
        help="Input txt: evid station north_sac east_sac (default: input_file.txt)",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default=".",
        help="Output directory for *_fpara.pkl files (default: current directory)",
    )
    parser.add_argument(
        "--fault-angle",
        type=float,
        default=300.0,
        help="Fault-parallel rotation angle in degrees (default: 300)",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=(-5.0, 55.0),
        help="Trace time window relative to origin time in seconds (default: -5 55)",
    )
    args = parser.parse_args()

    result = run_step1_load_sac(
        args.input_file,
        args.output,
        fault_angle=args.fault_angle,
        time_window=(float(args.time_window[0]), float(args.time_window[1])),
    )
    print("Step1 load SAC workflow done:")
    for key, value in result.items():
        if key == "pickle_files":
            print(f"  {key}: {len(value)} files")
        elif value is not None:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
