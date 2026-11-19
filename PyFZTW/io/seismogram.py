"""Read observed seismogram files and expose correlation windows.

The parser returns ``Seismogram`` objects containing header metadata plus time
and displacement arrays. Inversion code uses ``windowed_data`` as the observed
target for synthetic-trace alignment and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .crust import clean_lines


REQUIRED_SEIS_HEADERS = {"eq", "station", "window", "maxfreq", "accuracy", "gibbs"}
OPTIONAL_SEIS_HEADERS = {"lt"}


@dataclass
class SeismogramHeader:
    """Metadata that links a waveform to an event, station, and fit window."""
    earthquake_id: str
    station: str
    window_start: float
    window_end: float
    maxfreq: float
    accuracy: float
    gibbs: float
    lt: float | None = None


@dataclass
class Seismogram:
    """Observed waveform data with validation and window extraction helpers."""
    header: SeismogramHeader
    time: np.ndarray
    disp: np.ndarray
    source_file: str | None = None

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.disp = np.asarray(self.disp, dtype=float)
        self.validate()

    def validate(self) -> None:
        if self.time.ndim != 1 or self.disp.ndim != 1:
            raise ValueError("time and disp must be 1D arrays")
        if len(self.time) != len(self.disp):
            raise ValueError("time and disp length mismatch")
        if len(self.time) == 0:
            raise ValueError("seismogram data is empty")
        if self.header.window_start >= self.header.window_end:
            raise ValueError("window_start must be smaller than window_end")
        dt = np.median(np.diff(self.time))
        if self.time[0] - self.header.window_start > dt or self.header.window_end - self.time[-1] > dt:
            raise ValueError("correlation window is outside waveform time range")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be strictly increasing")

    @property
    def event_id(self) -> str:
        return self.header.earthquake_id

    @property
    def station(self) -> str:
        return self.header.station

    def window_mask(self) -> np.ndarray:
        return (self.time >= self.header.window_start) & (self.time <= self.header.window_end)

    def windowed_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the time and displacement samples inside the fit window."""
        mask = self.window_mask()
        return self.time[mask], self.disp[mask]


def parse_seismogram_file(path: str | Path) -> Seismogram:
    """Parse one observed waveform file into a validated ``Seismogram``."""
    path = Path(path)
    lines = clean_lines(path.read_text(encoding="utf-8").splitlines())

    header: dict[str, list[str]] = {}
    optional_header: dict[str, list[str]] = {}
    data: list[tuple[float, float]] = []

    for line in lines:
        parts = line.split()
        key = parts[0].lower()

        if key in REQUIRED_SEIS_HEADERS:
            if key in header:
                raise ValueError(f"Duplicate seismogram header field: {key}")
            header[key] = parts[1:]
            continue

        if key in OPTIONAL_SEIS_HEADERS:
            if key in optional_header:
                raise ValueError(f"Duplicate seismogram header field: {key}")
            optional_header[key] = parts[1:]
            continue

        if len(parts) != 2:
            raise ValueError(f"Invalid seismogram data line: {line}")

        data.append((float(parts[0]), float(parts[1])))

    missing = REQUIRED_SEIS_HEADERS - set(header)
    if missing:
        raise ValueError(f"Missing seismogram header fields: {sorted(missing)}")

    if not data:
        raise ValueError("Seismogram file contains no waveform data")

    time = np.array([x[0] for x in data], dtype=float)
    disp = np.array([x[1] for x in data], dtype=float)

    window = header["window"]
    if len(window) != 2:
        raise ValueError("window must have two values: start end")
    if len(header["eq"]) != 1:
        raise ValueError("eq must have one value")
    if len(header["station"]) != 1:
        raise ValueError("station must have one value")
    if len(header["maxfreq"]) != 1:
        raise ValueError("maxfreq must have one value")
    if len(header["accuracy"]) != 1:
        raise ValueError("accuracy must have one value")
    if len(header["gibbs"]) != 1:
        raise ValueError("gibbs must have one value")

    lt_val = float(optional_header["lt"][0]) if "lt" in optional_header else None
    return Seismogram(
        header=SeismogramHeader(
            earthquake_id=header["eq"][0],
            station=header["station"][0],
            window_start=float(window[0]),
            window_end=float(window[1]),
            maxfreq=float(header["maxfreq"][0]),
            accuracy=float(header["accuracy"][0]),
            gibbs=float(header["gibbs"][0]),
            lt=lt_val,
        ),
        time=time,
        disp=disp,
        source_file=path.name,
    )


def parse_seismogram_files(paths: list[str | Path]) -> list[Seismogram]:
    """Parse multiple waveform files in the order provided by the caller."""
    return [parse_seismogram_file(path) for path in paths]


def seismogram_key(gram: Seismogram) -> tuple[str, str]:
    """Return the event/station key used to detect duplicate observations."""
    return gram.header.earthquake_id, gram.header.station
