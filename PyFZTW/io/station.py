"""Read station geometry files used by the forward solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .crust import clean_lines


@dataclass
class Station:
    """Station name and coordinates referenced by seismogram headers."""
    name: str
    x: float
    y: float
    z: float


def parse_station_file(path: str | Path) -> list[Station]:
    """Parse station coordinates and return a duplicate-checked station list."""
    path = Path(path)
    lines = clean_lines(path.read_text(encoding="utf-8").splitlines())

    stations: list[Station] = []
    seen: set[str] = set()

    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Invalid station line: {line}")

        name = parts[0]
        if name in seen:
            raise ValueError(f"Duplicate station name: {name}")
        seen.add(name)

        stations.append(
            Station(
                name=name,
                x=float(parts[1]),
                y=float(parts[2]),
                z=float(parts[3]),
            )
        )

    return stations
