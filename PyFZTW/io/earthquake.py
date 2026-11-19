"""Read earthquake source parameter files.

Each earthquake block becomes an ``Earthquake`` object with its own parameter
space. The inversion joins these event-specific parameters with the shared
crust parameters when fitting multiple events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .crust import clean_lines
from .parameters import ParameterSpace, parse_parameter_line


@dataclass
class Earthquake:
    """Earthquake source definition and its event-local search space."""
    event_id: str
    parameter_space: ParameterSpace


def _check_earthquake_parameters(space: ParameterSpace) -> None:
    required = {"x", "y", "z"}
    names = {p.name for p in space.parameters}
    missing = required - names
    if missing:
        raise ValueError(f"Missing earthquake parameters: {sorted(missing)}")


def parse_earthquake_file(path: str | Path) -> list[Earthquake]:
    """Parse all earthquake blocks and return validated event definitions."""
    path = Path(path)
    lines = clean_lines(path.read_text(encoding="utf-8").splitlines())

    earthquakes: list[Earthquake] = []
    seen_ids: set[str] = set()

    i = 0
    while i < len(lines):
        event_id = lines[i]
        i += 1

        if event_id in seen_ids:
            raise ValueError(f"Duplicate earthquake id: {event_id}")
        seen_ids.add(event_id)

        params = []
        seen_names: set[str] = set()

        while i < len(lines):
            first = lines[i].split()[0]
            if first.isdigit():
                break

            param = parse_parameter_line(lines[i])
            if param.name in seen_names:
                raise ValueError(f"Duplicate earthquake parameter '{param.name}' in event {event_id}")
            seen_names.add(param.name)
            params.append(param)
            i += 1

        space = ParameterSpace(params)
        _check_earthquake_parameters(space)

        earthquakes.append(
            Earthquake(
                event_id=event_id,
                parameter_space=space,
            )
        )

    return earthquakes
