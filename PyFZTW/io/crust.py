"""Read and validate crust parameter files.

The parser returns a ``CrustConfig`` containing the crust parameter space used
by the inversion to build complete HS4PDIS model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .parameters import ParameterSpace, parse_parameter_line


@dataclass
class CrustConfig:
    """Container for crust parameters shared by all seismograms in a project."""
    parameter_space: ParameterSpace


def clean_lines(lines: list[str]) -> list[str]:
    """Remove comments and blank lines from simple PyFZTW text inputs."""
    cleaned: list[str] = []

    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cleaned.append(line)

    return cleaned


def check_crust_parameters(space: ParameterSpace) -> None:
    """Ensure the crust file provides all parameters required by HS4PDIS."""
    required = {
        "v1", "vfr", "v3r",
        "q1", "qf", "q3",
        "widthf",
        'edge',
    }

    names = {p.name for p in space.parameters}
    missing = required - names
    if missing:
        raise ValueError(f"Missing crust parameters: {sorted(missing)}")

    unsupported = names & {"density", "density1", "density2", "density3", "density4", "rou4"}
    if unsupported:
        raise ValueError(
            f"Unsupported crust parameters: {sorted(unsupported)}. "
            "Use optional rou1, rou2, rou3; omitted rou values are estimated from Vs."
        )


def parse_crust_file(path: str | Path) -> CrustConfig:
    """Parse a crust file and return a validated crust configuration."""
    path = Path(path)
    lines = clean_lines(path.read_text(encoding="utf-8").splitlines())

    parameters = []
    seen: set[str] = set()

    for line in lines:
        param = parse_parameter_line(line)

        if param.name in seen:
            raise ValueError(f"Duplicate crust parameter '{param.name}'")
        seen.add(param.name)
        parameters.append(param)

    space = ParameterSpace(parameters)
    check_crust_parameters(space)

    return CrustConfig(
        parameter_space=space,
    )
