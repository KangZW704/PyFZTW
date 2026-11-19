"""Forward modelling contracts shared by all synthetic waveform solvers.

Inversion code depends only on this interface: a forward model receives a
project, one observed seismogram context, and resolved model parameters, then
returns a ``SyntheticTrace`` with time and displacement arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from ..io.earthquake import Earthquake
from ..io.project import Project
from ..io.seismogram import Seismogram


@dataclass
class SyntheticTrace:
    """Synthetic displacement trace produced by a forward model."""
    time: np.ndarray
    disp: np.ndarray

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.disp = np.asarray(self.disp, dtype=float)

        if self.time.ndim != 1 or self.disp.ndim != 1:
            raise ValueError("SyntheticTrace time and disp must be 1D arrays")
        if len(self.time) != len(self.disp):
            raise ValueError("SyntheticTrace time and disp length mismatch")
        if len(self.time) == 0:
            raise ValueError("SyntheticTrace is empty")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("SyntheticTrace time must be strictly increasing")


class ForwardModel(Protocol):
    """Protocol implemented by concrete waveform synthesis backends."""

    def run(
        self,
        project: Project,
        earthquake: Earthquake,
        seismogram: Seismogram,
        parameters: dict[str, float],
    ) -> SyntheticTrace:
        ...
