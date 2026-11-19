"""HS4PDIS-backed forward model for synthetic displacement traces.

This module adapts PyFZTW project data and resolved inversion parameters to the
compiled Fortran ``hs4pdis`` routine. The public model returns a
``SyntheticTrace`` that can be aligned and scored against one observed
seismogram window.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import numpy as np

from .base import ForwardModel, SyntheticTrace
from ..io.density import density_from_vs
from ..io.project import Project
from ..io.seismogram import Seismogram


@dataclass
class HS4PDISForwardConfig:
    """Numerical defaults and geometry safeguards for HS4PDIS calls."""
    npts: int = 4096
    z_rec_default: float = 0.0
    lt_padding: float = 1.0
    delv: float = 1e-5
    min_xr_xs_gap: float = 0.005


class HS4PDISForwardModel(ForwardModel):
    """Generate synthetic traces through the compiled ``_hs4pdis`` backend."""

    def __init__(
        self,
        config: HS4PDISForwardConfig | None = None,
    ) -> None:
        try:
            from . import _hs4pdis
        except ImportError:
            raise ImportError(
                "_hs4pdis not compiled. Run:\n"
                "  pip install meson ninja\n"
                "  cd PyFZTW/forward\n"
                "  python -m numpy.f2py -c hs4pdis.f90 -m _hs4pdis --backend meson"
            ) from None
        self.backend = _hs4pdis
        self.config = config or HS4PDISForwardConfig()

    def __getstate__(self) -> dict:
        """Drop the compiled backend handle so the model can be pickled."""
        state = self.__dict__.copy()
        state["backend"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the compiled backend after process-based deserialization."""
        self.__dict__.update(state)
        from . import _hs4pdis
        self.backend = _hs4pdis

    def run(
        self,
        project: Project,
        seismogram: Seismogram,
        parameters: dict[str, float],
    ) -> SyntheticTrace:
        """Run HS4PDIS for one seismogram and return its synthetic trace."""
        cfg = self.config

        lt = seismogram.header.lt if seismogram.header.lt is not None else seismogram.header.window_end + cfg.lt_padding
        ifm = int(round(seismogram.header.maxfreq))
        gh = float(seismogram.header.gibbs)
        acc = float(seismogram.header.accuracy)

        v1 = float(parameters["v1"])
        vfr = float(parameters["vfr"])
        v3r = float(parameters.get("v3r", 1.0))
        q1 = float(parameters["q1"])
        qf = float(parameters["qf"])
        q3_val = float(parameters["q3"])
        widthf = float(parameters["widthf"])
        edge = float(parameters["edge"])

        x = float(parameters["x"])
        y = float(parameters["y"])
        z = float(parameters["z"])

        station = self._find_station(project, seismogram.header.station)
        xr_abs = float(station.x)
        yr_abs = float(station.y)
        zr_abs = float(station.z)

        # Convert absolute model geometry into the dimensionless values expected
        # by the legacy Fortran solver.
        if not np.isfinite(widthf) or widthf <= 0.0:
            raise ValueError("widthf must be a finite positive total waveguide width")
        if not np.isfinite(x) or not 0.0 <= x <= 1.0:
            raise ValueError("Source x must be a finite fraction of total width in [0, 1]")
        x_rel_source = x * widthf
        x_rel_station = xr_abs - edge

        # Media 2 and 3 are identical parts of one low-velocity waveguide.
        # Public widthf and source x remain defined using its TOTAL width.
        effective_widthf = 0.99 * widthf
        effective_widtht = 0.01 * widthf

        z0 = np.sqrt((yr_abs - y) ** 2 + (zr_abs - z) ** 2)

        rc1 = v1
        rc2 = vfr * rc1
        rc3 = rc2
        rc4 = v3r * rc1

        rou1 = float(parameters.get("rou1", self._density_from_vs(rc1)))
        rou2 = float(parameters.get("rou2", self._density_from_vs(rc2)))
        rou3 = rou2
        # Public rou3, v3r and q3 still describe the right host rock (medium 4).
        rou4 = float(parameters.get("rou3", self._density_from_vs(rc4)))

        q2 = qf
        q3 = q2
        q4 = q3_val

        z_rec = zr_abs if zr_abs is not None else cfg.z_rec_default
        normalized_gap = abs(x_rel_station / widthf - x)
        if normalized_gap <= cfg.min_xr_xs_gap:
            raise ValueError(
                f"Invalid geometry: |xr-xs|={normalized_gap:.6f} <= {cfg.min_xr_xs_gap:.6f}"
            )
        # The legacy kernel assumes a source in medium 2. For sources in the
        # rightmost 1%, reflect the horizontal geometry and exchange host rocks.
        # The internal interface is fictitious, so this preserves the physical
        # model while placing the source inside medium 2 in solver coordinates.
        reflected = x >= 0.99
        if reflected:
            x_rel_source = (1.0 - x) * widthf
            x_rel_station = widthf - x_rel_station
            rc1, rc4 = rc4, rc1
            rou1, rou4 = rou4, rou1
            q1, q4 = q4, q1
        xs = x_rel_source / effective_widthf
        xr = x_rel_station / effective_widthf
        rfz = z0 / max(effective_widthf, 1e-12)
        rtra = z0 / max(effective_widtht, 1e-12)
        # Keep the same numerical integration spacing after a reflection.
        repmac = np.pi * lt * v1 / max(z0, 1e-12)

        time = np.zeros(cfg.npts, dtype=np.float64)
        disp = np.zeros(cfg.npts, dtype=np.float64)

        self._call_backend(
            # The kernel's signed dislocation reverses under x reflection.
            delv=-cfg.delv if reflected else cfg.delv,
            lt=lt,
            ifm=ifm,
            gh=gh,
            rc1=rc1,
            rc2=rc2,
            rc3=rc3,
            rc4=rc4,
            rou1=rou1,
            rou2=rou2,
            rou3=rou3,
            rou4=rou4,
            q1=q1,
            q2=q2,
            q3=q3,
            q4=q4,
            z_rec=z_rec,
            z_source=z0,
            rfz=rfz,
            rtra=rtra,
            xr=xr,
            xs=xs,
            acc=acc,
            repmac=repmac,
            time=time,
            disp=disp,
        )

        return SyntheticTrace(time=time, disp=disp)

    def _call_backend(self, **kwargs) -> None:
        """Call the Fortran routine, which fills the provided time/disp arrays."""
        self.backend.hs4pdis(
            kwargs["delv"],
            kwargs["lt"],
            kwargs["ifm"],
            kwargs["gh"],
            kwargs["rc1"],
            kwargs["rc2"],
            kwargs["rc3"],
            kwargs["rc4"],
            kwargs["rou1"],
            kwargs["rou2"],
            kwargs["rou3"],
            kwargs["rou4"],
            kwargs["q1"],
            kwargs["q2"],
            kwargs["q3"],
            kwargs["q4"],
            kwargs["z_rec"],
            kwargs["z_source"],
            kwargs["rfz"],
            kwargs["rtra"],
            kwargs["xr"],
            kwargs["xs"],
            kwargs["acc"],
            kwargs["repmac"],
            kwargs["time"],
            kwargs["disp"],
        )

    @staticmethod
    def _density_from_vs(vs: float) -> float:
        """Estimate density from Vs using the CVM-S4.26 polynomial."""
        return density_from_vs(vs)

    @staticmethod
    def _find_station(project: Project, station_name: str):
        """Find the station referenced by a seismogram header."""
        for station in project.stations:
            if station.name == station_name:
                return station
        raise KeyError(f"Station not found: {station_name}")
