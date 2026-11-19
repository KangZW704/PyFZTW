"""Project assembly and cross-file validation.

``load_project`` combines crust, earthquake, station, and seismogram inputs into
one validated object. The resulting ``Project`` is the main input passed to
forward modelling and inversion routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .crust import CrustConfig, parse_crust_file
from .earthquake import Earthquake, parse_earthquake_file
from .station import Station, parse_station_file
from .seismogram import Seismogram, parse_seismogram_files, seismogram_key


@dataclass
class Project:
    """All parsed inputs needed to run one PyFZTW inversion project."""
    crust: CrustConfig
    earthquakes: list[Earthquake]
    stations: list[Station]
    seismograms: list[Seismogram]

    def earthquake_index(self) -> dict[str, Earthquake]:
        result: dict[str, Earthquake] = {}
        for eq in self.earthquakes:
            if eq.event_id in result:
                raise ValueError(f"Duplicate earthquake id: {eq.event_id}")
            result[eq.event_id] = eq
        return result

    def station_index(self) -> dict[str, Station]:
        result: dict[str, Station] = {}
        for st in self.stations:
            if st.name in result:
                raise ValueError(f"Duplicate station name: {st.name}")
            result[st.name] = st
        return result


def validate_project(project: Project) -> None:
    """Check that every seismogram references known events and stations."""
    eq_index = project.earthquake_index()
    st_index = project.station_index()

    seen_keys: set[tuple[str, str]] = set()

    for gram in project.seismograms:
        eq_id = gram.header.earthquake_id
        st_name = gram.header.station
        key = seismogram_key(gram)

        if eq_id not in eq_index:
            raise ValueError(f"Seismogram references missing earthquake id: {eq_id}")

        if st_name not in st_index:
            raise ValueError(f"Seismogram references missing station: {st_name}")

        if key in seen_keys:
            raise ValueError(f"Duplicate seismogram (event_id, station): {key}")

        seen_keys.add(key)


def check_all_earthquakes_used(project: Project) -> None:
    """Require every configured earthquake to have at least one seismogram."""
    used_ids = {gram.header.earthquake_id for gram in project.seismograms}
    all_ids = {eq.event_id for eq in project.earthquakes}

    missing = all_ids - used_ids
    if missing:
        raise ValueError(f"These earthquakes have no associated seismogram: {sorted(missing)}")


def load_project(
    crust_file: str | Path,
    earthquake_file: str | Path,
    station_file: str | Path,
    seismogram_files: list[str | Path],
) -> Project:
    """Load all input files and return a cross-validated project object."""
    crust = parse_crust_file(crust_file)
    earthquakes = parse_earthquake_file(earthquake_file)
    stations = parse_station_file(station_file)
    seismograms = parse_seismogram_files(seismogram_files)

    project = Project(
        crust=crust,
        earthquakes=earthquakes,
        stations=stations,
        seismograms=seismograms,
    )

    validate_project(project)
    check_all_earthquakes_used(project)

    return project
