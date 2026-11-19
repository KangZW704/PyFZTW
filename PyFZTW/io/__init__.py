"""Input parsers and project containers for PyFZTW.

The IO package reads crust, earthquake, station, and seismogram text files into
validated Python objects that are consumed by forward modelling and inversion.
"""

from .parameters import ParameterSpec, ParameterSpace, parse_parameter_line
from .crust import CrustConfig, clean_lines, parse_crust_file
from .earthquake import Earthquake, parse_earthquake_file
from .station import Station, parse_station_file
from .seismogram import (
    SeismogramHeader,
    Seismogram,
    parse_seismogram_file,
    parse_seismogram_files,
    seismogram_key,
)
from .project import Project, load_project, validate_project, check_all_earthquakes_used

__all__ = [
    "ParameterSpec",
    "ParameterSpace",
    "parse_parameter_line",
    "CrustConfig",
    "clean_lines",
    "parse_crust_file",
    "Earthquake",
    "parse_earthquake_file",
    "Station",
    "parse_station_file",
    "SeismogramHeader",
    "Seismogram",
    "parse_seismogram_file",
    "parse_seismogram_files",
    "seismogram_key",
    "Project",
    "load_project",
    "validate_project",
    "check_all_earthquakes_used",
]
