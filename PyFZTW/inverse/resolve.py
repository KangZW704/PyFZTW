"""Resolve fixed and free parameters into complete forward-model inputs.

Optimizers work with vectors of free parameters. This module combines those
candidate values with fixed crust and earthquake parameters, returning the flat
dictionary consumed by the HS4PDIS forward model.
"""

from __future__ import annotations

from ..io.crust import CrustConfig
from ..io.density import density_from_vs
from ..io.earthquake import Earthquake
from ..io.parameters import ParameterSpace
from ..io.project import Project


def resolve_parameter_space(
    space: ParameterSpace,
    variable_values: dict[str, float],
) -> dict[str, float]:
    """Resolve fixed constants and optimizer variables for one parameter space."""
    resolved: dict[str, float] = {}

    for param in space.parameters:
        if param.constant == 1:
            assert param.value is not None
            resolved[param.name] = param.value

        elif param.constant == 0:
            if param.name not in variable_values:
                raise ValueError(f"Missing variable parameter: {param.name}")
            resolved[param.name] = variable_values[param.name]

    return resolved


def resolve_crust_parameters(
    crust: CrustConfig,
    variable_values: dict[str, float],
) -> dict[str, float]:
    """Return resolved crust parameters for one optimizer candidate."""
    
    resolved = resolve_parameter_space(crust.parameter_space, variable_values)
    add_default_density_parameters(resolved)
    
    return resolved


def resolve_earthquake_parameters(
    earthquake: Earthquake,
    variable_values: dict[str, float],
) -> dict[str, float]:
    """Return resolved event parameters for one optimizer candidate."""
    
    resolved = resolve_parameter_space(earthquake.parameter_space, variable_values)

    return resolved


def resolve_model_parameters(
    project: Project,
    earthquake: Earthquake,
    crust_variable_values: dict[str, float],
    earthquake_variable_values: dict[str, float],
) -> dict[str, float]:
    """Merge resolved crust and event parameters for forward modelling."""
    crust_values = resolve_crust_parameters(project.crust, crust_variable_values)

    earthquake_values = resolve_earthquake_parameters(
        earthquake=earthquake,
        variable_values=earthquake_variable_values,
    )

    resolved = {
        **crust_values,
        **earthquake_values,
    }

    return resolved


def add_default_density_parameters(parameters: dict[str, float]) -> None:
    """Fill omitted rou1-rou3 values from the resolved layer velocities."""
    v1 = float(parameters["v1"])
    vfr = float(parameters["vfr"])
    v3r = float(parameters.get("v3r", 1.0))

    parameters.setdefault("rou1", density_from_vs(v1))
    parameters.setdefault("rou2", density_from_vs(vfr * v1))
    parameters.setdefault("rou3", density_from_vs(v3r * v1))
