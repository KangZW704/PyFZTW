"""Parameter definitions shared by input parsers and optimizers.

The module converts compact text-file parameter lines into validated parameter
spaces. Optimizers consume the free-parameter bounds and later map candidate
vectors back to named model values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParameterSpec:
    """One model parameter parsed from an input file.

    Fixed parameters carry a single value; variable parameters carry search
    bounds and become optimizer dimensions.
    """
    name: str
    constant: int
    value: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        self.name = self.name.lower()
        self.validate()

    @property
    def is_variable(self) -> bool:
        return self.constant == 0

    @property
    def is_fixed(self) -> bool:
        return self.constant == 1


    def validate(self) -> None:
        if self.constant == 0:
            if self.minimum is None or self.maximum is None:
                raise ValueError(f"Variable parameter '{self.name}' missing bounds")
            if self.minimum > self.maximum:
                raise ValueError(f"Variable parameter '{self.name}' has minimum > maximum")

        elif self.constant == 1:
            if self.value is None:
                raise ValueError(f"Fixed parameter '{self.name}' missing value")

        else:
            raise ValueError(f"Parameter '{self.name}' has invalid constant flag {self.constant}")


@dataclass
class ParameterSpace:
    """A validated collection of fixed and free model parameters.

    The free parameters define the optimizer search space; fixed parameters are
    resolved directly into the model dictionary used by the forward solver.
    """
    parameters: list[ParameterSpec]

    def free_parameters(self) -> list[ParameterSpec]:
        result: list[ParameterSpec] = []
        for p in self.parameters:
            if p.constant == 0:
                result.append(p)
        return result

    def fixed_parameters(self) -> list[ParameterSpec]:
        result: list[ParameterSpec] = []
        for p in self.parameters:
            if p.constant == 1:
                result.append(p)
        return result


    def variable_names(self) -> list[str]:
        return [p.name for p in self.free_parameters()]

    def bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        for p in self.free_parameters():
            if p.minimum is None or p.maximum is None:
                raise ValueError(f"Parameter '{p.name}' missing bounds")
            bounds.append((p.minimum, p.maximum))
        return bounds

    def vector_to_parameters(self, vector: list[float]) -> dict[str, float]:
        """Map an optimizer vector back to named free-parameter values."""
        free = self.free_parameters()

        if len(vector) != len(free):
            raise ValueError("Parameter vector length mismatch")

        result: dict[str, float] = {}
        for spec, value in zip(free, vector):
            result[spec.name] = value
        return result

    def get(self, name: str) -> ParameterSpec:
        name = name.lower()
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"Parameter '{name}' not found")


def parse_parameter_line(line: str) -> ParameterSpec:
    """Parse one parameter line from a crust or earthquake file.

    Supported forms are ``name 1 value`` for fixed parameters and
    ``name 0 min max`` for free parameters.
    """
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid parameter line: {line}")

    name = parts[0].lower()
    constant = int(parts[1])

    if constant == 1:
        if len(parts) != 3:
            raise ValueError(f"Fixed parameter must have 3 fields: {line}")
        return ParameterSpec(
            name=name,
            constant=constant,
            value=float(parts[2]),
        )

    if constant == 0:
        if len(parts) == 4:
            minimum = float(parts[2])
            maximum = float(parts[3])
        else:
            raise ValueError(f"Variable parameter must have 4 fields: {line}")
        return ParameterSpec(
            name=name,
            constant=constant,
            minimum=minimum,
            maximum=maximum,
        )

    raise ValueError(f"Unknown constant flag {constant} in line: {line}")
