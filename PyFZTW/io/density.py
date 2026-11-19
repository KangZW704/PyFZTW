"""Density helpers shared by inversion parameter resolution and forward models."""

from __future__ import annotations


def density_from_vs(vs: float) -> float:
    """Estimate density from Vs using the CVM-S4.26 polynomial."""
    return (
        1.29483185483
        + 1.25507583371 * vs
        - 0.51231936640 * vs**2
        + 0.05196239948 * vs**3
        + 0.01560098789 * vs**4
        - 0.00241896593 * vs**5
    )
