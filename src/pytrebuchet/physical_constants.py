"""Module defining physical constants for simulations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """Class for holding physical constants used in simulations."""

    # Standard air density at sea level in kg/m^3
    AIR_DENSITY: float = 1.225  # kg/m^3

    # Approximate air kinematic viscosity at 15 degrees Celsius at sea level in m^2/s
    AIR_KINEMATIC_VISCOSITY: float = 1.47e-5  # m^2/s

    # Gravitational acceleration on Earth in m/s^2
    GRAVITATIONAL_ACCELERATION_EARTH: float = 9.81  # m/s^2
