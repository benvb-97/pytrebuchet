"""Module for defining environment configuration parameters for simulations."""

from dataclasses import dataclass

from pytrebuchet.constants import (
    AIR_DENSITY,
    AIR_KINEMATIC_VISCOSITY,
    GRAVITATIONAL_ACCELERATION_EARTH,
)


@dataclass
class EnvironmentConfig:
    """Class for holding environment configuration parameters."""

    # wind speed in m/s
    wind_speed: float = 0.0

    # air density in kg/m^3
    air_density: float = AIR_DENSITY

    # air kinematic viscosity in m^2/s
    air_kinematic_viscosity: float = AIR_KINEMATIC_VISCOSITY

    # gravitational acceleration in m/s^2
    gravitational_acceleration: float = GRAVITATIONAL_ACCELERATION_EARTH
