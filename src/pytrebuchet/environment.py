"""Module for defining environment configuration parameters for simulations."""

from dataclasses import dataclass

from pytrebuchet.physical_constants import PhysicalConstants as PhCo


@dataclass
class EnvironmentConfig:
    """Class for holding environment configuration parameters."""

    # wind speed in m/s
    wind_speed: float = 0.0

    # air density in kg/m^3
    air_density: float = PhCo.AIR_DENSITY

    # air kinematic viscosity in m^2/s
    air_kinematic_viscosity: float = PhCo.AIR_KINEMATIC_VISCOSITY

    # gravitational acceleration in m/s^2
    gravitational_acceleration: float = PhCo.GRAVITATIONAL_ACCELERATION_EARTH
