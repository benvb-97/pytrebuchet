"""Test cases for the environment module."""

from environment import EnvironmentConfig


def test_environment_initialization() -> None:
    """Test environment initialization."""
    _ = EnvironmentConfig()  # default initialization

    environment = EnvironmentConfig(
        wind_speed=3.0,
        air_density=1.2,
        air_kinematic_viscosity=1.6e-5,
        gravitational_acceleration=9.8,
    )
    assert environment.wind_speed == 3.0
    assert environment.air_density == 1.2
    assert environment.air_kinematic_viscosity == 1.6e-5
    assert environment.gravitational_acceleration == 9.8
