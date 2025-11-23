"""Test cases for the simulation module."""

import numpy as np
import pytest

from pytrebuchet import EnvironmentConfig, Projectile, Simulation, Trebuchet
from pytrebuchet.simulation import SimulationPhases


@pytest.fixture(scope="module", params=[True, False])
def hcw_trebuchet(
    *,
    projectile_touch_ground: bool = True,
) -> Trebuchet:
    """Initialize a hinged counterweight Trebuchet instance.

    :param projectile_touch_ground: If True, sets the parameters so that
      the projectile arm touches the ground.
    :return: Trebuchet instance
    """
    h_pivot = 5.0 if projectile_touch_ground else 15.0

    return Trebuchet(
        l_weight_arm=1.75,
        l_projectile_arm=6.8,
        l_sling_projectile=6.83,
        l_sling_weight=2.0,
        h_pivot=h_pivot,
        mass_arm=10.7,
        inertia_arm=65.0,
        d_pivot_to_arm_cog=2.52,
        mass_weight=100.0,
        release_angle=45.0 * np.pi / 180.0,
    )


@pytest.fixture(scope="module")
def whipper_trebuchet() -> Trebuchet:
    """Initialize a whipper Trebuchet instance."""
    return Trebuchet.default_whipper()


@pytest.fixture(scope="module")
def environment() -> EnvironmentConfig:
    """Initialize a default environment configuration."""
    return EnvironmentConfig()


def test_initialization_hcw(
    hcw_trebuchet: Trebuchet,
    environment: EnvironmentConfig,
) -> None:
    """Test simulation initialization."""
    projectile = Projectile(mass=4.0, diameter=0.35)
    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        projectile=projectile,
        environment=environment,
    )

    assert simulation.trebuchet == hcw_trebuchet
    assert simulation.projectile == projectile
    assert simulation.environment == environment

    assert simulation.phases == (
        SimulationPhases.GROUND_SLIDING,
        SimulationPhases.SLING_UNCONSTRAINED,
        SimulationPhases.BALLISTIC,
    )


def test_initialization_whipper(
    whipper_trebuchet: Trebuchet, environment: EnvironmentConfig
) -> None:
    """Test simulation initialization with whipper trebuchet."""
    projectile = Projectile.default()
    simulation = Simulation(
        trebuchet=whipper_trebuchet,
        projectile=projectile,
        environment=environment,
    )

    assert simulation.trebuchet == whipper_trebuchet
    assert simulation.projectile == projectile
    assert simulation.environment == environment

    assert simulation.phases == (
        SimulationPhases.WHIPPER_BOTH_CONSTRAINED,
        SimulationPhases.WHIPPER_PROJECTILE_CONSTRAINED,
        SimulationPhases.SLING_UNCONSTRAINED,
        SimulationPhases.BALLISTIC,
    )


def test_hcw_simulation(
    hcw_trebuchet: Trebuchet, environment: EnvironmentConfig
) -> None:
    """Test simulation with a hinged counterweight trebuchet."""
    projectile = Projectile(mass=4.0, diameter=0.35)
    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        projectile=projectile,
        environment=environment,
    )

    simulation.solve()

    assert simulation.get_phase_end_time(
        phase=SimulationPhases.GROUND_SLIDING
    ) == pytest.approx(0.6751505542485441, abs=1e-6)
    assert simulation.get_phase_end_time(
        phase=SimulationPhases.SLING_UNCONSTRAINED
    ) == pytest.approx(1.6497669591178958, abs=1e-6)
    assert simulation.get_phase_end_time(
        phase=SimulationPhases.BALLISTIC
    ) == pytest.approx(6.002567303576664, abs=1e-6)
    assert simulation.distance_traveled == pytest.approx(65.81262624344718, rel=1e-6)


def test_unsolved_simulation_raises_error(
    whipper_trebuchet: Trebuchet, environment: EnvironmentConfig
) -> None:
    """Test that accessing results before solving raises ValueError."""
    projectile = Projectile.default()
    simulation = Simulation(
        trebuchet=whipper_trebuchet,
        projectile=projectile,
        environment=environment,
    )

    # Test that accessing methods requiring solved simulation raises ValueError
    with pytest.raises(ValueError, match="Simulation has not been run yet"):
        simulation.get_tsteps()

    with pytest.raises(ValueError, match="Simulation has not been run yet"):
        simulation.get_trebuchet_state_variables()

    with pytest.raises(ValueError, match="Simulation has not been run yet"):
        simulation.get_projectile_state_variables()

    with pytest.raises(ValueError, match="Simulation has not been run yet"):
        simulation.where_sling_in_tension()


def test_sling_tension_verification(
    hcw_trebuchet: Trebuchet, environment: EnvironmentConfig
) -> None:
    """Test that sling tension verification works correctly."""
    projectile = Projectile.default()
    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        projectile=projectile,
        environment=environment,
        verify_sling_tension=False,  # will verify manually
    )

    simulation.solve()

    tension_array = simulation.where_sling_in_tension()

    # Check that the tension array is boolean,
    # has correct length and all values are True (sling in tension)
    assert isinstance(tension_array, np.ndarray)
    assert tension_array.dtype == bool
    assert len(tension_array) == len(
        simulation.get_tsteps(phase=SimulationPhases.TREBUCHET)
    )
    assert np.all(tension_array)

    # Now test a case where the sling goes slack
    total_arm_length = hcw_trebuchet.l_weight_arm + hcw_trebuchet.l_projectile_arm
    arm_fraction = 0.65
    hcw_trebuchet.l_projectile_arm = total_arm_length * arm_fraction
    hcw_trebuchet.l_weight_arm = total_arm_length * (1 - arm_fraction)

    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        projectile=projectile,
        environment=environment,
        verify_sling_tension=False,
    )  # will verify manually
    simulation.solve()
    tension_array = simulation.where_sling_in_tension()

    assert isinstance(tension_array, np.ndarray)
    assert tension_array.dtype == bool
    assert len(tension_array) == len(
        simulation.get_tsteps(phase=SimulationPhases.TREBUCHET)
    )
    assert not np.all(tension_array)


def test_whipper_simulation(
    whipper_trebuchet: Trebuchet, environment: EnvironmentConfig
) -> None:
    """Test simulation with a whipper-style trebuchet."""
    projectile = Projectile.default()
    simulation = Simulation(
        trebuchet=whipper_trebuchet, projectile=projectile, environment=environment
    )

    simulation.solve()

    # Attempt to access launch distance and times
    # to ensure simulation ran without errors
    _ = simulation.distance_traveled
    _ = simulation.get_phase_end_time(phase=SimulationPhases.BALLISTIC)
