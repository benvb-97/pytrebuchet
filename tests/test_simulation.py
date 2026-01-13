"""Test cases for the simulation module."""

import numpy as np
import pytest

from pytrebuchet.differential_equations.sling_phase import SlingPhases
from pytrebuchet.environment import EnvironmentConfig
from pytrebuchet.projectile import Projectile
from pytrebuchet.simulation import Simulation, SimulationPhases
from pytrebuchet.trebuchet import (
    Arm,
    HingedCounterweightTrebuchet,
    Pivot,
    Sling,
    Weight,
    WhipperTrebuchet,
)


@pytest.fixture(scope="module")
def projectile() -> Projectile:
    """Initialize a default projectile instance."""
    return Projectile(mass=4.0, diameter=0.35)


@pytest.fixture(scope="module", params=[True, False])
def hcw_trebuchet(
    projectile: Projectile,
    *,
    projectile_touch_ground: bool = True,
) -> HingedCounterweightTrebuchet:
    """Initialize a hinged counterweight Trebuchet instance.

    :param projectile_touch_ground: If True, sets the parameters so that
      the projectile arm touches the ground.
    :return: Trebuchet instance
    """
    h_pivot = 5.0 if projectile_touch_ground else 15.0

    return HingedCounterweightTrebuchet(
        arm=Arm(
            length_weight_side=1.75,
            length_projectile_side=6.8,
            mass=10.7,
            inertia=65.0,
            d_pivot_to_cog=2.52,
        ),
        weight=Weight(mass=100.0),
        sling_projectile=Sling(length=6.83),
        sling_weight=Sling(length=2.0),
        pivot=Pivot(height=h_pivot),
        release_angle=45.0 * np.pi / 180.0,
        projectile=projectile,
    )


@pytest.fixture(scope="module")
def whipper_trebuchet(projectile: Projectile) -> WhipperTrebuchet:
    """Initialize a whipper Trebuchet instance."""
    return WhipperTrebuchet(
        arm=Arm(
            length_weight_side=2.0,
            length_projectile_side=4.0,
            mass=10.0,
            inertia=None,
            d_pivot_to_cog=None,
        ),
        weight=Weight(mass=100.0),
        pivot=Pivot(height=5.0),
        projectile=projectile,
        sling_projectile=Sling(length=4.0),
        sling_weight=Sling(length=4.0),
        arm_angle=60.0 * np.pi / 180.0,
        release_angle=45.0 * np.pi / 180.0,
        weight_angle=10.0 * np.pi / 180.0,
    )


@pytest.fixture(scope="module")
def environment() -> EnvironmentConfig:
    """Initialize a default environment configuration."""
    return EnvironmentConfig()


def test_initialization_hcw(
    hcw_trebuchet: HingedCounterweightTrebuchet,
    environment: EnvironmentConfig,
) -> None:
    """Test simulation initialization."""
    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        environment=environment,
    )

    assert simulation.trebuchet == hcw_trebuchet
    assert simulation.environment == environment

    assert simulation._sling_phases == (
        SlingPhases.SLIDING_OVER_GROUND,
        SlingPhases.UNCONSTRAINED,
    )


def test_initialization_whipper(
    whipper_trebuchet: WhipperTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test simulation initialization with whipper trebuchet."""
    simulation = Simulation(
        trebuchet=whipper_trebuchet,
        environment=environment,
    )

    assert simulation.trebuchet == whipper_trebuchet
    assert simulation.environment == environment

    assert simulation._sling_phases == (
        SlingPhases.PROJECTILE_AND_COUNTERWEIGHT_CONTACT_ARM,
        SlingPhases.PROJECTILE_CONTACT_ARM,
        SlingPhases.UNCONSTRAINED,
    )


def test_hcw_simulation(
    hcw_trebuchet: HingedCounterweightTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test simulation with a hinged counterweight trebuchet."""
    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        environment=environment,
    )

    simulation.solve()

    assert simulation.get_phase_end_time(
        sim_phase=SimulationPhases.SLING,
        sling_phase=SlingPhases.SLIDING_OVER_GROUND,
    ) == pytest.approx(0.6751505542485441, abs=1e-6)
    assert simulation.get_phase_end_time(
        sim_phase=SimulationPhases.SLING,
        sling_phase=SlingPhases.UNCONSTRAINED,
    ) == pytest.approx(1.6497669591178958, abs=1e-6)
    assert simulation.get_phase_end_time(
        sim_phase=SimulationPhases.BALLISTIC
    ) == pytest.approx(6.002567303576664, abs=1e-6)
    assert simulation.distance_traveled == pytest.approx(65.81262624344718, rel=1e-6)


def test_unsolved_simulation_raises_error(
    whipper_trebuchet: WhipperTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test that accessing results before solving raises ValueError."""
    simulation = Simulation(
        trebuchet=whipper_trebuchet,
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
    hcw_trebuchet: HingedCounterweightTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test that sling tension verification works correctly."""
    simulation = Simulation(
        trebuchet=hcw_trebuchet,
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
        simulation.get_tsteps(
            sim_phase=SimulationPhases.SLING, sling_phase=SlingPhases.ALL
        )
    )
    assert np.all(tension_array)

    # Now test a case where the sling goes slack
    total_arm_length = (
        hcw_trebuchet.arm.length_weight_side + hcw_trebuchet.arm.length_projectile_side
    )
    arm_fraction = 0.65
    hcw_trebuchet.arm.length_projectile_side = total_arm_length * arm_fraction
    hcw_trebuchet.arm.length_weight_side = total_arm_length * (1 - arm_fraction)

    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        environment=environment,
        verify_sling_tension=False,
    )  # will verify manually
    simulation.solve()
    tension_array = simulation.where_sling_in_tension()

    assert isinstance(tension_array, np.ndarray)
    assert tension_array.dtype == bool
    assert len(tension_array) == len(
        simulation.get_tsteps(
            sim_phase=SimulationPhases.SLING, sling_phase=SlingPhases.ALL
        )
    )
    assert not np.all(tension_array)


def test_whipper_simulation(
    whipper_trebuchet: WhipperTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test simulation with a whipper-style trebuchet."""
    simulation = Simulation(trebuchet=whipper_trebuchet, environment=environment)

    simulation.solve()

    # Attempt to access launch distance and times
    # to ensure simulation ran without errors
    _ = simulation.distance_traveled
    _ = simulation.get_phase_end_time(sim_phase=SimulationPhases.BALLISTIC)


def test_invalid_trebuchet_type(projectile: Projectile) -> None:
    """Test that initialization with invalid trebuchet type raises TypeError."""

    class InvalidTrebuchet:
        """Mock invalid trebuchet class."""

        def __init__(self) -> None:
            self.projectile = projectile

    with pytest.raises(TypeError, match="Invalid trebuchet configuration"):
        Simulation(trebuchet=InvalidTrebuchet())


def test_sling_goes_slack_warning(
    hcw_trebuchet: HingedCounterweightTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test that a warning is raised when sling goes slack."""
    # Modify trebuchet to make sling go slack
    total_arm_length = (
        hcw_trebuchet.arm.length_weight_side + hcw_trebuchet.arm.length_projectile_side
    )
    arm_fraction = 0.65
    hcw_trebuchet.arm.length_projectile_side = total_arm_length * arm_fraction
    hcw_trebuchet.arm.length_weight_side = total_arm_length * (1 - arm_fraction)

    simulation = Simulation(
        trebuchet=hcw_trebuchet,
        environment=environment,
        verify_sling_tension=True,  # Enable automatic verification
    )

    with pytest.warns(UserWarning, match="Sling goes slack during the simulation"):
        simulation.solve()


def test_get_phase_end_time_errors(
    hcw_trebuchet: HingedCounterweightTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test error cases for get_phase_end_time method."""
    simulation = Simulation(trebuchet=hcw_trebuchet, environment=environment)
    simulation.solve()

    # Test missing sling_phase when sim_phase is SLING
    with pytest.raises(ValueError, match="sling_phase must be specified"):
        simulation.get_phase_end_time(sim_phase=SimulationPhases.SLING)

    # Test sling_phase provided when sim_phase is BALLISTIC
    with pytest.raises(ValueError, match="sling_phase should be None"):
        simulation.get_phase_end_time(
            sim_phase=SimulationPhases.BALLISTIC,
            sling_phase=SlingPhases.UNCONSTRAINED,
        )


def test_get_phase_state_variables_errors(
    hcw_trebuchet: HingedCounterweightTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test error cases for _get_phase_state_variables method."""
    simulation = Simulation(trebuchet=hcw_trebuchet, environment=environment)
    simulation.solve()

    # Test sling_phase provided when sim_phase is BALLISTIC
    with pytest.raises(ValueError, match="sling_phase should be None"):
        simulation._get_phase_state_variables(
            sim_phase=SimulationPhases.BALLISTIC,
            sling_phase=SlingPhases.UNCONSTRAINED,
        )

    # Test missing sling_phase when sim_phase is SLING
    with pytest.raises(ValueError, match="sling_phase must be specified"):
        simulation._get_phase_state_variables(sim_phase=SimulationPhases.SLING)


def test_get_projectile_state_variables_invalid_phase(
    hcw_trebuchet: HingedCounterweightTrebuchet, environment: EnvironmentConfig
) -> None:
    """Test error case for invalid phase in get_projectile_state_variables."""
    simulation = Simulation(trebuchet=hcw_trebuchet, environment=environment)
    simulation.solve()

    # Test invalid phase
    with pytest.raises(ValueError, match="Invalid phase"):
        simulation.get_projectile_state_variables(phase="INVALID")
