import numpy as np
import pytest

from pytrebuchet import Projectile, Simulation, Trebuchet
from pytrebuchet.simulation import SimulationPhases


def init_hcw_trebuchet(
    projectile_touch_ground: bool = True,
) -> Trebuchet:
    """Initializes a hinged counterweight Trebuchet instance with predefined parameters for testing.
    :param projectile_touch_ground: If True, sets the parameters so that the projectile arm touches the ground.
    :return: Trebuchet instance
    """
    if projectile_touch_ground:
        h_pivot = 5.0
    else:
        h_pivot = 15.0

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


class TestSimulation:
    def test_initialization_hcw(self):
        """Test simulation initialization"""
        trebuchet = init_hcw_trebuchet()
        projectile = Projectile(mass=4.0, diameter=0.35)
        simulation = Simulation(
            trebuchet,
            projectile,
            wind_speed=0.0,
            air_density=1.225,
            air_kinematic_viscosity=1.47e-5,
            gravitational_acceleration=9.81,
        )

        assert simulation.trebuchet == trebuchet
        assert simulation.projectile == projectile
        assert simulation.wind_speed == 0.0
        assert simulation.air_density == 1.225
        assert simulation.air_kinematic_viscosity == 1.47e-5
        assert simulation.gravitational_acceleration == 9.81

        assert simulation.phases == (
            SimulationPhases.GROUND_SLIDING,
            SimulationPhases.SLING_UNCONSTRAINED,
            SimulationPhases.BALLISTIC,
        )

    def test_initialization_whipper(self):
        """Test simulation initialization with whipper trebuchet"""
        trebuchet = Trebuchet.default_whipper()
        projectile = Projectile.default()
        simulation = Simulation(
            trebuchet,
            projectile,
            wind_speed=5.0,
            air_density=1.2,
            air_kinematic_viscosity=1.5e-5,
            gravitational_acceleration=9.81,
        )

        assert simulation.trebuchet == trebuchet
        assert simulation.projectile == projectile
        assert simulation.wind_speed == 5.0
        assert simulation.air_density == 1.2
        assert simulation.air_kinematic_viscosity == 1.5e-5
        assert simulation.gravitational_acceleration == 9.81

        assert simulation.phases == (
            SimulationPhases.WHIPPER_BOTH_CONSTRAINED,
            SimulationPhases.WHIPPER_PROJECTILE_CONSTRAINED,
            SimulationPhases.SLING_UNCONSTRAINED,
            SimulationPhases.BALLISTIC,
        )

    def test_hcw_simulation(self):
        trebuchet = init_hcw_trebuchet()
        projectile = Projectile(mass=4.0, diameter=0.35)
        simulation = Simulation(
            trebuchet,
            projectile,
            wind_speed=0.0,
            air_density=1.225,
            air_kinematic_viscosity=1.47e-5,
            gravitational_acceleration=9.81,
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
        assert simulation.distance_traveled == pytest.approx(
            65.81262624344718, rel=1e-6
        )

    def test_unsolved_simulation_raises_error(self):
        """Test that accessing results before solving raises ValueError."""
        trebuchet = init_hcw_trebuchet()
        projectile = Projectile.default()
        simulation = Simulation(trebuchet, projectile)

        # Test that accessing methods requiring solved simulation raises ValueError
        with pytest.raises(ValueError, match="Simulation has not been run yet"):
            simulation.get_tsteps()

        with pytest.raises(ValueError, match="Simulation has not been run yet"):
            simulation.get_trebuchet_state_variables()

        with pytest.raises(ValueError, match="Simulation has not been run yet"):
            simulation.get_projectile_state_variables()

        with pytest.raises(ValueError, match="Simulation has not been run yet"):
            simulation.where_sling_in_tension()

    def test_sling_tension_verification(self):
        """Test that sling tension verification works correctly."""
        trebuchet = init_hcw_trebuchet()
        projectile = Projectile.default()
        simulation = Simulation(
            trebuchet, projectile, verify_sling_tension=False
        )  # will verify manually

        simulation.solve()

        tension_array = simulation.where_sling_in_tension()

        # Check that the tension array is boolean, has correct length and all values are True (sling in tension)
        assert isinstance(tension_array, np.ndarray)
        assert tension_array.dtype == bool
        assert len(tension_array) == len(
            simulation.get_tsteps(phase=SimulationPhases.TREBUCHET)
        )
        assert np.all(tension_array)

        # Now test a case where the sling goes slack
        total_arm_length = trebuchet.l_weight_arm + trebuchet.l_projectile_arm
        arm_fraction = 0.65
        trebuchet.l_projectile_arm = total_arm_length * arm_fraction
        trebuchet.l_weight_arm = total_arm_length * (1 - arm_fraction)

        simulation = Simulation(
            trebuchet, projectile, verify_sling_tension=False
        )  # will verify manually
        simulation.solve()
        tension_array = simulation.where_sling_in_tension()

        assert isinstance(tension_array, np.ndarray)
        assert tension_array.dtype == bool
        assert len(tension_array) == len(
            simulation.get_tsteps(phase=SimulationPhases.TREBUCHET)
        )
        assert not np.all(tension_array)

    def test_whipper_simulation(self):
        """Test simulation with a whipper-style trebuchet."""
        trebuchet = Trebuchet.default_whipper()
        projectile = Projectile.default()
        simulation = Simulation(
            trebuchet,
            projectile,
            wind_speed=0.0,
            air_density=1.225,
            air_kinematic_viscosity=1.47e-5,
            gravitational_acceleration=9.81,
        )

        simulation.solve()

        # Attempt to access launch distance and times to ensure simulation ran without errors
        _ = simulation.distance_traveled
        _ = simulation.get_phase_end_time(phase=SimulationPhases.BALLISTIC)
