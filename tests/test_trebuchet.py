import numpy as np
import pytest

from pytrebuchet import Projectile, Simulation, Trebuchet


def init_trebuchet(
    projectile_touch_ground: bool = True,
) -> Trebuchet:
    """
    Initializes a Trebuchet instance with predefined parameters for testing.
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


class TestTrebuchet:

    def test_initialization(self):

        # Test trebuchet initialization with projectile arm touching the ground
        trebuchet = init_trebuchet()

        assert trebuchet.l_weight_arm == 1.75
        assert trebuchet.l_projectile_arm == 6.8
        assert trebuchet.l_sling_projectile == 6.83
        assert trebuchet.l_sling_weight == 2.0
        assert trebuchet.h_pivot == 5.0
        assert trebuchet.mass_arm == 10.7
        assert trebuchet.inertia_arm == 65.0
        assert trebuchet.d_pivot_to_arm_cog == 2.52

        assert trebuchet.init_angle_arm == pytest.approx(0.8261005419270769, abs=1e-12)
        assert trebuchet.init_angle_weight == pytest.approx(-np.pi / 2, abs=1e-12)
        assert trebuchet.init_angle_projectile == pytest.approx(0.0, abs=1e-12)

        # Test trebuchet initialization with projectile arm not touching the ground
        trebuchet = init_trebuchet(projectile_touch_ground=False)

        assert trebuchet.init_angle_arm == pytest.approx(55 * np.pi / 180, abs=1e-12)
        assert trebuchet.init_angle_weight == pytest.approx(-np.pi / 2, abs=1e-12)
        assert trebuchet.init_angle_projectile == pytest.approx(-np.pi / 2, abs=1e-12)

    def test_position_calculations(self):
        trebuchet = init_trebuchet()

        # test calculation of projectile arm end point
        x_ap, y_ap = trebuchet.calculate_arm_endpoint_projectile(
            trebuchet.init_angle_arm
        )
        assert x_ap == pytest.approx(-4.608687448721166, abs=1e-12)
        assert y_ap == pytest.approx(0.0, abs=1e-12)

        # test calculation of weight arm end point
        x_aw, y_aw = trebuchet.calculate_arm_endpoint_weight(trebuchet.init_angle_arm)
        assert x_aw == pytest.approx(1.1860592698914765, abs=1e-12)
        assert y_aw == pytest.approx(6.286764705882353, abs=1e-12)

        # test calculation of projectile position
        x_p, y_p = trebuchet.calculate_projectile_point(
            angle_arm=trebuchet.init_angle_arm,
            angle_projectile=trebuchet.init_angle_projectile,
        )
        assert x_p == pytest.approx(2.2213125512788343, abs=1e-12)
        assert y_p == pytest.approx(0.0, abs=1e-12)

        # test calculation of weight position
        x_w, y_w = trebuchet.calculate_weight_point(
            angle_arm=trebuchet.init_angle_arm, angle_weight=trebuchet.init_angle_weight
        )
        assert x_w == pytest.approx(1.1860592698914765, abs=1e-12)
        assert y_w == pytest.approx(4.286764705882353, abs=1e-12)

    def test_simulation(self):
        trebuchet = init_trebuchet()
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

        assert simulation.ground_separation_time == pytest.approx(
            0.6753801120981405, abs=1e-6
        )
        assert simulation.sling_release_time == pytest.approx(
            1.6496185045932605, abs=1e-6
        )
        assert simulation.projectile_hits_ground_time == pytest.approx(
            5.998627164992915, abs=1e-6
        )
        assert simulation.distance_traveled == pytest.approx(
            65.27682068472352, rel=1e-6
        )
