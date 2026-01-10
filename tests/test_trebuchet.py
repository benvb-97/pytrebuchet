"""Module containing unit tests for the Trebuchet class in pytrebuchet."""

import numpy as np
import pytest

from drag_coefficient import clift_grace_weber
from projectile import Projectile
from trebuchet import (
    Arm,
    HingedCounterweightTrebuchet,
    Pivot,
    Sling,
    Weight,
)


@pytest.fixture(scope="module")
def projectile() -> Projectile:
    """Fixture to create a default Projectile instance for tests."""
    return Projectile(
        mass=4.0,
        diameter=0.35,
        drag_coefficient=clift_grace_weber,
    )


def init_hcw_trebuchet(
    projectile: Projectile,
    projectile_touch_ground: bool = True,  # noqa: FBT001, FBT002
) -> HingedCounterweightTrebuchet:
    """Initialize a hinged counterweight Trebuchet instance with predefined parameters.

    :param projectile_touch_ground: If True, sets the parameters so that the projectile
    arm touches the ground.

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


def test_initialization(projectile: Projectile) -> None:
    """Test the initialization of the Trebuchet class with different parameters."""
    # Test trebuchet initialization with projectile arm touching the ground
    trebuchet = init_hcw_trebuchet(projectile=projectile, projectile_touch_ground=True)

    assert trebuchet.arm.length_weight_side == 1.75
    assert trebuchet.arm.length_projectile_side == 6.8
    assert trebuchet.sling_projectile.length == 6.83
    assert trebuchet.sling_weight.length == 2.0
    assert trebuchet.pivot.height == 5.0
    assert trebuchet.arm.mass == 10.7
    assert trebuchet.arm.inertia == 65.0
    assert trebuchet.arm.d_pivot_to_cog == 2.52

    assert trebuchet.init_angle_arm == pytest.approx(0.8261005419270769, abs=1e-12)
    assert trebuchet.init_angle_weight == pytest.approx(-np.pi / 2, abs=1e-12)
    assert trebuchet.init_angle_projectile == pytest.approx(0.0, abs=1e-12)

    # Test trebuchet initialization with projectile arm not touching the ground
    trebuchet = init_hcw_trebuchet(projectile=projectile, projectile_touch_ground=False)

    assert trebuchet.init_angle_arm == pytest.approx(55 * np.pi / 180, abs=1e-12)
    assert trebuchet.init_angle_weight == pytest.approx(-np.pi / 2, abs=1e-12)
    assert trebuchet.init_angle_projectile == pytest.approx(-np.pi / 2, abs=1e-12)


def test_position_calculations(projectile: Projectile) -> None:
    """Test the position calculation methods of the Trebuchet class."""
    trebuchet = init_hcw_trebuchet(projectile=projectile, projectile_touch_ground=True)

    # test calculation of projectile arm end point
    x_ap, y_ap = trebuchet.calculate_arm_endpoint_projectile(trebuchet.init_angle_arm)
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
