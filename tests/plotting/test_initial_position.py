"""Test initial position plotting."""

import numpy as np
import pytest

from pytrebuchet.plotting.initial_position import (
    _get_trebuchet_limits,
    plot_initial_position,
)
from pytrebuchet.projectile import Projectile
from pytrebuchet.trebuchet import (
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
    )


@pytest.fixture(scope="module")
def trebuchet(projectile: Projectile) -> HingedCounterweightTrebuchet:
    """Create a sample trebuchet for testing."""
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
        pivot=Pivot(height=5.0),
        release_angle=45.0 * np.pi / 180.0,
        projectile=projectile,
    )


def test_get_trebuchet_limits(trebuchet: HingedCounterweightTrebuchet) -> None:
    """Test the calculation of trebuchet plotting limits."""
    limits_x, limits_y = _get_trebuchet_limits(trebuchet)

    # Check that limits are reasonable
    assert limits_x[0] < limits_x[1]
    assert limits_y[0] < limits_y[1]

    # Check that limits encompass trebuchet dimensions
    max_length = (
        trebuchet.arm.length_projectile_side + trebuchet.sling_projectile.length
    )
    assert limits_x[0] == -max_length
    assert limits_x[1] == max_length
    assert limits_y[0] == 0.0
    assert limits_y[1] == max_length + trebuchet.pivot.height


def test_plot_initial_position(trebuchet: HingedCounterweightTrebuchet) -> None:
    """Test plotting the initial position of the trebuchet."""
    _, ax = plot_initial_position(trebuchet, show=False)

    # Check that pivot line plot has correct start and end (x,y) points
    pivot_line = ax.lines[0]
    x_data, y_data = pivot_line.get_data()
    assert x_data[0] == 0.0
    assert x_data[1] == 0.0
    assert y_data[0] == 0.0
    assert y_data[1] == trebuchet.pivot.height
