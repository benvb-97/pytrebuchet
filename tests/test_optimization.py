"""Test cases for the optimization module."""

from math import pi

import pytest

from pytrebuchet import DesignOptimizer
from pytrebuchet.optimization import Parameters


@pytest.fixture(scope="module", params=[[True, False]])
def optimizer_params(request: pytest.FixtureRequest) -> list[bool]:
    """Return parameters for the optimizer fixture."""
    return request.param


@pytest.fixture(scope="module")
def optimizer(optimizer_params: list[bool]) -> DesignOptimizer:
    """Initialize a DesignOptimizer instance with predefined parameters for testing.

    :param optimizer_params: Parameters for the optimizer
    :return: DesignOptimizer instance
    """
    constrain_sling_tension = optimizer_params[0]
    return DesignOptimizer(
        # Define fixed parameters and design variables (initial guesses, bounds)
        length_arm=1.75 + 6.792,  # fixed total arm length
        mass_arm=10,
        mass_projectile=4.0,
        diameter_projectile=0.35,
        fraction_projectile_arm=(0.795, (0.5, 0.9)),
        mass_counterweight=98.09,
        length_sling_projectile=(6.833, (1.0, 10.0)),
        length_sling_weight=2.0,
        height_pivot=5.0,
        release_angle=(45 * pi / 180, (25 * pi / 180, 60 * pi / 180)),
        # Define constraints
        constrain_sling_tension=constrain_sling_tension,  # Ensure sling tension
    )


def test_initialization(
    optimizer: DesignOptimizer, optimizer_params: list[bool]
) -> None:
    """Test DesignOptimizer initialization with various parameters."""
    constrain_sling_tension: bool = optimizer_params[0]

    assert optimizer._fixed_params[Parameters.length_arm] == 1.75 + 6.792
    assert optimizer._fixed_params[Parameters.mass_arm] == 10
    assert optimizer._fixed_params[Parameters.mass_projectile] == 4.0
    assert optimizer._fixed_params[Parameters.diameter_projectile] == 0.35
    assert (
        optimizer._x0[optimizer._vars_index_map[Parameters.fraction_projectile_arm]]
        == 0.795
    )
    assert optimizer._bounds[
        optimizer._vars_index_map[Parameters.fraction_projectile_arm]
    ] == (0.5, 0.9)
    assert optimizer._fixed_params[Parameters.mass_counterweight] == 98.09
    assert (
        optimizer._x0[optimizer._vars_index_map[Parameters.length_sling_projectile]]
        == 6.833
    )
    assert optimizer._bounds[
        optimizer._vars_index_map[Parameters.length_sling_projectile]
    ] == (1.0, 10.0)
    assert optimizer._fixed_params[Parameters.length_sling_weight] == 2.0
    assert optimizer._fixed_params[Parameters.height_pivot] == 5.0
    assert (
        optimizer._x0[optimizer._vars_index_map[Parameters.release_angle]]
        == 45 * pi / 180
    )
    assert optimizer._bounds[optimizer._vars_index_map[Parameters.release_angle]] == (
        25 * pi / 180,
        60 * pi / 180,
    )
    assert optimizer._constrain_sling_tension is constrain_sling_tension


def test_incorrect_inputs() -> None:
    """Verify that incorrect inputs raise appropriate exceptions."""
    with pytest.raises(ValueError):  # noqa: PT011
        DesignOptimizer(
            # Define fixed parameters and design variables (initial guesses, bounds)
            length_arm=1.75 + 6.792,  # fixed total arm length
            mass_arm=10,
            mass_projectile=4.0,
            diameter_projectile=0.35,
            fraction_projectile_arm="test",
            mass_counterweight=98.09,
            length_sling_projectile={"invalid": "length"},
            length_sling_weight=2.0,
            height_pivot=5.0,
            release_angle=(45 * pi / 180, (25 * pi / 180, 60 * pi / 180)),
        )


def test_optimize(optimizer: DesignOptimizer, optimizer_params: list[bool]) -> None:
    """Test the optimization process of the DesignOptimizer."""
    constrain_sling_tension: bool = optimizer_params[0]
    if not constrain_sling_tension:
        return  # Skip further checks if no constraints are applied
    result = optimizer.optimize()

    # Check that optimization was successful
    assert result.success is True

    # Check that optimized parameters are within bounds
    fraction_projectile_arm_opt = result.x[0]
    length_sling_projectile_opt = result.x[1]
    release_angle_opt = result.x[2]

    # Verify bounds are respected
    assert 0.5 <= fraction_projectile_arm_opt <= 0.9
    assert 1.0 <= length_sling_projectile_opt <= 10.0
    assert 25 * pi / 180 <= release_angle_opt <= 60 * pi / 180

    # Verify optimization improved the design (negative objective = better distance)
    optimized_distance = -optimizer._objective(result.x)
    assert optimized_distance > optimizer.get_baseline_distance(), (
        "Optimization should improve throwing distance"
    )
