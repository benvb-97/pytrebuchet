from math import pi
import pytest

from pytrebuchet import DesignOptimizer, Trebuchet, Projectile, Simulation
from pytrebuchet.optimization import Parameters


def init_optimizer(constrain_sling_tension: bool = True,
                   incorrect_input: bool = False) -> DesignOptimizer:
    """
    Initializes a DesignOptimizer instance with predefined parameters for testing.

    :param constrain_sling_tension: Whether to enforce sling tension constraint
    :param incorrect_input: Whether to use incorrect input types for testing

    :return: DesignOptimizer instance
    """

    if incorrect_input:
        fraction_projectile_arm = "test"
        length_sling_projectile = {"invalid": "length"}
    else:
        fraction_projectile_arm = (0.795, (0.5, 0.9))
        length_sling_projectile = (6.833, (1.0, 10.0))

    optimizer = DesignOptimizer(
    # Define fixed parameters and design variables (initial guesses, bounds)
    length_arm=1.75 + 6.792,  # fixed total arm length
    mass_arm=10,
    mass_projectile=4.0,
    diameter_projectile=0.35,
    fraction_projectile_arm=fraction_projectile_arm,
    mass_counterweight=98.09,
    length_sling_projectile=length_sling_projectile,
    length_sling_weight=2.0,
    height_pivot=5.0,
    release_angle=(45 * pi / 180, (25 * pi / 180, 60 * pi / 180)),
    # Define constraints
    constrain_sling_tension=constrain_sling_tension,  # Ensure sling remains in tension
    )

    return optimizer


class TestDesignOptimizer:
    def test_initialization(self):
        # Test DesignOptimizer initialization with sling tension constraint
        optimizer = init_optimizer(constrain_sling_tension=True)

        assert optimizer._fixed_params[Parameters.length_arm] == 1.75 + 6.792
        assert optimizer._fixed_params[Parameters.mass_arm] == 10
        assert optimizer._fixed_params[Parameters.mass_projectile] == 4.0
        assert optimizer._fixed_params[Parameters.diameter_projectile] == 0.35
        assert optimizer._x0[optimizer._vars_index_map[Parameters.fraction_projectile_arm]] == 0.795
        assert optimizer._bounds[optimizer._vars_index_map[Parameters.fraction_projectile_arm]] == (0.5, 0.9)
        assert optimizer._fixed_params[Parameters.mass_counterweight] == 98.09
        assert optimizer._x0[optimizer._vars_index_map[Parameters.length_sling_projectile]] == 6.833
        assert optimizer._bounds[optimizer._vars_index_map[Parameters.length_sling_projectile]] == (1.0, 10.0)
        assert optimizer._fixed_params[Parameters.length_sling_weight] == 2.0
        assert optimizer._fixed_params[Parameters.height_pivot] == 5.0
        assert optimizer._x0[optimizer._vars_index_map[Parameters.release_angle]] == 45 * pi / 180
        assert optimizer._bounds[optimizer._vars_index_map[Parameters.release_angle]] == (25 * pi / 180, 60 * pi / 180)
        assert optimizer._constrain_sling_tension is True

        # Test DesignOptimizer initialization without sling tension constraint
        optimizer = init_optimizer(constrain_sling_tension=False)

        assert optimizer._constrain_sling_tension is False

    def test_incorrect_inputs(self):
        """
        Verify that incorrect inputs raise appropriate exceptions.
        """
        with pytest.raises(ValueError):
            init_optimizer(incorrect_input=True)

    def test_optimize(self):
        # Test optimization process
        optimizer = init_optimizer(constrain_sling_tension=True)
        result = optimizer.optimize()

        # Check that optimization was successful
        assert result.success is True

        # Check that optimized parameters are within bounds
        fraction_projectile_arm_opt = result.x[0]
        length_sling_projectile_opt = result.x[1]
        release_angle_opt = result.x[2]

        assert fraction_projectile_arm_opt == pytest.approx(0.6820410412899455, rel=1e-3)
        assert length_sling_projectile_opt == pytest.approx(6.002357005460152, rel=1e-3)
        assert release_angle_opt == pytest.approx(0.7330314208247114, rel=1e-3)