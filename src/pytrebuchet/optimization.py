from enum import StrEnum

import numpy as np
from scipy.optimize import OptimizeResult, minimize, shgo

from pytrebuchet import Projectile, Simulation, Trebuchet


class Parameters(StrEnum):
    length_arm = "length_arm"
    mass_arm = "mass_arm"
    mass_projectile = "mass_projectile"
    diameter_projectile = "diameter_projectile"

    fraction_projectile_arm = "fraction_projectile_arm"
    mass_counterweight = "mass_counterweight"
    length_sling_projectile = "length_sling_projectile"
    length_sling_weight = "length_sling_weight"
    height_pivot = "height_pivot"
    release_angle = "release_angle"


class DesignOptimizer:
    """
    Class to perform design optimization of a trebuchet.
    Caches simulation results to avoid redundant computations.

    Note: it only supports hinged counterweight trebuchets for now, not whipper-style trebuchets.
    """

    def __init__(
        self,
        length_arm: float,
        mass_arm: float,
        mass_projectile: float,
        diameter_projectile: float,
        fraction_projectile_arm: float | tuple[float, tuple[float, float]],
        mass_counterweight: float | tuple[float, tuple[float, float]],
        length_sling_projectile: float | tuple[float, tuple[float, float]],
        length_sling_weight: float | tuple[float, tuple[float, float]],
        height_pivot: float | tuple[float, tuple[float, float]],
        release_angle: float | tuple[float, tuple[float, float]],
        constrain_sling_tension: bool = True,
    ) -> None:
        """
        Initialize the DesignOptimizer with fixed parameters and design variable bounds.

        :param length_arm: Length of the trebuchet arm (projectile + weight arm) (m)
        :param mass_arm: Mass of the trebuchet arm (kg)
        :param mass_projectile: Mass of the projectile (kg)
        :param diameter_projectile: Diameter of the projectile (m)
        :param fraction_projectile_arm: Fraction of arm length that is the projectile arm (float or (float, (min, max)))
            if float, fixed value; if tuple, (initial_guess, (min, max))
        :param mass_counterweight: Mass of the counterweight (float or (float, (min, max)))
            if float, fixed value; if tuple, (initial_guess, (min, max))
        :param length_sling_projectile: Length of the sling (float or (float, (min, max)))
            if float, fixed value; if tuple, (initial_guess, (min, max))
        :param length_sling_weight: Length of the weight sling (float or (float, (min, max)))
            if float, fixed value; if tuple, (initial_guess, (min, max))
        :param height_pivot: Height of the pivot point (float or (float, (min, max)))
            if float, fixed value; if tuple, (initial_guess, (min, max))
        :param release_angle: Release angle of the sling (float or (float, (min, max)))
            if float, fixed value; if tuple, (initial_guess, (min, max))
        :param constrain_sling_tension: Whether to enforce sling tension constraint (sling should not go slack) during optimization (default: True)
        """

        self._cache = {}  # Cache to store simulation results

        # Store fixed parameters
        self._fixed_params: dict[Parameters, float] = {
            Parameters.length_arm: length_arm,
            Parameters.mass_projectile: mass_projectile,
            Parameters.mass_arm: mass_arm,
            Parameters.diameter_projectile: diameter_projectile,
        }

        # Store additional fixed parameters and design variables from inputs
        candidate_vars = {
            Parameters.fraction_projectile_arm: fraction_projectile_arm,
            Parameters.mass_counterweight: mass_counterweight,
            Parameters.length_sling_projectile: length_sling_projectile,
            Parameters.length_sling_weight: length_sling_weight,
            Parameters.height_pivot: height_pivot,
            Parameters.release_angle: release_angle,
        }
        self._vars_index_map: dict[Parameters, int] = {}
        self._x0 = []
        self._bounds = []

        for name, var in candidate_vars.items():
            if type(var) is float:  # considered fixed parameter
                self._fixed_params[name] = var
            elif type(var) is tuple:  # considered design variable
                self._vars_index_map[name] = len(self._x0)
                self._x0.append(var[0])  # initial guess
                self._bounds.append(var[1])  # (min, max)
            else:
                raise ValueError(
                    f"Design variable {name} must be float or tuple(float, (min, max))"
                )

        # Store constraint flags
        self._constrain_sling_tension: bool = constrain_sling_tension

    def optimize(
        self,
        global_optimization: bool = False,
        local_method: str = None,
        options: dict = None,
    ) -> OptimizeResult:
        """
        Perform the optimization of the trebuchet design.
        :param global_optimization: Whether to use global optimization (SHGO) or local (default: False, local)
        :param method: Optimization method to use for local optimization (default: None, uses default method of scipy.optimize.minimize)
        :param options: Additional options to pass to the optimizer

        :return: OptimizeResult object containing optimization results
        """

        # Define constraints
        constraints = []
        if self._constrain_sling_tension:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": self._sling_tension_constraint,
                }
            )

        if global_optimization:
            result = shgo(
                func=self._objective,
                bounds=self._bounds,
                constraints=constraints,
                options=options,
            )
        else:
            result = minimize(
                fun=self._objective,
                x0=self._x0,
                method=local_method,
                bounds=self._bounds,
                constraints=constraints,
                options=options,
                tol=1e-6
            )

        return result

    def _get_by_name(self, x: np.ndarray, name: Parameters) -> float:
        """
        Get the value of a fixed parameter or design variable by its name.

        :param x: Array containing design variables
        :param name: Name of the design variable or fixed parameter

        :return: Value of the design variable
        """
        if name in self._vars_index_map:  # design variable
            index = self._vars_index_map[name]
            return x[index]
        elif name in self._fixed_params:  # fixed parameter
            return self._fixed_params[name]
        else:
            raise ValueError(f"{name} not found.")

    def _solve_simulation(self, x: np.ndarray) -> Simulation:
        """
        Run the simulation with given design variables.
        Caches results to avoid redundant computations.

        :param x: Array containing design variables

        :return: Simulation object with results
        """

        # Check if result is already cached
        key = tuple(x)
        if key in self._cache:
            return self._cache[key]

        # Empty cache before running new simulation
        self._cache.clear()

        # Run simulation with provided parameters
        l_arm = self._get_by_name(x, Parameters.length_arm)
        frac_projectile_arm = self._get_by_name(x, Parameters.fraction_projectile_arm)

        projectile = Projectile(
            mass=self._get_by_name(x, Parameters.mass_projectile),
            diameter=self._get_by_name(x, Parameters.diameter_projectile),
        )
        trebuchet = Trebuchet(
            l_weight_arm=l_arm * (1 - frac_projectile_arm),
            l_projectile_arm=l_arm * frac_projectile_arm,
            l_sling_projectile=self._get_by_name(x, Parameters.length_sling_projectile),
            l_sling_weight=self._get_by_name(x, Parameters.length_sling_weight),
            h_pivot=self._get_by_name(x, Parameters.height_pivot),
            mass_arm=self._get_by_name(x, Parameters.mass_arm),
            mass_weight=self._get_by_name(x, Parameters.mass_counterweight),
            release_angle=self._get_by_name(x, Parameters.release_angle),
        )

        simulation = Simulation(
            trebuchet=trebuchet, projectile=projectile, verify_sling_tension=False
        )
        try:
            simulation.solve()
        except Exception:
            simulation = None

        # Cache the result for future use
        self._cache[key] = simulation
        return simulation

    def _objective(self, x: np.ndarray, *args) -> float:
        """
        Objective function to minimize: negative horizontal distance of the projectile.

        :param x: Array containing design variables
        :param args: tuple of fixed parameters needed for simulation

        :return: Negative horizontal distance traveled by the projectile
        """

        simulation = self._solve_simulation(x, *args)
        if simulation is None:
            return 1e6  # Large penalty for failed simulations

        distance = simulation.distance_traveled
        return -distance

    def _sling_tension_constraint(self, x: np.ndarray, *args) -> float:
        """
        Constraint function: sling must be in tension during trebuchet operation.

        :param x: Array containing design variables
        :param args: tuple of fixed parameters needed for simulation

        :return: minimum sling tension during trebuchet operation
        """

        simulation = self._solve_simulation(x, *args)
        if simulation is None:
            return -1e6  # Large negative value for failed simulations

        min_tension = simulation.where_sling_in_tension(
            return_projection_array=True
        ).min()

        return min_tension

    def get_baseline_distance(self) -> float:
        """
        Get the distance traveled by the projectile for the baseline design (initial guess).

        :return: Horizontal distance traveled by the projectile for baseline design
        """

        simulation = self._solve_simulation(np.array(self._x0))
        if simulation is None:
            raise RuntimeError("Baseline simulation failed.")

        return simulation.distance_traveled
