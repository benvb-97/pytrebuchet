import warnings
from functools import wraps

import numpy as np
from scipy.integrate import solve_ivp

from pytrebuchet.constants import (
    AIR_DENSITY,
    AIR_KINEMATIC_VISCOSITY,
    GRAVITATIONAL_ACCELERATION_EARTH,
)
from pytrebuchet.differential_equations import (
    ballistic_ode,
    ground_separation_event,
    projectile_hits_ground_event,
    projectile_release_event,
    sliding_projectile_ode,
    sling_projectile_ode,
)
from pytrebuchet.projectile import Projectile
from pytrebuchet.trebuchet import Trebuchet


def requires_solved(func):
    """
    Decorator that ensures the simulation has been solved before accessing the decorated method.

    Raises:
        ValueError: If the simulation has not been run yet (self.solved is False).
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.solved:
            raise ValueError("Simulation has not been run yet.")
        return func(self, *args, **kwargs)

    return wrapper


class Simulation:

    def __init__(
        self,
        trebuchet: Trebuchet,
        projectile: Projectile,
        wind_speed: float = 0.0,
        air_density: float = AIR_DENSITY,
        air_kinematic_viscosity: float = AIR_KINEMATIC_VISCOSITY,
        gravitational_acceleration: float = GRAVITATIONAL_ACCELERATION_EARTH,
        verify_sling_tension: bool = True,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        """
        Initializes the simulation with the given trebuchet and projectile.

        :param trebuchet: Trebuchet object containing trebuchet parameters
        :param projectile: Projectile object containing projectile parameters
        :param wind_speed: Wind speed in m/s (default is 0.0)
        :param air_density: Density of the air in kg/m^3 (default is standard air density at sea level)
        :param air_kinematic_viscosity: Kinematic viscosity of the air in m^2/s (default is approximate value at 15 degrees Celsius at sea level)
        :param gravitational_acceleration: Gravitational acceleration in m/s^2 (default is Earth's gravity)
        :param verify_sling_tension: Whether to verify sling tension after solving the simulation (default is True)
        :param atol: Absolute tolerance for the ODE solver (default is 1e-6)
        :param rtol: Relative tolerance for the ODE solver (default is 1e-5), spikes in the distance calculations seem to occur for rtol >= 1e-4

        """

        self.trebuchet = trebuchet
        self.projectile = projectile

        self.wind_speed = wind_speed
        self.air_density = air_density
        self.air_kinematic_viscosity = air_kinematic_viscosity
        self.gravitational_acceleration = gravitational_acceleration

        self._verify_sling_tension = verify_sling_tension

        # tolerances for the ODE solver
        self._atol = atol  # absolute tolerance
        self._rtol = rtol  # relative tolerance

        # solve_ivp solutions for each phase
        self._solution_sliding_phase = None
        self._solution_sling_phase = None
        self._solution_ballistic_phase = None

    def solve(self):
        """
        Runs the simulation of the trebuchet launching the projectile.
        """

        # Solve differential equations for each phase
        self._solve_ground_sliding_phase()
        self._solve_sling_phase()
        self._solve_ballistic_phase()

        # Assert that projectile hits the ground
        assert self.projectile_hits_ground_time is not None

        # Warn the user if sling tension verification fails
        if self._verify_sling_tension is True and not np.all(
            self.where_sling_in_tension()
        ):
            warnings.warn(
                "Sling tension verification failed: sling goes slack during the simulation."
            )

    def _get_args_sliding_phase(self):
        """
        Returns the arguments for the sliding phase differential equations.

        :return: Tuple of arguments for the sliding phase ODEs.
        """
        return (
            self.trebuchet.l_weight_arm,
            self.trebuchet.l_projectile_arm,
            self.trebuchet.l_sling_projectile,
            self.trebuchet.l_sling_weight,
            self.trebuchet.d_pivot_to_arm_cog,
            self.trebuchet.inertia_arm,
            self.trebuchet.mass_weight,
            self.projectile.mass,
            self.trebuchet.mass_arm,
            self.gravitational_acceleration,
        )

    def _get_args_sling_phase(self):
        """
        Returns the arguments for the sling phase differential equations.

        :return: Tuple of arguments for the sling phase ODEs.
        """
        return (
            self.trebuchet.l_weight_arm,
            self.trebuchet.l_projectile_arm,
            self.trebuchet.l_sling_projectile,
            self.trebuchet.l_sling_weight,
            self.trebuchet.d_pivot_to_arm_cog,
            self.trebuchet.inertia_arm,
            self.trebuchet.mass_weight,
            self.projectile.mass,
            self.trebuchet.mass_arm,
            self.gravitational_acceleration,
            self.trebuchet.release_angle,
        )

    def _get_args_ballistic_phase(self):
        """
        Returns the arguments for the ballistic phase differential equations.

        :return: Tuple of arguments for the ballistic phase ODEs.
        """
        return (
            self.wind_speed,
            self.air_density,
            self.air_kinematic_viscosity,
            self.gravitational_acceleration,
            self.projectile,
        )

    def _solve_ground_sliding_phase(self):
        """
        Solves the differential equations for the ground sliding phase of the trebuchet simulation.

        The differential equations have the following constants:
        l1: length of the arm from pivot to weight attachment point
        l2: length of the arm from pivot to projectile attachment point
        l3: length of the sling to which the projectile is attached
        l4: length of the sling to which the weight is attached
        la: distance from the pivot to the arm's center of gravity
        Ia: inertia of the arm
        m1: mass of the weight
        m2: mass of the projectile
        ma: mass of the arm
        g: gravitational acceleration
        """

        # Define the constants for the differential equations
        args = self._get_args_sliding_phase()

        # Define the event to stop the integration when the projectile separates from the ground
        def event(t, y, *args):
            return ground_separation_event(t, y, *args)

        event.terminal = True
        event.direction = 0

        # Define the initial conditions
        y0 = (
            self.trebuchet.init_angle_arm,
            self.trebuchet.init_angle_weight,
            self.trebuchet.init_angle_projectile,
            0.0,
            0.0,
            0.0,
        )

        # Define the time span for the integration, and the time evaluation points
        t_span = (0.0, 5.0)
        t_eval = np.linspace(0.0, 5.0, 5 * 200)

        # Solve the ODE
        self._solution_sliding_phase = solve_ivp(
            fun=sliding_projectile_ode,
            t_span=t_span,
            y0=y0,
            args=args,
            t_eval=t_eval,
            events=event,
            atol=self._atol,
            rtol=self._rtol,
        )

    def _solve_sling_phase(self):
        """
        Solves the differential equations for the sling phase of the trebuchet simulation.

        The differential equations have the following constants:
        l1: length of the arm from pivot to weight attachment point
        l2: length of the arm from pivot to projectile attachment point
        l3: length of the sling to which the projectile is attached
        l4: length of the sling to which the weight is attached
        la: distance from the pivot to the arm's center of gravity
        Ia: inertia of the arm
        m1: mass of the weight
        m2: mass of the projectile
        ma: mass of the arm
        g: gravitational acceleration
        """

        # Define the constants for the differential equations
        args = self._get_args_sling_phase()

        # Define the event to stop the integration when the projectile is released from the sling
        def event(t, y, *args):
            return projectile_release_event(t, y, *args)

        event.terminal = True
        event.direction = 0

        # Define the initial conditions from the end of the ground sliding phase
        y0 = self._solution_sliding_phase.y_events[0][0, :]

        # Define the time span for the integration, and the time evaluation points
        t_span = (self.ground_separation_time, self.ground_separation_time + 5.0)
        t_eval = np.linspace(
            self.ground_separation_time, self.ground_separation_time + 5.0, 5 * 200
        )

        # solve the ODE
        self._solution_sling_phase = solve_ivp(
            fun=sling_projectile_ode,
            t_span=t_span,
            y0=y0,
            args=args,
            t_eval=t_eval,
            events=event,
            atol=self._atol,
            rtol=self._rtol,
        )

    def _solve_ballistic_phase(self):
        """
        Solves the differential equations for the ballistic phase of the trebuchet simulation.

        The differential equations have the following arguments:
        wind_speed: wind speed
        rho: air density
        nu: kinematic viscosity of the air
        g: gravitational acceleration
        projectile: Projectile object containing properties of the projectile
        """

        # Define the constants for the differential equations
        args = self._get_args_ballistic_phase()

        # Define the event to stop the integration when the projectile hits the ground
        def event(t, y, *args):
            return projectile_hits_ground_event(t, y, *args)

        event.terminal = True
        event.direction = 0

        # Define the initial conditions from the end of the sling phase
        theta0, _, psi0, dtheta0, _, dpsi0 = self._solution_sling_phase.y_events[0][
            0, :
        ]
        px0, py0 = self.trebuchet.calculate_projectile_point(
            angle_arm=theta0, angle_projectile=psi0
        )

        vx0 = self.trebuchet.l_projectile_arm * dtheta0 * np.sin(
            theta0
        ) - self.trebuchet.l_sling_projectile * dpsi0 * np.sin(psi0)
        vy0 = -self.trebuchet.l_projectile_arm * dtheta0 * np.cos(
            theta0
        ) + self.trebuchet.l_sling_projectile * dpsi0 * np.cos(psi0)

        y0 = (px0, py0, vx0, vy0)

        # Define the time span for the integration, and the time evaluation points
        t_span = (self.sling_release_time, self.sling_release_time + 100.0)
        t_eval = np.linspace(
            self.sling_release_time, self.sling_release_time + 100.0, 100 * 200
        )

        # solve the ODE
        self._solution_ballistic_phase = solve_ivp(
            fun=ballistic_ode,
            t_span=t_span,
            y0=y0,
            args=args,
            t_eval=t_eval,
            events=event,
            atol=self._atol,
            rtol=self._rtol,
        )

    @property
    def solved(self) -> bool:
        """
        Returns whether the simulation has been run.
        :return: True if the simulation has been run, False otherwise.
        """
        return (
            self._solution_sliding_phase is not None
            and self._solution_sling_phase is not None
            and self._solution_ballistic_phase is not None
        )

    @property
    def ground_separation_time(self) -> float:
        """
        Returns the time when the projectile separates from the ground, marking the end of the ground sliding phase.

        :return: Time when the projectile separates from the ground.
        """
        if self._solution_sliding_phase is None:
            raise ValueError("Simulation has not been run yet.")
        return self._solution_sliding_phase.t_events[0][0]

    @property
    def sling_release_time(self) -> float:
        """
        Returns the time when the projectile is released from the sling, marking the end of the sling phase.

        :return: Time when the projectile is released from the sling.
        """
        if self._solution_sling_phase is None:
            raise ValueError("Simulation has not been run yet.")
        return self._solution_sling_phase.t_events[0][0]

    @property
    def projectile_hits_ground_time(self) -> float:
        """
        Returns the time when the projectile hits the ground, marking the end of the ballistic phase.

        :return: Time when the projectile hits the ground.
        """
        if self._solution_ballistic_phase is None:
            raise ValueError("Simulation has not been run yet.")
        return self._solution_ballistic_phase.t_events[0][0]

    @property
    def distance_traveled(self) -> float:
        """
        Returns the horizontal distance traveled by the projectile as measured from the pivot's x-coordinate.
        :return: Horizontal distance traveled by the projectile.
        """
        if self._solution_ballistic_phase is None:
            raise ValueError("Simulation has not been run yet.")

        return self._solution_ballistic_phase.y_events[0][0, 0]

    @property
    @requires_solved
    def release_velocity(self) -> float:
        """
        Returns the velocity of the projectile at sling release.

        :return: Velocity of the projectile at sling release.
        """
        # Kinetic energy of projectile at sling release
        projectile_vars = self.get_projectile_state_variables(
            phase="trebuchet", calculate_accelerations=False
        )
        vel_release = np.sqrt(projectile_vars[-1, 2] ** 2 + projectile_vars[-1, 3] ** 2)

        return vel_release

    @requires_solved
    def get_tsteps(self, phase: str = "all") -> np.ndarray[float]:
        """
        Returns the time steps for the specified phase(s) of the simulation.

        :param phase: Phase of the simulation to get time steps for.
            Options are:
            "all" - all phases (default)
            "sliding" - ground sliding phase
            "sling" - sling phase
            "ballistic" - ballistic phase
            "trebuchet" - trebuchet phases (ground sliding + sling)

        :return: Numpy array of time steps for the specified phase(s).
        """

        if phase == "all":
            t_arrays = (
                self._solution_sliding_phase.t,
                self._solution_sliding_phase.t_events[0],
                self._solution_sling_phase.t,
                self._solution_sling_phase.t_events[0],
                self._solution_ballistic_phase.t,
                self._solution_ballistic_phase.t_events[0],
            )
        elif phase == "sliding":
            t_arrays = (
                self._solution_sliding_phase.t,
                self._solution_sliding_phase.t_events[0],
            )
        elif phase == "sling":
            t_arrays = (
                self._solution_sling_phase.t,
                self._solution_sling_phase.t_events[0],
            )
        elif phase == "ballistic":
            t_arrays = (
                self._solution_ballistic_phase.t,
                self._solution_ballistic_phase.t_events[0],
            )
        elif phase == "trebuchet":
            t_arrays = (
                self._solution_sliding_phase.t,
                self._solution_sliding_phase.t_events[0],
                self._solution_sling_phase.t,
                self._solution_sling_phase.t_events[0],
            )
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Valid options are 'all', 'sliding', 'sling', 'ballistic', 'trebuchet'."
            )

        return np.concatenate(t_arrays)

    @requires_solved
    def get_trebuchet_state_variables(
        self, calculate_accelerations: bool = False
    ) -> np.ndarray[float]:
        """
        Returns the state variables (angles and angular velocities) of the trebuchet throughout its operation.

        :param calculate_accelerations:
            Whether to include angular accelerations in the returned array (default is False)

        :return: Numpy array of shape (n, 6) or (n, 9) where n is the number of time steps, containing the state variables
            [
            angle_arm,
            angle_weight,
            angle_projectile,
            angular_velocity_arm,
            angular_velocity_weight,
            angular_velocity_projectile,
            angular_acceleration_arm (if calculate_accelerations is True),
            angular_acceleration_weight (if calculate_accelerations is True),
            angular_acceleration_projectile (if calculate_accelerations is True),
            ]
        """

        # Fetch state variables from sliding and sling phases
        variables_sliding = np.concatenate(
            (
                self._solution_sliding_phase.y.T,
                self._solution_sliding_phase.y_events[0],
            ),
            axis=0,
        )
        variables_sling = np.concatenate(
            (self._solution_sling_phase.y.T, self._solution_sling_phase.y_events[0]),
            axis=0,
        )

        # Add angular accelerations if requested
        if calculate_accelerations:
            t_sliding = self.get_tsteps(phase="sliding")
            t_sling = self.get_tsteps(phase="sling")

            # Calculate angular accelerations at each time step
            acc_sliding = np.zeros((variables_sliding.shape[0], 3))
            acc_sling = np.zeros((variables_sling.shape[0], 3))

            for i in range(t_sliding.size):
                acc_sliding[i, :] = sliding_projectile_ode(
                    None, variables_sliding[i, :], *self._get_args_sliding_phase()
                )[3:]
            for i in range(t_sling.size):
                acc_sling[i, :] = sling_projectile_ode(
                    None, variables_sling[i, :], *self._get_args_sling_phase()
                )[3:]

            # Concatenate accelerations to state variables
            variables_sliding = np.hstack((variables_sliding, acc_sliding))
            variables_sling = np.hstack((variables_sling, acc_sling))

        # Concatenate sliding and sling phase variables
        return np.concatenate((variables_sliding, variables_sling), axis=0)

    @requires_solved
    def get_projectile_state_variables(
        self, phase: str = "all", calculate_accelerations: bool = False
    ) -> np.ndarray[float]:
        """
        Returns the state variables (positions, velocities, optional accelerations) of the projectile throughout its flight.

        :param phase: Phase of the simulation to get projectile state variables for.
            Options are:
            "all" - all phases (default)
            "ballistic" - ballistic phase
            "trebuchet" - trebuchet phases (ground sliding + sling)

        :param calculate_accelerations:
            Whether to include accelerations in the returned array (default is False)

        :return: Numpy array of shape (n, 4) or (n, 6) where n is the number of time steps, containing the state variables
            [
            position_x,
            position_y,
            velocity_x,
            velocity_y,
            acceleration_x (if calculate_accelerations is True),
            acceleration_y (if calculate_accelerations is True),
            ]
        """
        if phase not in ("all", "ballistic", "trebuchet"):
            raise ValueError(
                f"Invalid phase '{phase}'. Valid options are 'all', 'ballistic', 'trebuchet'."
            )

        # Initialize array to hold projectile state variables
        projectile_vars = np.empty(
            (
                self.get_tsteps(phase=phase).shape[0],
                4 + (2 if calculate_accelerations else 0),
            ),
            dtype=float,
        )
        start_idx = 0

        if phase in (
            "all",
            "trebuchet",
        ):  # Calculate positions and velocities from trebuchet state variables
            trebuchet_vars = self.get_trebuchet_state_variables(
                calculate_accelerations=calculate_accelerations
            )

            theta, psi, dtheta, dpsi = (
                trebuchet_vars[:, 0],
                trebuchet_vars[:, 2],
                trebuchet_vars[:, 3],
                trebuchet_vars[:, 5],
            )

            px, py = self.trebuchet.calculate_projectile_point(
                angle_arm=theta, angle_projectile=psi
            )
            vx, vy = self.trebuchet.calculate_projectile_velocity(
                angle_arm=theta,
                angle_projectile=psi,
                angular_velocity_arm=dtheta,
                angular_velocity_projectile=dpsi,
            )

            ntsteps_trebuchet = trebuchet_vars.shape[0]
            projectile_vars[:ntsteps_trebuchet, 0] = px
            projectile_vars[:ntsteps_trebuchet, 1] = py
            projectile_vars[:ntsteps_trebuchet, 2] = vx
            projectile_vars[:ntsteps_trebuchet, 3] = vy

            # Calculate accelerations if requested
            if calculate_accelerations:
                ddtheta, ddpsi = trebuchet_vars[:, 6], trebuchet_vars[:, 8]
                ax, ay = self.trebuchet.calculate_projectile_acceleration(
                    angle_arm=theta,
                    angle_projectile=psi,
                    angular_velocity_arm=dtheta,
                    angular_velocity_projectile=dpsi,
                    angular_acceleration_arm=ddtheta,
                    angular_acceleration_projectile=ddpsi,
                )
                projectile_vars[:ntsteps_trebuchet, 4] = ax
                projectile_vars[:ntsteps_trebuchet, 5] = ay

            # Update index
            start_idx += ntsteps_trebuchet

        if phase in (
            "all",
            "ballistic",
        ):  # Fetch positions and velocities from ballistic phase solution
            t_ballistic = self.get_tsteps(phase="ballistic")
            end_idx = start_idx + t_ballistic.size

            projectile_vars[start_idx:end_idx, :4] = np.concatenate(
                (
                    self._solution_ballistic_phase.y.T,
                    self._solution_ballistic_phase.y_events[0],
                ),
                axis=0,
            )

            # Calculate accelerations if requested
            if calculate_accelerations:
                # Calculate angular accelerations at each time step
                acc_ballistic = np.zeros((t_ballistic.size, 3))

                for i in range(t_ballistic.size):
                    acc_ballistic[i, :] = ballistic_ode(
                        None, projectile_vars[i, :4], *self._get_args_ballistic_phase()
                    )[2:]

                # Concatenate accelerations to state variables
                projectile_vars[start_idx:end_idx, 4:] = acc_ballistic

        return projectile_vars

    @requires_solved
    def where_sling_in_tension(
        self, return_projection_array: bool = False
    ) -> np.ndarray[bool]:
        """
        Determines whether the sling is in tension throughout the trebuchet phases of the simulation.

        :param return_projection_array:
            If True, returns the projection values instead of booleans (default is False)

        :return: Numpy array of booleans indicating whether the sling is in tension at each time step during the trebuchet phases.
        """

        # Get acceleration vector of the projectile
        projectile_vars = self.get_projectile_state_variables(
            phase="trebuchet", calculate_accelerations=True
        )
        ax, ay = projectile_vars[:, 4], projectile_vars[:, 5]

        # Get sling direction vector
        trebuchet_vars = self.get_trebuchet_state_variables(
            calculate_accelerations=False
        )
        x_arm, y_arm = self.trebuchet.calculate_arm_endpoint_projectile(
            angle_arm=trebuchet_vars[:, 0]
        )
        x_pro, y_pro = self.trebuchet.calculate_projectile_point(
            angle_arm=trebuchet_vars[:, 0], angle_projectile=trebuchet_vars[:, 2]
        )

        dx = x_pro - x_arm
        dy = y_pro - y_arm

        # Sling is in tension when the projection of the acceleration vector onto the sling direction is negative
        projection = -(ax * dx + ay * dy)
        sling_tension_positive = projection > 0.0

        if return_projection_array:
            return projection
        return sling_tension_positive
