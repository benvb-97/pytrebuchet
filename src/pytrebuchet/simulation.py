import warnings
from enum import IntEnum
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
    whipper_both_constrained_ode,
    whipper_projectile_constrained_ode,
    whipper_projectile_separation_event,
    whipper_weight_separation_event,
)
from pytrebuchet.projectile import Projectile
from pytrebuchet.trebuchet import Trebuchet


class SimulationPhases(IntEnum):
    # Special phases, for post-processing but not simulation
    ALL = -2  # all phases
    TREBUCHET = -1  # all phases except ballistic

    # General phases
    SLING_UNCONSTRAINED = 0
    BALLISTIC = 1

    # Hinged counterweight phases
    GROUND_SLIDING = 2

    # Whipper phases
    WHIPPER_BOTH_CONSTRAINED = 3
    WHIPPER_PROJECTILE_CONSTRAINED = 4


phase_to_event_map = {
    SimulationPhases.GROUND_SLIDING: ground_separation_event,
    SimulationPhases.WHIPPER_BOTH_CONSTRAINED: whipper_weight_separation_event,
    SimulationPhases.WHIPPER_PROJECTILE_CONSTRAINED: whipper_projectile_separation_event,
    SimulationPhases.SLING_UNCONSTRAINED: projectile_release_event,
    SimulationPhases.BALLISTIC: projectile_hits_ground_event,
}

phase_to_ode_map = {
    SimulationPhases.GROUND_SLIDING: sliding_projectile_ode,
    SimulationPhases.WHIPPER_BOTH_CONSTRAINED: whipper_both_constrained_ode,
    SimulationPhases.WHIPPER_PROJECTILE_CONSTRAINED: whipper_projectile_constrained_ode,
    SimulationPhases.SLING_UNCONSTRAINED: sling_projectile_ode,
    SimulationPhases.BALLISTIC: ballistic_ode,
}


def requires_solved(func):
    """Decorator that ensures the simulation has been solved before accessing the decorated method.

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
        """Initializes the simulation with the given trebuchet and projectile.

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

        # phases to be simulated, in order
        if self.trebuchet.configuration == "hcw":
            self.phases = (
                SimulationPhases.GROUND_SLIDING,
                SimulationPhases.SLING_UNCONSTRAINED,
                SimulationPhases.BALLISTIC,
            )
        elif self.trebuchet.configuration == "whipper":
            self.phases = (
                SimulationPhases.WHIPPER_BOTH_CONSTRAINED,
                SimulationPhases.WHIPPER_PROJECTILE_CONSTRAINED,
                SimulationPhases.SLING_UNCONSTRAINED,
                SimulationPhases.BALLISTIC,
            )
        else:
            raise ValueError(
                f"Invalid trebuchet configuration '{self.trebuchet.configuration}'. Valid options are 'hcw' and 'whipper'."
            )

        # solve_ivp solutions for each phase
        self._phase_solutions = dict.fromkeys(self.phases)

    def solve(self):
        """Runs the simulation of the trebuchet launching the projectile."""
        # Solve differential equations for each phase
        for i, phase in enumerate(self.phases):
            if phase != SimulationPhases.BALLISTIC:
                self._solve_trebuchet_phase(phase_index=i)
            else:
                self._solve_ballistic_phase()

        # Assert that projectile hits the ground
        assert self.get_phase_end_time(SimulationPhases.BALLISTIC) is not None

        # Warn the user if sling tension verification fails
        if self._verify_sling_tension is True and not np.all(
            self.where_sling_in_tension()
        ):
            warnings.warn(
                "Sling tension verification failed: sling goes slack during the simulation."
            )

    def _get_args_trebuchet_phases(self):
        """Returns the arguments for the trebuchet phases differential equations.

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
        release_angle: angle at which the projectile is released from the sling

        :return: Tuple of arguments for the trebuchet phases ODEs.
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
        """Returns the arguments for the ballistic phase differential equations.

        :return: Tuple of arguments for the ballistic phase ODEs.
        """
        return (
            self.wind_speed,
            self.air_density,
            self.air_kinematic_viscosity,
            self.gravitational_acceleration,
            self.projectile,
        )

    def _solve_trebuchet_phase(self, phase_index: int):
        """Solves the differential equations for one of the trebuchet phases of the simulation.

        :param phase_index: The index of the trebuchet phase to be solved
        """
        phase = self.phases[phase_index]

        # Define the event to stop the integration when the projectile separates from the ground
        def event(t, y, *args):
            return phase_to_event_map[phase](t, y, *args)

        event.terminal = True
        event.direction = 0

        # Define the initial conditions and time span for the integration
        if phase_index == 0:  # first phase
            y0 = (
                self.trebuchet.init_angle_arm,
                self.trebuchet.init_angle_weight,
                self.trebuchet.init_angle_projectile,
                0.0,
                0.0,
                0.0,
            )
            t_span = (0.0, 5.0)
            t_eval = np.linspace(0.0, 5.0, 5 * 200)

        else:  # retrieve from previous phase solution
            prev_phase = self.phases[phase_index - 1]

            y0 = self._phase_solutions[prev_phase].y_events[0][0, :]

            t_end_prev = self.get_phase_end_time(prev_phase)
            t_span = (t_end_prev, t_end_prev + 5.0)
            t_eval = np.linspace(t_end_prev, t_end_prev + 5.0, 5 * 200)

        # Solve the ODE
        self._phase_solutions[phase] = solve_ivp(
            fun=phase_to_ode_map[phase],
            t_span=t_span,
            y0=y0,
            args=self._get_args_trebuchet_phases(),
            t_eval=t_eval,
            events=event,
            atol=self._atol,
            rtol=self._rtol,
        )

    def _solve_ballistic_phase(self):
        """Solves the differential equations for the ballistic phase of the trebuchet simulation.

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
        theta0, _, psi0, dtheta0, _, dpsi0 = self._phase_solutions[
            SimulationPhases.SLING_UNCONSTRAINED
        ].y_events[0][0, :]
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
        release_time = self.get_phase_end_time(SimulationPhases.SLING_UNCONSTRAINED)
        t_span = (release_time, release_time + 100.0)
        t_eval = np.linspace(release_time, release_time + 100.0, 100 * 200)

        # solve the ODE
        self._phase_solutions[SimulationPhases.BALLISTIC] = solve_ivp(
            fun=ballistic_ode,
            t_span=t_span,
            y0=y0,
            args=args,
            t_eval=t_eval,
            events=event,
            atol=self._atol,
            rtol=self._rtol,
        )

    def get_phase_end_time(self, phase: SimulationPhases) -> float:
        """Returns the end time of the specified phase.

        :param phase: The phase to get the end time for.

        :return: The end time of the specified phase.
        """
        if self._phase_solutions[phase] is None:
            raise ValueError(f"Phase {phase} has not been solved yet.")

        end_time = self._phase_solutions[phase].t_events[0][0]
        return end_time

    @property
    def solved(self) -> bool:
        """Returns whether the simulation has been run.
        :return: True if the simulation has been run, False otherwise.
        """
        for phase in self.phases:
            if self._phase_solutions[phase] is None:
                return False
        return True

    @property
    def distance_traveled(self) -> float:
        """Returns the horizontal distance traveled by the projectile as measured from the pivot's x-coordinate.
        :return: Horizontal distance traveled by the projectile.
        """
        if self._phase_solutions[SimulationPhases.BALLISTIC] is None:
            raise ValueError("Simulation has not been run yet.")

        return self._phase_solutions[SimulationPhases.BALLISTIC].y_events[0][0, 0]

    @property
    @requires_solved
    def release_velocity(self) -> float:
        """Returns the velocity of the projectile at sling release.

        :return: Velocity of the projectile at sling release.
        """
        # Kinetic energy of projectile at sling release
        projectile_vars = self.get_projectile_state_variables(
            phase=SimulationPhases.TREBUCHET, calculate_accelerations=False
        )
        vel_release = np.sqrt(projectile_vars[-1, 2] ** 2 + projectile_vars[-1, 3] ** 2)

        return vel_release

    @requires_solved
    def get_tsteps(
        self, phase: SimulationPhases = SimulationPhases.ALL
    ) -> np.ndarray[float]:
        """Returns the time steps for the specified phase(s) of the simulation.

        :param phase: Phase of the simulation to get time steps for.
            Options are:
            ALL - all phases (default)
            BALLISTIC - ballistic phase
            TREBUCHET - trebuchet phases

        :return: Numpy array of time steps for the specified phase(s).
        """
        t_arrays = []
        if phase == SimulationPhases.ALL:
            for _phase in self.phases:
                t_arrays.append(self._phase_solutions[_phase].t)
                t_arrays.append(self._phase_solutions[_phase].t_events[0])
        elif phase == SimulationPhases.TREBUCHET:
            for _phase in self.phases[:-1]:  # exclude ballistic phase
                t_arrays.append(self._phase_solutions[_phase].t)
                t_arrays.append(self._phase_solutions[_phase].t_events[0])
        elif phase in self.phases:
            t_arrays = (
                self._phase_solutions[phase].t,
                self._phase_solutions[phase].t_events[0],
            )
        else:
            raise ValueError(
                f"Invalid phase '{phase}'. Valid options are {SimulationPhases.ALL}, {SimulationPhases.TREBUCHET}, {self.phases}."
            )

        return np.concatenate(t_arrays)

    @requires_solved
    def _get_phase_state_variables(self, phase: SimulationPhases) -> np.ndarray[float]:
        """Returns the state variables of the specified trebuchet phase.
        Combines the regular solution and the event solution.
        """
        return np.concatenate(
            (
                self._phase_solutions[phase].y.T,
                self._phase_solutions[phase].y_events[0],
            ),
            axis=0,
        )

    @requires_solved
    def get_trebuchet_state_variables(
        self, calculate_accelerations: bool = False
    ) -> np.ndarray[float]:
        """Returns the state variables (angles and angular velocities) of the trebuchet throughout its operation.

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
        # Fetch state variables from trebuchet phases
        variables = []
        for phase in self.phases[:-1]:  # exclude ballistic phase
            variables_phase = self._get_phase_state_variables(phase)

            # Add angular accelerations if requested
            if calculate_accelerations:
                t_steps = self.get_tsteps(phase=phase)

                # Calculate angular accelerations at each time step
                acc_variables = np.zeros((variables_phase.shape[0], 3))

                for i in range(t_steps.size):
                    acc_variables[i, :] = phase_to_ode_map[phase](
                        None, variables_phase[i, :], *self._get_args_trebuchet_phases()
                    )[3:]

                # Concatenate accelerations to state variables
                variables_phase = np.hstack((variables_phase, acc_variables))

            variables.append(variables_phase)

        # Concatenate sliding and sling phase variables
        return np.concatenate(variables, axis=0)

    @requires_solved
    def get_projectile_state_variables(
        self,
        phase: SimulationPhases = SimulationPhases.ALL,
        calculate_accelerations: bool = False,
    ) -> np.ndarray[float]:
        """Returns the state variables (positions, velocities, optional accelerations) of the projectile throughout its flight.

        :param phase: Phase of the simulation to get projectile state variables for.
            Options are:
            ALL - all phases (default)
            BALLISTIC - ballistic phase
            TREBUCHET - trebuchet phases

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
        if phase not in (
            SimulationPhases.ALL,
            SimulationPhases.BALLISTIC,
            SimulationPhases.TREBUCHET,
        ):
            raise ValueError(
                f"Invalid phase '{phase}'. Valid options are {SimulationPhases.ALL}, {SimulationPhases.BALLISTIC}, {SimulationPhases.TREBUCHET}."
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
            SimulationPhases.ALL,
            SimulationPhases.TREBUCHET,
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
            SimulationPhases.ALL,
            SimulationPhases.BALLISTIC,
        ):  # Fetch positions and velocities from ballistic phase solution
            t_ballistic = self.get_tsteps(phase=SimulationPhases.BALLISTIC)
            end_idx = start_idx + t_ballistic.size

            projectile_vars[start_idx:end_idx, :4] = np.concatenate(
                (
                    self._phase_solutions[SimulationPhases.BALLISTIC].y.T,
                    self._phase_solutions[SimulationPhases.BALLISTIC].y_events[0],
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
        """Determines whether the sling is in tension throughout the trebuchet phases of the simulation.

        :param return_projection_array:
            If True, returns the projection values instead of booleans (default is False)

        :return: Numpy array of booleans indicating whether the sling is in tension at each time step during the trebuchet phases.
        """
        # Get acceleration vector of the projectile
        projectile_vars = self.get_projectile_state_variables(
            phase=SimulationPhases.TREBUCHET, calculate_accelerations=True
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
