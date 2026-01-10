"""Module for simulating trebuchet projectile launches."""

import warnings
from enum import StrEnum
from functools import wraps
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from differential_equations import (
    SlingPhases,
    ballistic_ode,
    projectile_hits_ground_event,
    sling_ode,
    sling_terminate_event,
)
from environment import EnvironmentConfig
from trebuchet import HingedCounterweightTrebuchet, WhipperTrebuchet

if TYPE_CHECKING:
    from trebuchet import Trebuchet


def requires_solved(func: callable) -> callable:
    """Ensure that the simulation has been solved before accessing the decorated method.

    :raises ValueError: If the simulation has not been run yet (self.solved is False).

    """

    @wraps(func)
    def wrapper(self: "Simulation", *args: object, **kwargs: object) -> callable:
        if not self.solved:
            msg = "Simulation has not been run yet."
            raise ValueError(msg)
        return func(self, *args, **kwargs)

    return wrapper


class SimulationPhases(StrEnum):
    """Enumeration of the different phases of the simulation."""

    # Both sling and ballistic phases
    ALL = "ALL"

    # Projectile in sling phase
    SLING = "SLING"

    # Projectile in ballistic phase
    BALLISTIC = "BALLISTIC"


class Simulation:
    """Class for simulating trebuchet projectile launches.

    :param trebuchet: Trebuchet object containing trebuchet parameters
    :param wind_speed: Wind speed in m/s (default is 0.0)
    :param air_density: Density of the air in kg/m^3
     (default is standard air density at sea level)
    :param air_kinematic_viscosity: Kinematic viscosity of the air in m^2/s
     (default is approximate value at 15 degrees Celsius at sea level)
    :param gravitational_acceleration: Gravitational acceleration in m/s^2
     (default is Earth's gravity)
    :param verify_sling_tension: Whether to verify sling tension after solving
     the simulation (default is True)
    :param atol: Absolute tolerance for the ODE solver
     (default is 1e-6)
    :param rtol: Relative tolerance for the ODE solver
     (default is 1e-5), spikes in the distance calculations occur for rtol >= 1e-4
    """

    def __init__(
        self,
        trebuchet: "Trebuchet",
        environment: EnvironmentConfig | None = None,
        *,
        verify_sling_tension: bool = True,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        """Initialize the simulation with the given trebuchet and projectile.

        :param trebuchet: Trebuchet object containing trebuchet parameters
        :param environment: EnvironmentConfig object containing environment parameters
        :param verify_sling_tension: Whether to verify sling tension
         after solving the simulation (default is True)
        :param atol: Absolute tolerance for the ODE solver (default is 1e-6)
        :param rtol: Relative tolerance for the ODE solver (default is 1e-5),
         spikes in the distance calculations seem to occur for rtol >= 1e-4

        """
        self.trebuchet = trebuchet
        self.projectile = trebuchet.projectile

        if environment is None:
            environment = EnvironmentConfig()
        self.environment = environment

        self._verify_sling_tension = verify_sling_tension

        # tolerances for the ODE solver
        self._atol = atol  # absolute tolerance
        self._rtol = rtol  # relative tolerance

        # sling phases to be simulated, in order
        if isinstance(self.trebuchet, HingedCounterweightTrebuchet):
            self._sling_phases = (
                SlingPhases.SLIDING_OVER_GROUND,
                SlingPhases.UNCONSTRAINED,
            )
        elif isinstance(self.trebuchet, WhipperTrebuchet):
            self._sling_phases = (
                SlingPhases.PROJECTILE_AND_COUNTERWEIGHT_CONTACT_ARM,
                SlingPhases.PROJECTILE_CONTACT_ARM,
                SlingPhases.UNCONSTRAINED,
            )
        else:
            msg = f"Invalid trebuchet configuration '{type(self.trebuchet)}'."
            raise TypeError(msg)

        # solve_ivp solutions for each sling phase
        self._sling_phase_solutions = dict.fromkeys(self._sling_phases)
        self._ballistic_solution = None

    def solve(self) -> None:
        """Run the simulation of the trebuchet launching the projectile."""
        # Solve differential equations for each phase
        for i in range(len(self._sling_phases)):
            self._solve_sling_phase(phase_index=i)
        self._solve_ballistic_phase()

        # Assert that projectile hits the ground
        if self.get_phase_end_time(ballistic_phase=True) is None:
            msg = "Projectile did not hit the ground during the simulation."
            raise RuntimeError(msg)

        # Warn the user if sling tension verification fails
        if self._verify_sling_tension and not np.all(self.where_sling_in_tension()):
            msg = "Sling goes slack during the simulation."
            warnings.warn(msg, stacklevel=1)

    def _solve_sling_phase(self, phase_index: int) -> None:
        """Solve the differential equations for a specific trebuchet phase.

        :param phase_index: The index of the trebuchet phase to be solved
        """
        phase = self._sling_phases[phase_index]

        # Define the stopping event for the phase
        def event(t: float, y: tuple[float, ...], *args: tuple) -> callable:
            return sling_terminate_event(t, y, *args)

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
            prev_phase = self._sling_phases[phase_index - 1]

            y0 = self._sling_phase_solutions[prev_phase].y_events[0][0, :]

            t_end_prev = self.get_phase_end_time(prev_phase)
            t_span = (t_end_prev, t_end_prev + 5.0)
            t_eval = np.linspace(t_end_prev, t_end_prev + 5.0, 5 * 200)

        # Solve the ODE
        self._sling_phase_solutions[phase] = solve_ivp(
            fun=sling_ode[phase],
            t_span=t_span,
            y0=y0,
            args=(
                self.trebuchet,
                self.projectile,
                self.environment,
                self._sling_phases[phase_index],
            ),
            t_eval=t_eval,
            events=event,
            atol=self._atol,
            rtol=self._rtol,
        )

    def _solve_ballistic_phase(self) -> None:
        """Solve the differential equations for the ballistic phase.

        The differential equations have the following arguments:
        environment: EnvironmentConfig object containing environment parameters
        projectile: Projectile object containing properties of the projectile
        """
        # Define the constants for the differential equations
        args = (self.environment, self.projectile)

        # Define the event to stop the integration when the projectile hits the ground
        def event(t: float, y: tuple[float, ...], *args: tuple) -> callable:
            return projectile_hits_ground_event(t, y, *args)

        event.terminal = True
        event.direction = 0

        # Define the initial conditions from the end of the sling phase
        theta0, _, psi0, dtheta0, _, dpsi0 = self._sling_phase_solutions[
            SlingPhases.UNCONSTRAINED
        ].y_events[0][0, :]
        px0, py0 = self.trebuchet.calculate_projectile_point(
            angle_arm=theta0, angle_projectile=psi0
        )

        vx0 = self.trebuchet.arm.length_projectile_side * dtheta0 * np.sin(
            theta0
        ) - self.trebuchet.sling_projectile.length * dpsi0 * np.sin(psi0)
        vy0 = -self.trebuchet.arm.length_projectile_side * dtheta0 * np.cos(
            theta0
        ) + self.trebuchet.sling_projectile.length * dpsi0 * np.cos(psi0)

        y0 = (px0, py0, vx0, vy0)

        # Define the time span for the integration, and the time evaluation points
        release_time = self.get_phase_end_time(sling_phase=SlingPhases.UNCONSTRAINED)
        t_span = (release_time, release_time + 100.0)
        t_eval = np.linspace(release_time, release_time + 100.0, 100 * 200)

        # solve the ODE
        self._ballistic_solution = solve_ivp(
            fun=ballistic_ode,
            t_span=t_span,
            y0=y0,
            args=args,
            t_eval=t_eval,
            events=event,
            atol=self._atol,
            rtol=self._rtol,
        )

    def get_phase_end_time(
        self,
        *,
        sling_phase: SlingPhases | None,
        ballistic_phase: bool = False,
    ) -> float:
        """Return the end time of the specified phase.

        :param sling_phase: The sling phase to get the end time for.
            If specified, ballistic_phase should be False.
        :param ballistic_phase: Whether to get the end time of the ballistic phase.
            If True, sling_phase should be None.

        :return: The end time of the specified phase.
        """
        if sling_phase is not None:
            if ballistic_phase:
                msg = "Only one of sling_phase or ballistic_phase should be specified."
                raise ValueError(msg)
            if self._sling_phase_solutions[sling_phase] is None:
                msg = f"Phase {sling_phase} has not been solved yet."
                raise ValueError(msg)
            return self._sling_phase_solutions[sling_phase].t_events[0][0]

        if not ballistic_phase:
            msg = "One of sling_phase or ballistic_phase must be specified."
            raise ValueError(msg)
        if self._ballistic_solution is None:
            msg = "Ballistic phase has not been solved yet."
            raise ValueError(msg)
        return self._ballistic_solution.t_events[0][0]

    @property
    def solved(self) -> bool:
        """Return whether the simulation has been run.

        :return: True if the simulation has been run, False otherwise.
        """
        if not all(
            self._sling_phase_solutions[phase] is not None
            for phase in self._sling_phases
        ):
            return False
        return self._ballistic_solution is not None

    @property
    @requires_solved
    def distance_traveled(self) -> float:
        """Return the horizontal distance traveled by the projectile.

        Distance as measured from the pivot's x-coordinate.

        :return: Horizontal distance traveled by the projectile.
        """
        return self._ballistic_solution.y_events[0][0, 0]

    @property
    @requires_solved
    def release_velocity(self) -> float:
        """Return the velocity of the projectile at sling release.

        :return: Velocity of the projectile at sling release.
        """
        # Kinetic energy of projectile at sling release
        projectile_vars = self.get_projectile_state_variables(
            phase=SimulationPhases.SLING, calculate_accelerations=False
        )
        return np.sqrt(projectile_vars[-1, 2] ** 2 + projectile_vars[-1, 3] ** 2)

    @requires_solved
    def get_tsteps(
        self, phase: SimulationPhases = SimulationPhases.ALL
    ) -> np.ndarray[float]:
        """Return the time steps for the specified phase(s) of the simulation.

        :param phase: Phase of the simulation to get time steps for.
            Options are:
            ALL - all phases (default)
            BALLISTIC - ballistic phase
            TREBUCHET - trebuchet phases

        :return: Numpy array of time steps for the specified phase(s).
        """
        t_arrays = []
        if phase in (SimulationPhases.ALL, SimulationPhases.SLING):
            for _phase in self._sling_phases:
                t_arrays.append(self._sling_phase_solutions[_phase].t)
                t_arrays.append(self._sling_phase_solutions[_phase].t_events[0])
        if phase in (SimulationPhases.ALL, SimulationPhases.BALLISTIC):
            t_arrays.append(self._ballistic_solution.t)
            t_arrays.append(self._ballistic_solution.t_events[0])
        else:
            msg = (
                f"Invalid phase '{phase}'. Valid options are: "
                f"{SimulationPhases.ALL}, {SimulationPhases.SLING}, ",
                f"{SimulationPhases.BALLISTIC}.",
            )
            raise ValueError(msg)

        return np.concatenate(t_arrays)

    @requires_solved
    def _get_phase_state_variables(self, phase: SimulationPhases) -> np.ndarray[float]:
        """Return the state variables of the specified trebuchet phase.

        Combines the regular solution and the event solution.

        :param phase: The trebuchet phase to get the state variables for.

        :return: Numpy array of shape (n, 6) where n is the number of time steps,
        containing the state variables
        """
        y_arrays = []
        if phase in (SimulationPhases.ALL, SimulationPhases.SLING):
            for _phase in self._sling_phases:
                y_arrays.append(self._sling_phase_solutions[_phase].y.T)
                y_arrays.append(self._sling_phase_solutions[_phase].y_events[0])
        if phase in (SimulationPhases.ALL, SimulationPhases.BALLISTIC):
            y_arrays.append(self._ballistic_solution.y.T)
            y_arrays.append(self._ballistic_solution.y_events[0])
        else:
            msg = (
                f"Invalid phase '{phase}'. Valid options are: "
                f"{SimulationPhases.ALL}, {SimulationPhases.SLING}, ",
                f"{SimulationPhases.BALLISTIC}.",
            )
            raise ValueError(msg)
        return np.concatenate(y_arrays, axis=0)

    @requires_solved
    def get_trebuchet_state_variables(
        self, *, calculate_accelerations: bool = False
    ) -> np.ndarray[float]:
        """Return the state variables (angles and angular velocities) of the trebuchet.

        :param calculate_accelerations:
            Whether to include angular accelerations in the returned array
            (default is False)

        :return: Numpy array of shape (n, 6) or (n, 9)
        where n is the number of time steps, containing the state variables
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
        for phase in self._sling_phases:  # exclude ballistic phase
            variables_phase = self._get_phase_state_variables(phase)

            # Add angular accelerations if requested
            if calculate_accelerations:
                t_steps = self.get_tsteps(phase=phase)

                # Calculate angular accelerations at each time step
                acc_variables = np.zeros((variables_phase.shape[0], 3))
                args = (self.trebuchet, self.projectile, self.environment, phase)
                for i in range(t_steps.size):
                    acc_variables[i, :] = sling_ode[phase](
                        None, variables_phase[i, :], *args
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
        *,
        calculate_accelerations: bool = False,
    ) -> np.ndarray[float]:
        """Return the state variables of the projectile throughout its flight.

        This includes the (positions, velocities, optional accelerations)

        :param phase: Phase of the simulation to get projectile state variables for.
            Options are:
            ALL - all phases (default)
            BALLISTIC - ballistic phase
            TREBUCHET - trebuchet phases

        :param calculate_accelerations:
            Whether to include accelerations in the returned array (default is False)

        :return: Numpy array of shape (n, 4) or (n, 6)
        where n is the number of time steps, containing the state variables
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
            SimulationPhases.SLING,
        ):
            msg = (
                f"Invalid phase '{phase}'. Valid options are:",
                f"{SimulationPhases.ALL}, {SimulationPhases.BALLISTIC}, ",
                f"{SimulationPhases.SLING}.",
            )
            raise ValueError(msg)

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
                    self._ballistic_solution.y.T,
                    self._ballistic_solution.y_events[0],
                ),
                axis=0,
            )

            # Calculate accelerations if requested
            if calculate_accelerations:
                # Calculate angular accelerations at each time step
                acc_ballistic = np.zeros((t_ballistic.size, 3))
                args = (self.environment, self.projectile)
                for i in range(t_ballistic.size):
                    acc_ballistic[i, :] = ballistic_ode(
                        None, projectile_vars[i, :4], *args
                    )[2:]

                # Concatenate accelerations to state variables
                projectile_vars[start_idx:end_idx, 4:] = acc_ballistic

        return projectile_vars

    @requires_solved
    def where_sling_in_tension(
        self, *, return_projection_array: bool = False
    ) -> np.ndarray[bool]:
        """Determine whether the sling is in tension throughout the trebuchet phases.

        :param return_projection_array:
            If True, returns the projection values instead of booleans
            (default is False)

        :return:
            if return_projection_array: Numpy array of projection values
                at each time step.
            else: Numpy boolean array indicating whether the sling is in tension
                at each time step.
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

        # Sling is in tension when the projection of the
        # acceleration vector onto the sling direction is negative
        projection = -(ax * dx + ay * dy)
        sling_tension_positive = projection > 0.0

        if return_projection_array:
            return projection
        return sling_tension_positive
