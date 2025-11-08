import numpy as np
from scipy.integrate import solve_ivp

from pytrebuchet.constants import (
    AIR_DENSITY,
    AIR_KINEMATIC_VISCOSITY,
    GRAVITATIONAL_ACCELERATION_EARTH,
)
from pytrebuchet.differential_equations import (
    free_flight_ode,
    ground_separation_event,
    projectile_hits_ground_event,
    projectile_release_event,
    sliding_projectile_ode,
    sling_projectile_ode,
)
from pytrebuchet.projectile import Projectile
from pytrebuchet.trebuchet import Trebuchet


class Simulation:

    def __init__(
        self,
        trebuchet: Trebuchet,
        projectile: Projectile,
        wind_speed: float = 0.0,
        air_density: float = AIR_DENSITY,
        air_kinematic_viscosity: float = AIR_KINEMATIC_VISCOSITY,
        gravitational_acceleration: float = GRAVITATIONAL_ACCELERATION_EARTH,
    ) -> None:
        """
        Initializes the simulation with the given trebuchet and projectile.
        :param trebuchet: Trebuchet object containing trebuchet parameters
        :param projectile: Projectile object containing projectile parameters
        :param wind_speed: Wind speed in m/s (default is 0.0)
        :param air_density: Density of the air in kg/m^3 (default is standard air density at sea level)
        :param air_kinematic_viscosity: Kinematic viscosity of the air in m^2/s (default is approximate value at 15 degrees Celsius at sea level)
        :param gravitational_acceleration: Gravitational acceleration in m/s^2 (default is Earth's gravity)
        """

        self.trebuchet = trebuchet
        self.projectile = projectile

        self.wind_speed = wind_speed
        self.air_density = air_density
        self.air_kinematic_viscosity = air_kinematic_viscosity
        self.gravitational_acceleration = gravitational_acceleration

        # solve_ivp solutions for each phase
        self._solution_sliding_phase = None
        self._solution_sling_phase = None
        self._solution_free_flight_phase = None

    def solve(self):
        """
        Runs the simulation of the trebuchet launching the projectile.
        """

        # Solve differential equations for each phase
        self._solve_ground_sliding_phase()
        self._solve_sling_phase()
        self._solve_free_flight_phase()

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
        # l1, l2,
        # l3, l4,
        # la, Ia,
        # m1, m2,
        # ma, g
        args = (
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
        # l1, l2,
        # l3, l4,
        # la, Ia,
        # m1, m2,
        # ma, g
        # release_angle
        args = (
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
        )

    def _solve_free_flight_phase(self):
        """
        Solves the differential equations for the free flight phase of the trebuchet simulation.

        The differential equations have the following arguments:
        wind_speed: wind speed
        rho: air density
        nu: kinematic viscosity of the air
        g: gravitational acceleration
        projectile: Projectile object containing properties of the projectile
        """

        # Define the constants for the differential equations
        # wind_speed, rho, nu,
        # g, projectile
        args = (
            self.wind_speed,
            self.air_density,
            self.air_kinematic_viscosity,
            self.gravitational_acceleration,
            self.projectile,
        )

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
        self._solution_free_flight_phase = solve_ivp(
            fun=free_flight_ode,
            t_span=t_span,
            y0=y0,
            args=args,
            t_eval=t_eval,
            events=event,
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
            and self._solution_free_flight_phase is not None
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
        Returns the time when the projectile hits the ground, marking the end of the free flight phase.
        :return: Time when the projectile hits the ground.
        """
        if self._solution_free_flight_phase is None:
            raise ValueError("Simulation has not been run yet.")
        return self._solution_free_flight_phase.t_events[0][0]

    @property
    def distance_traveled(self) -> float:
        """
        Returns the horizontal distance traveled by the projectile as measured from the pivot's x-coordinate.
        :return: Horizontal distance traveled by the projectile.
        """
        if self._solution_free_flight_phase is None:
            raise ValueError("Simulation has not been run yet.")

        return self._solution_free_flight_phase.y_events[0][0, 0]

    @property
    def tsteps_trebuchet(self) -> np.ndarray[float] | None:
        """
        Returns the time steps for the trebuchet phases concatenated into a single array.
        :return: Numpy array of time steps for the trebuchet phases, or None if the simulation has not been run.
        """
        if not self.solved:
            raise ValueError("Simulation has not been run yet.")

        tsteps = np.concatenate(
            (
                self._solution_sliding_phase.t,
                self._solution_sling_phase.t,
            )
        )
        return tsteps

    @property
    def tsteps_projectile(self) -> np.ndarray[float] | None:
        """
        Returns the time steps for the projectile phases concatenated into a single array.
        :return: Numpy array of time steps for the projectile phases, or None if the simulation has not been run.
        """
        if not self.solved:
            raise ValueError("Simulation has not been run yet.")

        tsteps = np.concatenate(
            (
                self._solution_sliding_phase.t,
                self._solution_sling_phase.t,
                self._solution_free_flight_phase.t,
            )
        )
        return tsteps

    @property
    def angles_trebuchet(
        self,
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]] | None:
        """
        Returns the angles of the trebuchet components (arm, weight, projectile) concatenated into single arrays.
        :return: Tuple of numpy arrays containing the angles of the arm, weight, and projectile, or None if the simulation has not been run.
        """
        if not self.solved:
            raise ValueError("Simulation has not been run yet.")

        variables = np.concatenate(
            (
                self._solution_sliding_phase.y.T,
                self._solution_sliding_phase.y_events[0],
                self._solution_sling_phase.y.T,
                self._solution_sling_phase.y_events[0],
            ),
            axis=0,
        )

        angles_arm, angles_weight, angles_projectile = (
            variables[:, 0],
            variables[:, 1],
            variables[:, 2],
        )
        return angles_arm, angles_weight, angles_projectile

    @property
    def projectile_trajectory(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Returns the x and y positions of the projectile throughout its flight.
        :return: Tuple of numpy arrays containing the x and y positions of the projectile.
        """
        if not self.solved:
            raise ValueError("Simulation has not been run yet.")

        # Calculate projectile positions during trebuchet phases
        angles_arm, angles_weight, angles_projectile = self.angles_trebuchet
        x_projectile, y_projectile = self.trebuchet.calculate_projectile_point(
            angle_arm=angles_arm, angle_projectile=angles_projectile
        )

        # Concatenate with projectile positions during free flight phase
        x_projectile = np.concatenate(
            (
                x_projectile,
                self._solution_free_flight_phase.y[0, :].T,
                self._solution_free_flight_phase.y_events[0][:, 0],
            )
        )
        y_projectile = np.concatenate(
            (
                y_projectile,
                self._solution_free_flight_phase.y[1, :].T,
                self._solution_free_flight_phase.y_events[0][:, 1],
            )
        )

        return x_projectile, y_projectile
