import warnings

import numpy as np
from numpy import cos, sin


class Trebuchet:
    """
    Class representing a trebuchet

    The trebuchet's position is defined by three angles:
    - angle_arm: angle of the arm with respect to the horizontal (radians)
    - angle_projectile: angle of the projectile sling with respect to the horizontal (radians)
    - angle_weight: angle of the weight sling with respect to the horizontal (radians)
    """

    def __init__(
        self,
        l_weight_arm: float,
        l_projectile_arm: float,
        l_sling_projectile: float,
        l_sling_weight: float,
        h_pivot: float,
        mass_arm: float,
        mass_weight: float,
        release_angle: float,
        inertia_arm: float = None,
        d_pivot_to_arm_cog: float = None,
        configuration: str = "hcw",
    ) -> None:
        """
        :param l_weight_arm: length of the arm positioned between the pivot and the weight. units: m
        :param l_projectile_arm: length of the arm positioned between the pivot and the projectile. units: m
        :param l_sling_projectile: length of the sling to which the projectile is attached. units: m
        :param l_sling_weight: length of the sling to which the weight is attached. units: m
        :param h_pivot: height of the pivot. units: m
        :param mass_arm: mass of the arm. units: kg
        :param mass_weight: mass of the counterweight. units: kg
        :param release_angle: angle at which the projectile is released from the sling. units: radians
        :param inertia_arm: inertia of the arm. units: kg*m^2
        :param d_pivot_to_arm_cog: distance from the pivot to the arm's center of gravity. units: m
        :param configuration: configuration of the trebuchet. Options are 'hcw' (hinged counterweight) and 'whipper'.
        """

        self.l_weight_arm = l_weight_arm
        self.l_projectile_arm = l_projectile_arm
        self.l_sling_projectile = l_sling_projectile
        self.l_sling_weight = l_sling_weight
        self.h_pivot = h_pivot
        self.mass_arm = mass_arm
        self.mass_weight = mass_weight
        self.release_angle = release_angle

        if inertia_arm is None:  # calculate from moment of inertia of thin rod
            inertia_arm = 1 / 12 * mass_arm * (l_projectile_arm + l_weight_arm) ** 2
        self.inertia_arm = inertia_arm
        if d_pivot_to_arm_cog is None:  # assume uniform rod
            d_pivot_to_arm_cog = (l_projectile_arm - l_weight_arm) / 2
        self.d_pivot_to_arm_cog = d_pivot_to_arm_cog

        self.configuration = configuration
        if self.configuration == "hcw":
            self._calculate_initial_angles_hcw()
        elif self.configuration == "whipper":
            self._calculate_initial_angles_whipper()
        else:
            raise ValueError(
                f"Invalid configuration '{self.configuration}'. Valid options are 'hcw' and 'whipper'."
            )
        
    @classmethod
    def default(cls) -> "Trebuchet":
        """Creates a Trebuchet instance with default parameters as used by https://virtualtrebuchet.com/."""
        return cls(
            l_weight_arm=1.75,
            l_projectile_arm=6.792,
            l_sling_projectile=6.833,
            l_sling_weight=2,
            h_pivot=5,
            mass_arm=10.65,
            mass_weight=98.09,
            release_angle=45 * np.pi / 180.0,
            inertia_arm=None,
            d_pivot_to_arm_cog=None,
        )

    def _calculate_initial_angles_hcw(self) -> None:
        """
        Calculates the initial angles of the trebuchet (arm, projectile sling and weight sling) at the starting position.

        Calculate:
         -angle_arm such that the projectile arm end just touches the ground. If the arm is too short to reach the ground, set a default angle of 55 degrees.
         -angle_weight such that the weight hangs vertically downwards.
         -angle_projectile such that the projectile just touches the ground. If the sling is too short to reach the ground,
          set the sling angle such that it hangs vertically downwards.
        """

        # Arm angle
        if (
            self.h_pivot >= self.l_projectile_arm
        ):  # arm not long enough to reach the ground, set default angle
            self.init_angle_arm = 55.0 * np.pi / 180.0
        else:  # arm touches the ground
            self.init_angle_arm = np.arcsin(self.h_pivot / self.l_projectile_arm)

        # Weight angle: hangs vertically downwards
        self.init_angle_weight = -np.pi / 2

        # Projectile angle
        _, y_arm_projectile = self.calculate_arm_endpoint_projectile(
            self.init_angle_arm
        )
        if (
            y_arm_projectile - self.l_sling_projectile >= 0.0
        ):  # projectile does not reach the ground, sling hangs vertically downwards
            self.init_angle_projectile = -np.pi / 2
        else:  # projectile just touches the ground
            self.init_angle_projectile = np.arcsin(
                (self.l_projectile_arm * sin(self.init_angle_arm) - self.h_pivot)
                / self.l_sling_projectile
            )

    def _calculate_initial_angles_whipper(self) -> None:
        """
        Calculates the initial angles of the trebuchet (arm, projectile sling and weight sling) at the starting position for a whipper trebuchet.

        Calculate:
         -angle_arm such that the projectile arm end points upwards at a 45 degree angle in the direction of the launch.
         -angle_weight such that the weight points upwards at a 55 degree angle in the direction of the launch.
         -angle_projectile such that the projectile points downwards at a 45 degree angle in the direction opposite to the launch.,
        """

        self.init_angle_arm = (180 + 45) * np.pi / 180.0
        self.init_angle_weight = (-270 - 30) * np.pi / 180.0
        self.init_angle_projectile = (-90 - 50) * np.pi / 180.0

    # Position calculation functions
    def calculate_arm_endpoint_projectile(
        self, angle_arm: float | np.ndarray[float]
    ) -> tuple[float | np.ndarray[float], float | np.ndarray[float]]:
        """
        Calculates the x, y coordinates of the arm endpoint attached to the projectile sling based on the given arm angle.

        :param angle_arm: angle of the arm (radians)
        :return: x, y coordinates of the projectile arm endpoint, respectively
        """

        x_arm_projectile = -self.l_projectile_arm * cos(angle_arm)
        y_arm_projectile = self.h_pivot - self.l_projectile_arm * sin(angle_arm)

        # perform checks (arm should not clip through the ground)
        if np.any(y_arm_projectile < -1e-15):
            warnings.warn(
                f"Projectile arm endpoint clips through the ground. lowest y-coordinate: {np.min(y_arm_projectile)}."
            )

        return x_arm_projectile, y_arm_projectile

    def calculate_arm_endpoint_weight(
        self, angle_arm: float | np.ndarray[float]
    ) -> tuple[float | np.ndarray[float], float | np.ndarray[float]]:
        """
        Calculates the x, y coordinates of arm endpoint attached to the weight sling based on the given arm angle.

        :param angle_arm: angle of the arm (radians)
        :return: x, y coordinates of the weight arm endpoint, respectively
        """

        x_arm_weight = self.l_weight_arm * cos(angle_arm)
        y_arm_weight = self.h_pivot + self.l_weight_arm * sin(angle_arm)

        # perform checks (arm should not clip through the ground)
        if np.any(y_arm_weight < -1e-15):
            warnings(
                f"Weight arm endpoint clips through the ground. lowest y-coordinate: {np.min(y_arm_weight)}."
            )

        return x_arm_weight, y_arm_weight

    def calculate_weight_point(
        self,
        angle_arm: float | np.ndarray[float],
        angle_weight: float | np.ndarray[float],
    ) -> tuple[float | np.ndarray[float], float | np.ndarray[float]]:
        """
        Calculates the x, y coordinates of the weight point based on the given arm endpoint and weight sling angle.
        :param angle_arm: angle of the arm (radians)
        :param angle_weight: angle of the weight sling (radians)
        :return: x, y coordinates of the weight point
        """

        x_arm_weight, y_arm_weight = self.calculate_arm_endpoint_weight(angle_arm)

        x_weight = x_arm_weight + cos(angle_weight) * self.l_sling_weight
        y_weight = y_arm_weight + sin(angle_weight) * self.l_sling_weight

        return x_weight, y_weight

    def calculate_projectile_point(
        self,
        angle_arm: float | np.ndarray[float],
        angle_projectile: float | np.ndarray[float],
    ) -> tuple[float | np.ndarray[float], float | np.ndarray[float]]:
        """
        Calculates the x, y coordinates of the projectile point based on the given arm endpoint and projectile angle.
        :param angle_arm: angle of the arm (radians)
        :param angle_projectile: angle of the projectile (radians)
        :return: x, y coordinates of the projectile point
        """

        x_arm_projectile, y_arm_projectile = self.calculate_arm_endpoint_projectile(
            angle_arm
        )

        x_projectile = (
            x_arm_projectile + cos(angle_projectile) * self.l_sling_projectile
        )
        y_projectile = (
            y_arm_projectile + sin(angle_projectile) * self.l_sling_projectile
        )

        # perform checks (projectile should not clip through the ground)
        if np.min(y_projectile < -1e15):
            warnings.warn(
                f"Projectile clips through the ground. lowest y-coordinate: {np.min(y_projectile)}."
            )

        return x_projectile, y_projectile

    def calculate_projectile_velocity(
        self,
        angle_arm: float | np.ndarray[float],
        angle_projectile: float | np.ndarray[float],
        angular_velocity_arm: float | np.ndarray[float],
        angular_velocity_projectile: float | np.ndarray[float],
    ) -> tuple[float | np.ndarray[float], float | np.ndarray[float]]:
        """
        Calculates the x, y components of the projectile velocity based on the given arm and projectile angles and angular velocities.
        :param angle_arm: angle of the arm (radians)
        :param angle_projectile: angle of the projectile (radians)
        :param angular_velocity_arm: angular velocity of the arm (radians/s)
        :param angular_velocity_projectile: angular velocity of the projectile sling (radians/s)
        :return: x, y components of the projectile velocity
        """

        vx = self.l_projectile_arm * angular_velocity_arm * sin(
            angle_arm
        ) - self.l_sling_projectile * angular_velocity_projectile * sin(
            angle_projectile
        )

        vy = -self.l_projectile_arm * angular_velocity_arm * cos(
            angle_arm
        ) + self.l_sling_projectile * angular_velocity_projectile * cos(
            angle_projectile
        )

        return vx, vy

    def calculate_projectile_acceleration(
        self,
        angle_arm: float | np.ndarray[float],
        angle_projectile: float | np.ndarray[float],
        angular_velocity_arm: float | np.ndarray[float],
        angular_velocity_projectile: float | np.ndarray[float],
        angular_acceleration_arm: float | np.ndarray[float],
        angular_acceleration_projectile: float | np.ndarray[float],
    ) -> tuple[float | np.ndarray[float], float | np.ndarray[float]]:
        """
        Calculates the x, y components of the projectile acceleration based on the given arm and projectile angles, angular velocities and angular accelerations.

        :param angle_arm: angle of the arm (radians)
        :param angle_projectile: angle of the projectile (radians)
        :param angular_velocity_arm: angular velocity of the arm (radians/s)
        :param angular_velocity_projectile: angular velocity of the projectile sling (radians/s)
        :param angular_acceleration_arm: angular acceleration of the arm (radians/s^2)
        :param angular_acceleration_projectile: angular acceleration of the projectile sling (radians/s^2)

        :return: x, y components of the projectile acceleration
        """

        theta, psi, dtheta, dpsi, ddtheta, ddpsi = (
            angle_arm,
            angle_projectile,
            angular_velocity_arm,
            angular_velocity_projectile,
            angular_acceleration_arm,
            angular_acceleration_projectile,
        )
        l2, l3 = self.l_projectile_arm, self.l_sling_projectile

        ax = (
            l2 * sin(theta) * ddtheta
            + l2 * cos(theta) * dtheta**2
            - l3 * sin(psi) * ddpsi
            - l3 * cos(psi) * dpsi**2
        )
        ay = (
            l2 * sin(theta) * dtheta**2
            - l2 * cos(theta) * ddtheta
            - l3 * sin(psi) * dpsi**2
            + l3 * cos(psi) * ddpsi
        )

        return ax, ay

    # Plotting functions
    def get_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Returns the maximum x- and y- positions that the points making up the trebuchet can take for plotting.
        :return: (x_min, x_max), (y_min, y_max)
        """
        projectile_length = self.l_projectile_arm + self.l_sling_projectile
        weight_length = self.l_weight_arm + self.l_sling_weight

        max_length = (
            projectile_length if projectile_length > weight_length else weight_length
        )

        x_min = -max_length
        x_max = max_length
        y_min = 0.0
        y_max = self.h_pivot + max_length
        return (x_min, x_max), (y_min, y_max)
