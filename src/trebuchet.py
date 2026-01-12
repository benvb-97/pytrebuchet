"""Module defining the Trebuchet class.

Provides methods for calculating positions, velocities, and accelerations
of trebuchet components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray

from projectile import Projectile


@dataclass
class Arm:
    """Dataclass representing the trebuchet arm."""

    length_weight_side: float  # length of the arm on the weight side (m)
    length_projectile_side: float  # length of the arm on the projectile side (m)
    mass: float  # mass of the arm (kg)
    inertia: float | None = None  # moment of inertia of the arm (kg*m^2)
    d_pivot_to_cog: float | None = None  # distance from pivot to center of gravity (m)

    def __post_init__(self) -> None:
        """Calculate inertia and d_pivot_to_cog if not provided."""
        if self.inertia is None:  # calculate from moment of inertia of thin rod
            self.inertia = 1 / 12 * self.mass * self.total_length**2
        if self.d_pivot_to_cog is None:  # assume uniform rod
            self.d_pivot_to_cog = (
                self.length_projectile_side - self.length_weight_side
            ) / 2

    @property
    def total_length(self) -> float:
        """Calculate the total length of the arm.

        :return: total length of the arm (m)
        """
        return self.length_weight_side + self.length_projectile_side


@dataclass
class Weight:
    """Dataclass representing the trebuchet counterweight."""

    mass: float  # mass of the counterweight (kg)


@dataclass
class Pivot:
    """Dataclass representing the trebuchet pivot."""

    height: float  # height of the pivot (m)


@dataclass
class Sling:
    """Dataclass representing a trebuchet sling."""

    length: float  # length of the sling (m)


class Trebuchet(ABC):
    """Class representing a trebuchet.

    The trebuchet's position is defined by three angles:
    - angle_arm: angle of the arm w.r.t. the horizontal (radians)
    - angle_projectile: angle of the projectile sling w.r.t. the horizontal (radians)
    - angle_weight: angle of the weight sling w.r.t. the horizontal (radians)
    """

    def __init__(
        self,
        arm: Arm,
        weight: Weight,
        pivot: Pivot,
        sling_projectile: Sling,
        sling_weight: Sling,
        release_angle: float = 45 * np.pi / 180.0,
        projectile: Projectile | None = None,
    ) -> None:
        """Initialize a Trebuchet instance with the given parameters.

        :param arm: Arm instance
        :param weight: Weight instance
        :param pivot: Pivot instance
        :param projectile: Projectile instance
        :param sling_projectile: Sling instance for the projectile
        :param sling_weight: Sling instance for the weight
        :param release_angle: angle at which the projectile is released (radians)

        """
        self.arm = arm
        self.weight = weight
        self.pivot = pivot
        self.sling_projectile = sling_projectile
        self.sling_weight = sling_weight
        self.release_angle = release_angle
        self.projectile = projectile if projectile is not None else Projectile.default()

        self.init_angle_arm: float
        self.init_angle_weight: float
        self.init_angle_projectile: float
        self._initialize_angles()

    @abstractmethod
    def _initialize_angles(self) -> None:
        """Calculate the initial angles of the trebuchet based on its configuration."""

    @classmethod
    @abstractmethod
    def default(cls) -> "Trebuchet":
        """Create a Trebuchet instance with default parameters."""

    @overload
    def calculate_arm_cog(self, angle_arm: float) -> tuple[float, float]: ...

    @overload
    def calculate_arm_cog(
        self, angle_arm: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...

    # Position calculation functions
    def calculate_arm_cog(
        self, angle_arm: float | NDArray[np.floating]
    ) -> tuple[float, float] | tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Calculate the x and y coordinates of the arm center of gravity."""
        if self.arm.d_pivot_to_cog is None:
            msg = "Arm center of gravity distance from pivot is not defined."
            raise ValueError(msg)

        x_arm_cog = -self.arm.d_pivot_to_cog * np.cos(angle_arm)
        y_arm_cog = self.pivot.height - self.arm.d_pivot_to_cog * np.sin(angle_arm)
        return x_arm_cog, y_arm_cog

    def calculate_arm_endpoint_projectile(
        self, angle_arm: float | NDArray[np.floating]
    ) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
        """Calculate the x, y coordinates of the projectile arm endpoint.

        :param angle_arm: angle of the arm (radians)
        :return: x, y coordinates of the projectile arm endpoint, respectively
        """
        x_arm_projectile = -self.arm.length_projectile_side * np.cos(angle_arm)
        y_arm_projectile = self.pivot.height - self.arm.length_projectile_side * np.sin(
            angle_arm
        )

        return x_arm_projectile, y_arm_projectile

    def calculate_arm_endpoint_weight(
        self, angle_arm: float | NDArray[np.floating]
    ) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
        """Calculate the x, y coordinates of the weight arm endpoint.

        :param angle_arm: angle of the arm (radians)
        :return: x, y coordinates of the weight arm endpoint, respectively
        """
        x_arm_weight = self.arm.length_weight_side * np.cos(angle_arm)
        y_arm_weight = self.pivot.height + self.arm.length_weight_side * np.sin(
            angle_arm
        )

        return x_arm_weight, y_arm_weight

    def calculate_weight_point(
        self,
        angle_arm: float | NDArray[np.floating],
        angle_weight: float | NDArray[np.floating],
    ) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
        """Calculate the x, y coordinates of the weight point.

        :param angle_arm: angle of the arm (radians)
        :param angle_weight: angle of the weight sling (radians)

        :return: x, y coordinates of the weight point
        """
        x_arm_weight, y_arm_weight = self.calculate_arm_endpoint_weight(angle_arm)

        x_weight = x_arm_weight + np.cos(angle_weight) * self.sling_weight.length
        y_weight = y_arm_weight + np.sin(angle_weight) * self.sling_weight.length

        return x_weight, y_weight

    def calculate_projectile_point(
        self,
        angle_arm: float | NDArray[np.floating],
        angle_projectile: float | NDArray[np.floating],
    ) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
        """Calculate the x, y coordinates of the projectile point.

        :param angle_arm: angle of the arm (radians)
        :param angle_projectile: angle of the projectile (radians)

        :return: x, y coordinates of the projectile point
        """
        x_arm_projectile, y_arm_projectile = self.calculate_arm_endpoint_projectile(
            angle_arm
        )

        x_projectile = (
            x_arm_projectile + np.cos(angle_projectile) * self.sling_projectile.length
        )
        y_projectile = (
            y_arm_projectile + np.sin(angle_projectile) * self.sling_projectile.length
        )

        return x_projectile, y_projectile

    def calculate_projectile_velocity(
        self,
        angle_arm: float | NDArray[np.floating],
        angle_projectile: float | NDArray[np.floating],
        angular_velocity_arm: float | NDArray[np.floating],
        angular_velocity_projectile: float | NDArray[np.floating],
    ) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
        """Calculate the x, y components of the projectile velocity.

        :param angle_arm: angle of the arm (radians)
        :param angle_projectile: angle of the projectile (radians)
        :param angular_velocity_arm: angular velocity of the arm (radians/s)
        :param angular_velocity_projectile: angular velocity of
          the projectile sling (radians/s)
        :return: x, y components of the projectile velocity
        """
        vx = self.arm.length_projectile_side * angular_velocity_arm * np.sin(
            angle_arm
        ) - self.sling_projectile.length * angular_velocity_projectile * np.sin(
            angle_projectile
        )

        vy = -self.arm.length_projectile_side * angular_velocity_arm * np.cos(
            angle_arm
        ) + self.sling_projectile.length * angular_velocity_projectile * np.cos(
            angle_projectile
        )

        return vx, vy

    def calculate_projectile_acceleration(
        self,
        angle_arm: float | NDArray[np.floating],
        angle_projectile: float | NDArray[np.floating],
        angular_velocity_arm: float | NDArray[np.floating],
        angular_velocity_projectile: float | NDArray[np.floating],
        angular_acceleration_arm: float | NDArray[np.floating],
        angular_acceleration_projectile: float | NDArray[np.floating],
    ) -> tuple[float | NDArray[np.floating], float | NDArray[np.floating]]:
        """Calculate the x, y components of the projectile acceleration.

        :param angle_arm: angle of the arm (radians)
        :param angle_projectile: angle of the projectile (radians)
        :param angular_velocity_arm: angular velocity of the arm (radians/s)
        :param angular_velocity_projectile: angular velocity
          of the projectile sling (radians/s)
        :param angular_acceleration_arm: angular acceleration of the arm (radians/s^2)
        :param angular_acceleration_projectile: angular acceleration
          of the projectile sling (radians/s^2)

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
        l2, l3 = self.arm.length_projectile_side, self.sling_projectile.length

        ax = (
            l2 * np.sin(theta) * ddtheta
            + l2 * np.cos(theta) * dtheta**2
            - l3 * np.sin(psi) * ddpsi
            - l3 * np.cos(psi) * dpsi**2
        )
        ay = (
            l2 * np.sin(theta) * dtheta**2
            - l2 * np.cos(theta) * ddtheta
            - l3 * np.sin(psi) * dpsi**2
            + l3 * np.cos(psi) * ddpsi
        )

        return ax, ay


class HingedCounterweightTrebuchet(Trebuchet):
    """Class representing a hinged counterweight trebuchet."""

    def _initialize_angles(self) -> None:
        """Calculate the initial angles of the trebuchet.

        Calculate:
         -angle_arm such that the projectile arm end just touches the ground.
         If the arm is too short to reach the ground, set a default angle of 55 degrees.
         -angle_weight such that the weight hangs vertically downwards.
         -angle_projectile such that the projectile just touches the ground.
         If the sling is too short to reach the ground,
          set the sling angle such that it hangs vertically downwards.
        """
        # Arm angle
        if (
            self.pivot.height >= self.arm.length_projectile_side
        ):  # arm not long enough to reach the ground, set default angle
            self.init_angle_arm = 55.0 * np.pi / 180.0
        else:  # arm touches the ground
            self.init_angle_arm = np.arcsin(
                self.pivot.height / self.arm.length_projectile_side
            )

        # Weight angle: hangs vertically downwards
        self.init_angle_weight = -np.pi / 2

        # Projectile angle
        _, y_arm_projectile = self.calculate_arm_endpoint_projectile(
            self.init_angle_arm
        )
        if (
            y_arm_projectile - self.sling_projectile.length >= 0.0
        ):  # projectile does not reach the ground, sling hangs vertically downwards
            self.init_angle_projectile = -np.pi / 2
        else:  # projectile just touches the ground
            self.init_angle_projectile = np.arcsin(
                (
                    self.arm.length_projectile_side * np.sin(self.init_angle_arm)
                    - self.pivot.height
                )
                / self.sling_projectile.length
            )

    @classmethod
    def default(cls) -> "HingedCounterweightTrebuchet":
        """Create a hinged counterweight Trebuchet instance with default parameters.

        These parameters are taken from https://virtualtrebuchet.com/.
        """
        return cls(
            arm=Arm(
                length_weight_side=1.75,
                length_projectile_side=6.792,
                mass=10.65,
                inertia=None,
                d_pivot_to_cog=None,
            ),
            weight=Weight(mass=98.09),
            pivot=Pivot(height=5),
            projectile=Projectile.default(),
            sling_projectile=Sling(length=6.833),
            sling_weight=Sling(length=2),
            release_angle=45 * np.pi / 180.0,
        )


class WhipperTrebuchet(Trebuchet):
    """Class representing a whipper trebuchet.

    A whipper trebuchet features a hinged counterweight system, but with the
        counterweight hanger positioned at the top of the throwing arm.
    When cocked, the arm points forward in the direction of the throw.
    At the start, the weight and projectile 'rest' on the trebuchet arm.
    """

    def __init__(  # noqa: PLR0913
        self,
        arm: Arm,
        weight: Weight,
        pivot: Pivot,
        projectile: Projectile,
        sling_projectile: Sling,
        sling_weight: Sling,
        release_angle: float,
        arm_angle: float = 60 * np.pi / 180.0,
        weight_angle: float = 10 * np.pi / 180.0,
    ) -> None:
        """Initialize a WhipperTrebuchet instance with the given parameters.

        :param arm: Arm instance
        :param weight: Weight instance
        :param pivot: Pivot instance
        :param projectile: Projectile instance
        :param sling_projectile: Sling instance for the projectile
        :param sling_weight: Sling instance for the weight
        :param release_angle: angle at which the projectile is released (radians)
        :param arm_angle: initial angle of the arm as measured from the horizontal
            (radians).
        :param weight_angle: initial angle of the weight sling as measured from the arm
            (radians).
        """
        self.init_angle_arm = arm_angle
        self.init_angle_weight = weight_angle

        super().__init__(
            arm=arm,
            weight=weight,
            pivot=pivot,
            projectile=projectile,
            sling_projectile=sling_projectile,
            sling_weight=sling_weight,
            release_angle=release_angle,
        )

    def _initialize_angles(self) -> None:
        """Calculate the initial angles of the trebuchet.

        Calculate:
         -angle_arm based on the specified arm_angle parameter.
         -angle_projectile such that the projectile rests on the arm.
         -angle_weight based on the specified weight_angle parameter.
        """
        self.init_angle_arm = np.pi + self.init_angle_arm

        # Initialize projectile angle such that projectile rests on the arm.
        alpha = np.arcsin(
            self.projectile.radius
            / (self.projectile.radius + self.sling_projectile.length)
        )
        self.init_angle_projectile = self.init_angle_arm - alpha

        # Initialize weight angle.
        self.init_angle_weight = self.init_angle_arm - np.pi + self.init_angle_weight

    @classmethod
    def default(cls) -> "WhipperTrebuchet":
        """Create a whipper Trebuchet instance with default parameters."""
        return cls(
            arm=Arm(
                length_weight_side=2.0,
                length_projectile_side=4.0,
                mass=10.0,
                inertia=None,
                d_pivot_to_cog=None,
            ),
            weight=Weight(mass=100.0),
            pivot=Pivot(height=5.0),
            projectile=Projectile.default(),
            sling_projectile=Sling(length=4.0),
            sling_weight=Sling(length=4.0),
            arm_angle=60.0 * np.pi / 180.0,
            release_angle=45.0 * np.pi / 180.0,
            weight_angle=10.0 * np.pi / 180.0,
        )
