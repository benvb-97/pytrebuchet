"""Module defining the Projectile class for trebuchet simulations."""

from collections.abc import Callable
from math import pi

import numpy as np
from numpy.typing import NDArray

from pytrebuchet.drag_coefficient import clift_grace_weber


class Projectile:
    """Represent a spherical projectile launched by the trebuchet."""

    drag_coefficient: Callable[
        [float | NDArray[np.floating]], float | NDArray[np.floating]
    ]

    def __init__(
        self,
        *,
        mass: float | None = 3.0,
        density: float | None = None,
        diameter: float = 0.2,
        drag_coefficient: float
        | Callable[[float | NDArray[np.floating]], float | NDArray[np.floating]]
        | None = None,
    ) -> None:
        """Initialize a Projectile instance.

        It assumes a spherical shape for drag calculations.

        :param mass: mass of the projectile (kg). If None, mass is calculated from
          density and diameter.
        :param density: density of the projectile (kg/m^3). If mass is given, density
          should be None.
        :param diameter: diameter of the projectile (m)
        :param drag_coefficient: drag coefficient (dimensionless)

          - float: constant drag coefficient
          - callable: function that takes Reynolds number as input
            and returns a drag coefficient. The function should be vectorized to
            handle numpy arrays.
          - None: use default drag coefficient function for smooth spheres calculated
            using the Clift-Grace-Weber correlation.
        """
        if mass is None:
            if density is None:
                msg = "Either mass or density must be provided."
                raise ValueError(msg)
            mass = density * self.volume
        elif density is not None:
            msg = "Only one of mass or density should be provided."
            raise ValueError(msg)

        self.mass = mass
        self.diameter = diameter

        if isinstance(drag_coefficient, (float, int, np.floating)):
            # constant drag coefficient
            constant_cd: float = float(drag_coefficient)
            self.drag_coefficient = lambda _: constant_cd
        elif drag_coefficient is None:  # default drag coefficient function
            self.drag_coefficient = clift_grace_weber
        else:
            self.drag_coefficient = drag_coefficient

    @classmethod
    def default(cls) -> "Projectile":
        """Create a Projectile instance with default parameters.

        :return: Projectile instance with default mass, diameter, and drag coefficient
        """
        return cls(mass=3.0, diameter=0.2, drag_coefficient=clift_grace_weber)

    @property
    def radius(self) -> float:
        """Get the radius of the spherical projectile.

        :return: radius (m)
        """
        return self.diameter / 2

    @property
    def volume(self) -> float:
        """Get the volume of the spherical projectile.

        :return: volume (m^3)
        """
        return (4 / 3) * pi * (self.radius**3)

    @property
    def density(self) -> float:
        """Get the density of the spherical projectile.

        :return: density (kg/m^3)
        """
        return self.mass / self.volume

    @property
    def effective_area(self) -> float:
        """Get the effective cross-sectional area of the spherical projectile.

        :return: effective area (m^2)
        """
        return pi * (self.radius**2)
