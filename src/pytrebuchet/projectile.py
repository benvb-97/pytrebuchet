from math import pi
from typing import Callable
from pytrebuchet.drag_coefficient import drag_coefficient_smooth_sphere_clift_grace_weber

class Projectile:
    """
    Represents a projectile launched by the trebuchet.
    It assumes a spherical shape for drag calculations.
    """

    def __init__(
        self,
        mass: float,
        diameter: float,
        drag_coefficient: float | Callable = None,
    ):
        """
        Initializes a Projectile instance. It assumes a spherical shape for drag calculations.
        :param mass: mass of the projectile (kg)
        :param diameter: diameter of the projectile (m)
        :param drag_coefficient: drag coefficient (dimensionless)
            - float: constant drag coefficient
            - callable: function that takes Reynolds number as input and returns drag coefficient.
                        The function should be vectorized to handle numpy arrays.
            - None: uses default drag coefficient calculation for smooth sphere
        """

        self.mass = mass
        self.diameter = diameter

        self.effective_area = pi * (diameter / 2) ** 2  # cross-sectional area

        if type(drag_coefficient) is float:  # constant drag coefficient
            self.drag_coefficient = lambda reynolds: drag_coefficient
        elif drag_coefficient is None:  # default drag coefficient for smooth sphere
            self.drag_coefficient = drag_coefficient_smooth_sphere_clift_grace_weber
        else:
            self.drag_coefficient = drag_coefficient

    @classmethod
    def default(cls) -> "Projectile":
        """
        Creates a projectile with default parameters used by https://virtualtrebuchet.com (pumpkin).
        """
        return cls(mass=4, diameter=0.35)
