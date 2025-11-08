from math import pi
import numpy as np
from typing import Callable


class Projectile:
    """
    Represents a projectile launched by the trebuchet.
    It assumes a spherical shape for drag calculations.
    """

    def __init__(self, 
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

        self.effective_area = pi * (diameter / 2)**2  # cross-sectional area

        if type(drag_coefficient) == float:  # constant drag coefficient
            self.drag_coefficient = lambda reynolds: drag_coefficient
        elif drag_coefficient is None:  # default drag coefficient for smooth sphere
            self.drag_coefficient = _drag_coefficient_smooth_sphere
        else:
            self.drag_coefficient = drag_coefficient

    @classmethod
    def default(cls) -> 'Projectile':
        """
        Creates a projectile with default parameters used by https://virtualtrebuchet.com (pumpkin).
        """
        return cls(mass=4, diameter=0.35)
    

def _drag_coefficient_smooth_sphere(reynolds_number: float | np.ndarray[float]) -> float | np.ndarray[float]:
    """
    Calculates the drag coefficient for a smooth sphere based on the Reynolds number.
    Source: Data Correlation for Drag Coefficient for Sphere Faith A. Morrison, 
            Department of Chemical Engineering Michigan Technological University, Houghton, MI 49931

    :param reynolds_number: Reynolds number of the sphere
    :return: drag coefficient of the sphere (dimensionless)
    """

    # Correlation is valid for Reynolds numbers between 0.1 and 1e6
    if isinstance(reynolds_number, float):
        msg = f"Reynolds number {reynolds_number} out of range {0.1, 1e6}"
        assert reynolds_number >= 0.1 and reynolds_number <= 1e6, msg
    else:
        min_re, max_re = np.min(reynolds_number), np.max(reynolds_number)
        msg = f"Reynolds number {min_re, max_re} out of range {0.1, 1e6}"
        assert (min_re >= 0.1) & (max_re <= 1e6), msg

    # Calculate drag coefficient using the correlation
    drag_coefficient = 24 / reynolds_number \
                       + 2.6*(reynolds_number/5.0) / (1+np.float_power(reynolds_number/5.0, 1.52)) \
                       + 0.411*np.float_power(reynolds_number/263000, -7.94) / (1+np.power(reynolds_number/263000, -8)) \
                       + np.float_power(reynolds_number, 0.8) / 461000.
    
    return drag_coefficient