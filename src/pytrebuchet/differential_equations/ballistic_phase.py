import numpy as np

from pytrebuchet.projectile import Projectile


def ballistic_ode(
    t,
    y,
    *args,
):
    """Represents the ordinary differential equations (ODEs) for a ballistic projectile after release from the sling.

    :param t: time variable (not used in this function but required for ODE solvers)

    :param y: tuple containing the state variables:
        (px, py, vx, vy)
        where:
        px: x position of the projectile
        py: y position of the projectile
        vx: x velocity of the projectile
        vy: y velocity of the projectile

    :param args: additional parameters required for the equations:
        (wind_speed, rho, nu, g, projectile)
        where:
        wind_speed: wind speed
        rho: air density
        nu: kinematic viscosity of the air
        g: gravitational acceleration
        projectile: Projectile object containing properties of the projectile

    :return: derivatives: tuple containing the derivatives of the state variables:
             (vx, vy, ax, ay)
    """
    # Fetch variables
    wind_speed, rho, nu, g, _ = args
    projectile: Projectile = args[-1]
    eff_area, mass_p = projectile.effective_area, projectile.mass
    _, _, vx, vy = y

    # Calculate drag coefficient
    reynolds = calculate_reynolds_number(
        velocity=np.sqrt(vx**2 + vy**2),
        diameter=np.sqrt(4 * eff_area / np.pi),
        air_kinematic_viscosity=nu,
    )
    drag_coeff = projectile.drag_coefficient(reynolds)

    # Calculate accelerations
    ax = -(
        rho
        * drag_coeff
        * eff_area
        * (vx - wind_speed)
        * np.sqrt(vy**2 + (wind_speed - vx) ** 2)
    ) / (2 * mass_p)
    ay = -g - (
        rho * drag_coeff * eff_area * vy * np.sqrt(vy**2 + (wind_speed - vx) ** 2)
    ) / (2 * mass_p)

    return vx, vy, ax, ay


def projectile_hits_ground_event(
    t,
    y,
    *args,
):
    """Represents the event function to determine when the projectile hits the ground.
    The event occurs when the vertical position of the projectile (py) reaches zero.

    :param t: time variable (not used in this function but required for ODE solvers)

    :param y: tuple containing the state variables:
        (px, py, vx, vy)
        where:
        px: x position of the projectile
        py: y position of the projectile
        vx: x velocity of the projectile
        vy: y velocity of the projectile

    :param args: additional parameters required for the equations:
        (wind_speed, rho, nu, g, projectile)
        where:
        wind_speed: wind speed
        rho: air density
        nu: kinematic viscosity of the air
        g: gravitational acceleration
        projectile: Projectile object containing properties of the projectile

    :return: py: vertical position of the projectile
    """
    _, py, _, _ = y
    return py


def calculate_reynolds_number(
    velocity: float,
    diameter: float,
    air_kinematic_viscosity: float,
) -> float:
    """Calculates the Reynolds number for a (spherical) projectile.

    :param velocity: Velocity of the projectile (m/s)
    :param diameter: Diameter of the projectile (m)
    :param air_kinematic_viscosity: Kinematic viscosity of the air (m**2/s)
    :return: Reynolds number (dimensionless)
    """
    return velocity * diameter / air_kinematic_viscosity
