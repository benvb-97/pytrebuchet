"""Ordinary differential equations (ODEs) for the unconstrained sling phase."""

from typing import TYPE_CHECKING

import numpy as np
from numpy import cos, sin

if TYPE_CHECKING:
    from pytrebuchet.environment import EnvironmentConfig
    from pytrebuchet.projectile import Projectile
    from pytrebuchet.trebuchet import Trebuchet


def sling_projectile_ode(
    t: float,
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    projectile: "Projectile",
    environment: "EnvironmentConfig",
) -> tuple[float, float, float, float, float, float]:
    """Solves the ODEs for an unconstrained projectile, but still in the sling.

    :param t: time variable (not used in this function but required for ODE solvers)
    :param y: tuple containing the state variables:
        (theta, phi, psi, dtheta, dphi, dpsi)
        where:
        theta: angle of the arm
        phi: angle of the weight
        psi: angle of the projectile
        dtheta: angular velocity of the arm
        dphi: angular velocity of the weight
        dpsi: angular velocity of the projectile
    :param trebuchet: Trebuchet object containing trebuchet parameters
    :param projectile: Projectile object containing projectile parameters
    :param environment: EnvironmentConfig object containing environmental parameters

    :return: derivatives: tuple containing the derivatives of the state variables:
    (dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi)
    """
    # Fetch variables
    _ = t  # time variable not used
    # theta_arm, theta_weight, theta_sling, dtheta_arm, dtheta_weight, dtheta_sling
    theta, phi, psi, dtheta, dphi, dpsi = y
    l1 = trebuchet.l_weight_arm
    l2 = trebuchet.l_projectile_arm
    l3 = trebuchet.l_sling_projectile
    l4 = trebuchet.l_sling_weight
    la = trebuchet.d_pivot_to_arm_cog
    inertia_a = trebuchet.inertia_arm
    m1 = trebuchet.mass_weight
    m2 = projectile.mass
    ma = trebuchet.mass_arm
    g = environment.gravitational_acceleration

    # Calculate terms
    I0 = m1 * l1**2 + m2 * l2**2 + ma * la**2 + inertia_a  # noqa: N806
    I1 = m1 * l4**2  # noqa: N806
    I2 = m2 * l3**2  # noqa: N806
    I14 = m1 * l1 * l4  # noqa: N806
    I23 = m2 * l2 * l3  # noqa: N806

    M = m1 * l1 - m2 * l2 - ma * la  # noqa: N806
    M14 = m1 * l4  # noqa: N806
    M23 = m2 * l3  # noqa: N806

    # Create Ax=B matrix
    A = np.array(  # noqa: N806
        [
            [I0, I14 * cos(theta - phi), -I23 * cos(theta - psi)],
            [I14 * cos(theta - phi), I1, 0],
            [-I23 * cos(theta - psi), 0, I2],
        ]
    )
    B = np.array(  # noqa: N806
        [
            -I14 * dphi**2 * sin(theta - phi)
            + I23 * dpsi**2 * sin(theta - psi)
            - M * g * cos(theta),
            I14 * dtheta**2 * sin(theta - phi) - M14 * g * cos(phi),
            -I23 * dtheta**2 * sin(theta - psi) - M23 * g * cos(psi),
        ]
    )

    ddtheta, ddphi, ddpsi = np.linalg.solve(A, B)

    return dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi


def projectile_release_event(
    t: float,
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    projectile: "Projectile",
    environment: "EnvironmentConfig",
) -> float:
    """Event function to determine when the projectile releases from the sling.

    Release happens when the velocity angle matches the desired release angle.

    :param t: time variable (not used in this function but required for ODE solvers)
    :param y: tuple containing the state variables:
        (theta, phi, psi, dtheta, dphi, dpsi)
        where:
        theta: angle of the arm
        phi: angle of the weight
        psi: angle of the projectile
        dtheta: angular velocity of the arm
        dphi: angular velocity of the weight
        dpsi: angular velocity of the projectile
    :param trebuchet: Trebuchet object containing trebuchet parameters
    :param projectile: Projectile object containing projectile parameters
    :param environment: EnvironmentConfig object containing environmental parameters

    :return: event_value: difference between the current velocity angle and the desired
    release angle
    """
    # Fetch variables
    _ = t  # time variable not used
    # theta_arm, theta_weight, theta_sling, dtheta_arm, dtheta_weight, dtheta_sling
    theta, _, psi, dtheta, _, dpsi = y
    l2 = trebuchet.l_projectile_arm
    l3 = trebuchet.l_sling_projectile
    release_angle = trebuchet.release_angle
    _ = projectile  # not used
    _ = environment  # not used

    # Calculate velocity of the projectile
    vx = l2 * dtheta * sin(theta) - l3 * dpsi * sin(psi)
    vy = -l2 * dtheta * cos(theta) + l3 * dpsi * cos(psi)

    # Calculate velocity angle
    velocity_angle = np.arctan2(vy, vx)

    if vx < 0:  # projectile is moving backwards, avoid false release
        return 1.0
    if vy < 0:  # projectile is moving downwards, avoid false release
        return 1.0

    return velocity_angle - release_angle
