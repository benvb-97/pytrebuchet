"""Module containing the differential equations for a sliding projectile."""

import numpy as np
from numpy import cos, sin


def sliding_projectile_ode(
    t: float,
    y: tuple[float, float, float, float, float, float],
    *args: tuple,
) -> tuple[float, float, float, float, float, float]:
    """Ordinary differential equations (ODEs) for a projectile sliding over the ground.

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
    :param args: additional parameters required for the equations:
        (l1, l2, l3, l4, la, Ia, m1, m2, ma, g)
        where:
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
        release_angle: angle at which the projectile is released

    :return: derivatives:
    tuple containing the derivatives of the state variables:
     (dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi)
    """
    _ = t  # Unused variable

    # Fetch variables, arm, weight, sling
    theta, phi, psi, dtheta, dphi, dpsi = y
    l1, l2, l3, l4, la, inertia_a, m1, m2, ma, g, _ = args

    # Calculate terms
    I0 = m1 * l1**2 + m2 * l2**2 + ma * la**2 + inertia_a  # noqa: N806
    I1 = m1 * l4**2  # noqa: N806
    I2 = m2 * l3**2  # noqa: N806
    I14 = m1 * l1 * l4  # noqa: N806
    I23 = m2 * l2 * l3  # noqa: N806

    M = m1 * l1 - m2 * l2 - ma * la  # noqa: N806
    M14 = m1 * l4  # noqa: N806
    M23 = m2 * l3  # noqa: N806

    gamma = -l2 * dtheta**2 * sin(theta) + l3 * dpsi**2 * sin(psi)

    # Create Ax=B matrix
    A = np.array(  # noqa: N806
        [
            [I0, I14 * cos(theta - phi), -I23 * cos(theta - psi), -l2 * cos(theta)],
            [I14 * cos(theta - phi), I1, 0, 0],
            [-I23 * cos(theta - psi), 0, I2, l3 * cos(psi)],
            [-l2 * cos(theta), 0, l3 * cos(psi), 0],
        ]
    )
    B = np.array(  # noqa: N806
        [
            -I14 * dphi**2 * sin(theta - phi)
            + I23 * dpsi**2 * sin(theta - psi)
            - M * g * cos(theta),
            I14 * dtheta**2 * sin(theta - phi) - M14 * g * cos(phi),
            -I23 * dtheta**2 * sin(theta - psi) - M23 * g * cos(psi),
            gamma,
        ]
    )

    ddtheta, ddphi, ddpsi, _ = np.linalg.solve(A, B)

    return dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi


def ground_separation_event(
    t: float,
    y: tuple[float, float, float, float, float, float],
    *args: tuple,
) -> float:
    """Calculate lagrange multiplier for the ground separation event.

    When the lagrange multiplier becomes zero, the projectile separates from the ground.

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
    :param args: additional parameters required for the equations:
        (l1, l2, l3, l4, la, Ia, m1, m2, ma, g)
        where:
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
        release_angle: angle at which the projectile is released

    :return: lambd: the lagrange multiplier
    """
    _ = t  # Unused variable

    # Fetch variables: arm, weight, sling
    theta, phi, psi, dtheta, dphi, dpsi = y
    l1, l2, l3, l4, la, Ia, m1, m2, ma, g, _ = args  # noqa: N806

    # Calculate terms
    I0 = m1 * l1**2 + m2 * l2**2 + ma * la**2 + Ia  # noqa: N806
    I1 = m1 * l4**2  # noqa: N806
    I2 = m2 * l3**2  # noqa: N806
    I14 = m1 * l1 * l4  # noqa: N806
    I23 = m2 * l2 * l3  # noqa: N806

    M = m1 * l1 - m2 * l2 - ma * la  # noqa: N806
    M14 = m1 * l4  # noqa: N806
    M23 = m2 * l3  # noqa: N806

    gamma = -l2 * dtheta**2 * sin(theta) + l3 * dpsi**2 * sin(psi)

    # Create Ax=B matrix
    A = np.array(  # noqa: N806
        [
            [I0, I14 * cos(theta - phi), -I23 * cos(theta - psi), -l2 * cos(theta)],
            [I14 * cos(theta - phi), I1, 0, 0],
            [-I23 * cos(theta - psi), 0, I2, l3 * cos(psi)],
            [-l2 * cos(theta), 0, l3 * cos(psi), 0],
        ]
    )
    B = np.array(  # noqa: N806
        [
            -I14 * dphi**2 * sin(theta - phi)
            + I23 * dpsi**2 * sin(theta - psi)
            - M * g * cos(theta),
            I14 * dtheta**2 * sin(theta - phi) - M14 * g * cos(phi),
            -I23 * dtheta**2 * sin(theta - psi) - M23 * g * cos(psi),
            gamma,
        ]
    )

    _, _, _, lambd = np.linalg.solve(A, B)

    return lambd
