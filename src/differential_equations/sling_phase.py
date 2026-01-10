"""Ordinary differential equations (ODEs) for the unconstrained sling phase."""

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy import cos, sin

if TYPE_CHECKING:
    from environment import EnvironmentConfig
    from trebuchet import Trebuchet


class SlingPhases(StrEnum):
    """Enumeration for the different sling phases."""

    ALL = "ALL"

    # Projectile is sliding on the ground
    SLIDING_OVER_GROUND = "SLIDING_OVER_GROUND"
    # The projectile and counterweight are in contact with the arm
    # This phase occurs for whipper trebuchets.
    PROJECTILE_AND_COUNTERWEIGHT_CONTACT_ARM = (
        "PROJECTILE_AND_COUNTERWEIGHT_CONTACT_ARM"
    )

    # The projectile is in contact with the arm, but the counterweight is not
    # This phase occurs for whipper trebuchets.
    PROJECTILE_CONTACT_ARM = "PROJECTILE_CONTACT_ARM"

    # Neither the projectile nor the counterweight are in contact with the arm
    UNCONSTRAINED = "UNCONSTRAINED"


def sling_ode(
    t: float,
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    environment: "EnvironmentConfig",
    sling_phase: SlingPhases,
) -> tuple[float, float, float, float, float, float]:
    """Solves the ODEs for a projectile that is still in the sling.

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
    :param environment: EnvironmentConfig object containing environmental parameters
    :param sling_phase: current phase of the sling (from SlingPhases enum)
        This determines the additional constraints to apply to the ODEs.
    :return: derivatives: tuple containing the derivatives of the state variables:
    (dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi)
    """
    _ = t  # time variable not used
    dtheta, dphi, dpsi = y[3], y[4], y[5]

    A, B = _get_ode_matrix(y, trebuchet, environment, sling_phase)  # noqa: N806

    # Solve for accelerations
    x = np.linalg.solve(A, B)
    ddtheta, ddphi, ddpsi = x[0], x[1], x[2]

    return dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi


def _get_ode_matrix(
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    environment: "EnvironmentConfig",
    sling_phase: SlingPhases,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the A and B matrices for the ODEs in the form Ax = B.

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
    :param environment: EnvironmentConfig object containing environmental parameters
    :param sling_phase: current phase of the sling (from SlingPhases enum)
        This determines the additional constraints to apply to the ODEs.
    :return: A, B: matrices for the ODEs in the form Ax = B
    """
    # theta_arm, theta_weight, theta_sling, dtheta_arm, dtheta_weight, dtheta_sling
    theta, phi, psi, dtheta, dphi, dpsi = y
    l1 = trebuchet.arm.length_weight_side
    l2 = trebuchet.arm.length_projectile_side
    l3 = trebuchet.sling_projectile.length
    l4 = trebuchet.sling_weight.length
    la = trebuchet.arm.d_pivot_to_cog
    inertia_a = trebuchet.arm.inertia
    m1 = trebuchet.weight.mass
    m2 = trebuchet.projectile.mass
    ma = trebuchet.arm.mass
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

    # Create unconstrained Ax=B matrix
    A_unconstrained = np.array(  # noqa: N806
        [
            [I0, I14 * cos(theta - phi), -I23 * cos(theta - psi)],
            [I14 * cos(theta - phi), I1, 0],
            [-I23 * cos(theta - psi), 0, I2],
        ]
    )
    B_unconstrained = np.array(  # noqa: N806
        [
            -I14 * dphi**2 * sin(theta - phi)
            + I23 * dpsi**2 * sin(theta - psi)
            - M * g * cos(theta),
            I14 * dtheta**2 * sin(theta - phi) - M14 * g * cos(phi),
            -I23 * dtheta**2 * sin(theta - psi) - M23 * g * cos(psi),
        ]
    )

    # Add constraints based on sling phase
    match sling_phase:
        case SlingPhases.UNCONSTRAINED:
            constraint_eqs_a = []
            constraint_eqs_b = []
        case SlingPhases.SLIDING_OVER_GROUND:
            constraint_eqs_a = [
                np.array([-l2 * cos(theta), 0.0, l3 * cos(psi)]),
            ]
            constraint_eqs_b = [
                -l2 * dtheta**2 * sin(theta) + l3 * dpsi**2 * sin(psi),
            ]
        case SlingPhases.PROJECTILE_AND_COUNTERWEIGHT_CONTACT_ARM:
            constraint_eqs_a = [
                np.array([-1, 1, 0, 0]),
                np.array([-1, 0, 1, 0]),
            ]
            constraint_eqs_b = [
                0.0,
                0.0,
            ]

        case SlingPhases.PROJECTILE_CONTACT_ARM:
            constraint_eqs_a = [
                np.array([-1, 0, 1, 0]),
            ]
            constraint_eqs_b = [
                0.0,
            ]
        case _:
            msg = f"Unknown sling phase: {sling_phase}"
            raise ValueError(msg)

    A = np.zeros((3 + len(constraint_eqs_a), 3 + len(constraint_eqs_a)), dtype=float)  # noqa: N806
    B = np.zeros((3 + len(constraint_eqs_a)), dtype=float)  # noqa: N806
    A[:3, :3] = A_unconstrained
    B[:3] = B_unconstrained

    for i, (eq_a, eq_b) in enumerate(
        zip(constraint_eqs_a, constraint_eqs_b, strict=False)
    ):
        A[3 + i, : len(eq_a)] = eq_a
        A[: len(eq_a), 3 + i] = eq_a
        B[3 + i] = eq_b

    return A, B


# Terminate event functions
def sling_terminate_event(
    t: float,
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    environment: "EnvironmentConfig",
    sling_phase: SlingPhases,
) -> float:
    """Event function to determine when to terminate the ODE integration.

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
    :param environment: EnvironmentConfig object containing environmental parameters
    :param sling_phase: current phase of the sling (from SlingPhases enum)
        This parameter is used to determine the correct event function.

    :return: event_value: difference between the current velocity angle and the desired
    release angle
    """
    _ = t  # time variable not used

    match sling_phase:
        case SlingPhases.SLIDING_OVER_GROUND:
            return _ground_separation_event(y, trebuchet, environment)
        case SlingPhases.PROJECTILE_AND_COUNTERWEIGHT_CONTACT_ARM:
            return _weight_separates_from_arm_event(y, trebuchet, environment)
        case SlingPhases.PROJECTILE_CONTACT_ARM:
            return _projectile_separates_from_arm_event(y, trebuchet, environment)
        case SlingPhases.UNCONSTRAINED:
            return _release_projectile_event(y, trebuchet, environment)
        case _:
            msg = f"Terminate event not defined for sling phase: {sling_phase}"
            raise ValueError(msg)


def _release_projectile_event(
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    environment: "EnvironmentConfig",
) -> float:
    """Event function to determine when to terminate the ODE integration.

    Release happens when the velocity angle matches the desired release angle.

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
    :param environment: EnvironmentConfig object containing environmental parameters

    :return: event_value: difference between the current velocity angle and the desired
    release angle
    """
    # theta_arm, theta_weight, theta_sling, dtheta_arm, dtheta_weight, dtheta_sling
    theta, _, psi, dtheta, _, dpsi = y
    l2 = trebuchet.arm.length_projectile_side
    l3 = trebuchet.sling_projectile.length
    release_angle = trebuchet.release_angle
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
    # if np.sqrt(vx**2 + vy**2) < 10.0:  # prevent release at low speeds
    #     return 1.0  # noqa: ERA001

    return velocity_angle - release_angle


def _ground_separation_event(
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    environment: "EnvironmentConfig",
) -> float:
    """Calculate lagrange multiplier for the ground separation event.

    When the lagrange multiplier becomes zero, the projectile separates from the ground.

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
    :param environment: EnvironmentConfig object containing environmental parameters

    :return: lambd: the lagrange multiplier
    """
    A, B = _get_ode_matrix(y, trebuchet, environment, SlingPhases.SLIDING_OVER_GROUND)  # noqa: N806
    _, _, _, lambd = np.linalg.solve(A, B)

    return lambd


def _projectile_separates_from_arm_event(
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    environment: "EnvironmentConfig",
) -> float:
    """Event function that determines when the projectile separates from the arm.

    The projectile separates when the unconstrained angular accelerations of the
    projectile and arm result in an increasing angle between them.

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
    :param environment: EnvironmentConfig object containing environmental parameters

    :return: lambd: the lagrange multiplier, representing the normal force at the ground
    contact point.
    """
    A, B = _get_ode_matrix(  # noqa: N806
        y,
        trebuchet,
        environment,
        SlingPhases.PROJECTILE_CONTACT_ARM,
    )
    ddtheta, _, ddpsi = np.linalg.solve(A, B)

    return ddpsi - ddtheta


def _weight_separates_from_arm_event(
    y: tuple[float, float, float, float, float, float],
    trebuchet: "Trebuchet",
    environment: "EnvironmentConfig",
) -> float:
    """Event function that determines weight separation.

    The weight separates when the unconstrained angular accelerations of the weight and
    arm result in an increasing angle between them.
    This event function returns the difference in angular accelerations
    (ddphi - ddtheta).

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
    :param environment: EnvironmentConfig object containing environmental parameters

    :return: lambd: the lagrange multiplier, representing the normal force at the
    ground contact point.
    """
    A, B = _get_ode_matrix(  # noqa: N806
        y,
        trebuchet,
        environment,
        SlingPhases.PROJECTILE_AND_COUNTERWEIGHT_CONTACT_ARM,
    )
    ddtheta, ddphi, _ = np.linalg.solve(A, B)

    return ddphi - ddtheta
