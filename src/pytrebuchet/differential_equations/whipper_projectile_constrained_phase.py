import numpy as np
from numpy import cos, sin


def whipper_projectile_constrained_ode(
    t,
    y,
    *args,
):
    """Represents the ordinary differential equations (ODEs) for a whipper trebuchet with the projectile constrained by the arm.
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
        release_angle: angle at which the sling releases the projectile

    :return: derivatives: tuple containing the derivatives of the state variables: (dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi)
    """
    # Fetch variables
    theta, phi, psi, dtheta, dphi, dpsi = (
        y  # theta_arm, theta_weight, theta_sling, dtheta_arm, dtheta_weight, dtheta_sling
    )
    l1, l2, l3, l4, la, Ia, m1, m2, ma, g, _ = args

    # Calculate terms
    I0 = m1 * l1**2 + m2 * l2**2 + ma * la**2 + Ia
    I1 = m1 * l4**2
    I2 = m2 * l3**2
    I14 = m1 * l1 * l4
    I23 = m2 * l2 * l3

    M = m1 * l1 - m2 * l2 - ma * la
    M14 = m1 * l4
    M23 = m2 * l3

    # weight constraint
    # gamma = 0 # l1*l4*(6*l1**2*l4**2*(-dphi + dtheta)*sin(phi - theta)**3 + 3*l1*l4*(-dphi + dtheta)*(l1**2 + 2*l1*l4*cos(phi - theta) + l4**2)*sin(2*phi - 2*theta) + 2*(dphi - dtheta)*(l1**2 + 2*l1*l4*cos(phi - theta) + l4**2)**2*sin(phi - theta))/(l1**2 + 2*l1*l4*cos(phi - theta) + l4**2)**(5/2)
    # c_theta = l1*l4*(-l1*l4*(-dphi + dtheta)*sin(phi - theta)**2 + (dphi - dtheta)*(l1**2 + 2*l1*l4*cos(phi - theta) + l4**2)*cos(phi - theta))/(l1**2 + 2*l1*l4*cos(phi - theta) + l4**2)**(3/2)
    # c_phi = l1*l4*(l1*l4*sin(phi - theta)**2 + (l1**2 + 2*l1*l4*cos(phi - theta) + l4**2)*cos(phi - theta))*(-dphi + dtheta)/(l1**2 + 2*l1*l4*cos(phi - theta) + l4**2)**(3/2)

    # Create Ax=B matrix
    A = np.array(
        [
            [I0, I14 * cos(theta - phi), -I23 * cos(theta - psi), -1],
            [I14 * cos(theta - phi), I1, 0, 0],
            [-I23 * cos(theta - psi), 0, I2, 1],
            [-1, 0, 1, 0],
        ]
    )
    B = np.array(
        [
            -I14 * dphi**2 * sin(theta - phi)
            + I23 * dpsi**2 * sin(theta - psi)
            - M * g * cos(theta),
            I14 * dtheta**2 * sin(theta - phi) - M14 * g * cos(phi),
            -I23 * dtheta**2 * sin(theta - psi) - M23 * g * cos(psi),
            0,
        ]
    )

    ddtheta, ddphi, ddpsi, _ = np.linalg.solve(A, B)

    return dtheta, dphi, dpsi, ddtheta, ddphi, ddpsi


def whipper_projectile_separation_event(
    t,
    y,
    *args,
):
    """Projectile separates when the unconstrained angular accelerations of the projectile and arm result in an increasing angle between them.
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
        release_angle: angle at which the sling releases the projectile

    :return: lambd: the lagrange multiplier, representing the normal force at the ground contact point
    """
    # Fetch variables
    theta, phi, psi, dtheta, dphi, dpsi = (
        y  # theta_arm, theta_weight, theta_sling, dtheta_arm, dtheta_weight, dtheta_sling
    )
    l1, l2, l3, l4, la, Ia, m1, m2, ma, g, _ = args

    # Calculate terms
    I0 = m1 * l1**2 + m2 * l2**2 + ma * la**2 + Ia
    I1 = m1 * l4**2
    I2 = m2 * l3**2
    I14 = m1 * l1 * l4
    I23 = m2 * l2 * l3

    M = m1 * l1 - m2 * l2 - ma * la
    M14 = m1 * l4
    M23 = m2 * l3

    # Create Ax=B matrix
    A = np.array(
        [
            [I0, I14 * cos(theta - phi), -I23 * cos(theta - psi)],
            [I14 * cos(theta - phi), I1, 0],
            [-I23 * cos(theta - psi), 0, I2],
        ]
    )
    B = np.array(
        [
            -I14 * dphi**2 * sin(theta - phi)
            + I23 * dpsi**2 * sin(theta - psi)
            - M * g * cos(theta),
            I14 * dtheta**2 * sin(theta - phi) - M14 * g * cos(phi),
            -I23 * dtheta**2 * sin(theta - psi) - M23 * g * cos(psi),
        ]
    )

    ddtheta, ddphi, ddpsi = np.linalg.solve(A, B)

    return ddpsi - ddtheta
