"""Module for plotting the initial position of trebuchets."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pytrebuchet.trebuchet import Trebuchet


def plot_initial_position(
    trebuchet: Trebuchet, *, show: bool = True
) -> tuple[Figure, Axes]:
    """Plot the initial position of the trebuchet and the projectile.

    :param trebuchet: Trebuchet instance
    :param show: If True, display the plot immediately.
        Else, just return the figure and axes objects.

    :return: None
    """
    # Create figure
    fig, ax = plt.subplots()
    limits_x, limits_y = _get_trebuchet_limits(trebuchet)
    ax.set_xlim(limits_x[0], limits_x[1])
    ax.set_ylim(limits_y[0], limits_y[1])

    # Calculate trebuchet points
    x_arm_weight, y_arm_weight = trebuchet.calculate_arm_endpoint_weight(
        trebuchet.init_angle_arm
    )
    x_arm_projectile, y_arm_projectile = trebuchet.calculate_arm_endpoint_projectile(
        trebuchet.init_angle_arm
    )
    x_weight, y_weight = trebuchet.calculate_weight_point(
        trebuchet.init_angle_arm, trebuchet.init_angle_weight
    )
    x_projectile, y_projectile = trebuchet.calculate_projectile_point(
        trebuchet.init_angle_arm, trebuchet.init_angle_projectile
    )

    # Plot lines between trebuchet points
    ax.plot([0.0, 0.0], [0.0, trebuchet.pivot.height], c="black")  # pivot line
    ax.plot(
        [0.0, x_arm_projectile], [trebuchet.pivot.height, y_arm_projectile], c="red"
    )
    ax.plot([0.0, x_arm_weight], [trebuchet.pivot.height, y_arm_weight], c="green")
    ax.plot([x_arm_weight, x_weight], [y_arm_weight, y_weight], c="blue")
    ax.plot(
        [x_arm_projectile, x_projectile], [y_arm_projectile, y_projectile], c="orange"
    )

    # Plot weight and projectile as circles
    projectile = trebuchet.projectile
    circle_weight = plt.Circle(
        (x_weight, y_weight), projectile.diameter, color="blue", fill=True
    )
    circle_projectile = plt.Circle(
        (x_projectile, y_projectile), projectile.diameter / 2, color="orange", fill=True
    )
    ax.add_patch(circle_weight)
    ax.add_patch(circle_projectile)

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trebuchet Initial Position")

    if show:
        plt.show()
    return fig, ax


def _get_trebuchet_limits(
    trebuchet: "Trebuchet",
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the trebuchet limits.

    These are the minimum and maximum x- and y- positions that the points
      making up the trebuchet can take for plotting.

    :param trebuchet: Trebuchet instance

    :return: (x_min, x_max), (y_min, y_max)
    """
    projectile_length = (
        trebuchet.arm.length_projectile_side + trebuchet.sling_projectile.length
    )
    weight_length = trebuchet.arm.length_weight_side + trebuchet.sling_weight.length
    max_length = max(weight_length, projectile_length)

    x_min = -max_length
    x_max = max_length
    y_min = 0.0
    y_max = trebuchet.pivot.height + max_length
    return (x_min, x_max), (y_min, y_max)
