"""Plotting functions for trebuchet launch simulation."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from differential_equations.sling_phase import SlingPhases
from simulation import Simulation, SimulationPhases


def animate_launch(
    simulation: Simulation, skip: int = 5, delay: float = 25, *, show: bool = True
) -> None | animation.FuncAnimation:
    """Animate the trebuchet launch and projectile motion using matplotlib.

    :param simulation: Simulation instance with completed run
    :param skip: Number of frames to skip for faster animation
    :param delay: Delay between frames in milliseconds
    :param show: Whether to display the animation immediately
    :return: None
    """
    if not simulation.solved:
        msg = "Simulation has not been run yet."
        raise ValueError(msg)
    trebuchet = simulation.trebuchet
    projectile = simulation.projectile

    # Create figure
    fig, ax = plt.subplots()

    # Fetch time steps and angles
    tsteps_trebuchet = simulation.get_tsteps(
        sim_phase=SimulationPhases.SLING, sling_phase=SlingPhases.ALL
    )[::skip]
    tsteps_projectile = simulation.get_tsteps(sim_phase=SimulationPhases.ALL)[::skip]

    trebuchet_vars = simulation.get_trebuchet_state_variables()
    angles_arm = trebuchet_vars[:, 0][::skip]
    angles_weight = trebuchet_vars[:, 1][::skip]

    # Calculate trebuchet points
    x_arm_weight, y_arm_weight = trebuchet.calculate_arm_endpoint_weight(angles_arm)
    x_arm_projectile, y_arm_projectile = trebuchet.calculate_arm_endpoint_projectile(
        angles_arm
    )
    x_weight, y_weight = trebuchet.calculate_weight_point(angles_arm, angles_weight)

    # Calculate projectile trajectory
    projectile_vars = simulation.get_projectile_state_variables(
        phase=SimulationPhases.ALL
    )
    x_projectile = projectile_vars[::skip, 0]
    y_projectile = projectile_vars[::skip, 1]

    # Set figure limits
    ax.set_xlim(np.min(x_projectile) - 10.0, np.max(x_projectile) + 10.0)
    ax.set_ylim(-1.0, np.max(y_projectile) + 10.0)

    # Create line plot objects between trebuchet points
    ax.plot([0.0, 0.0], [0.0, trebuchet.pivot.height], c="black")  # pivot line
    (line_arm_projectile,) = ax.plot([], [], c="red")
    (line_arm_weight,) = ax.plot([], [], c="green")
    (line_weight,) = ax.plot([], [], c="blue")
    (line_projectile,) = ax.plot([], [], c="orange")

    # Create circles for weight and projectile
    circle_weight = plt.Circle((0.0, 0.0), projectile.diameter, color="blue", fill=True)
    circle_projectile = plt.Circle(
        (0.0, 0.0), projectile.diameter / 2, color="orange", fill=True
    )
    ax.add_patch(circle_projectile)
    ax.add_patch(circle_weight)

    # Create line plot for projectile trajectory
    ax.plot(
        x_projectile,
        y_projectile,
        linestyle="--",
        color="gray",
        linewidth=0.5,
        label="Projectile Trajectory",
    )

    def update(frame: int) -> animation.FuncAnimation:
        if frame < tsteps_trebuchet.size:  # Animate both trebuchet and projectile
            line_arm_projectile.set_data(
                [0.0, x_arm_projectile[frame]],
                [trebuchet.pivot.height, y_arm_projectile[frame]],
            )
            line_arm_weight.set_data(
                [0.0, x_arm_weight[frame]],
                [trebuchet.pivot.height, y_arm_weight[frame]],
            )
            line_weight.set_data(
                [x_arm_weight[frame], x_weight[frame]],
                [y_arm_weight[frame], y_weight[frame]],
            )
            line_projectile.set_data(
                [x_arm_projectile[frame], x_projectile[frame]],
                [y_arm_projectile[frame], y_projectile[frame]],
            )

            circle_projectile.set_center((x_projectile[frame], y_projectile[frame]))
            circle_weight.set_center((x_weight[frame], y_weight[frame]))

        else:  # Animate only projectile (circle)
            circle_projectile.set_center((x_projectile[frame], y_projectile[frame]))

        return (
            line_arm_weight,
            line_arm_projectile,
            line_weight,
            line_projectile,
            circle_weight,
            circle_projectile,
        )

    ani = animation.FuncAnimation(
        fig, update, frames=tsteps_projectile.size, blit=True, interval=delay
    )

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trebuchet Animation")
    ax.legend()

    if show:
        plt.show()
    else:
        plt.close(fig)  # close the figure to prevent static display

    return ani
