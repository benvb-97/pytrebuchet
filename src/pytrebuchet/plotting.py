import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pytrebuchet import Trebuchet, Projectile, Simulation
import numpy as np


def plot_initial_position(trebuchet: Trebuchet, projectile: Projectile) -> None:
    """
    Plots the initial position of the trebuchet and the projectile.
    :param trebuchet: Trebuchet instance
    :param projectile: Projectile instance
    :return: None
    """

    # Create figure
    fig, ax = plt.subplots()
    limits_x, limits_y = trebuchet.get_limits()
    ax.set_xlim(limits_x[0], limits_x[1])
    ax.set_ylim(limits_y[0], limits_y[1])

    # Calculate trebuchet points
    x_arm_weight, y_arm_weight = trebuchet.calculate_arm_endpoint_weight(trebuchet.init_angle_arm)
    x_arm_projectile, y_arm_projectile = trebuchet.calculate_arm_endpoint_projectile(trebuchet.init_angle_arm)
    x_weight, y_weight = trebuchet.calculate_weight_point(trebuchet.init_angle_arm, trebuchet.init_angle_weight)
    x_projectile, y_projectile = trebuchet.calculate_projectile_point(trebuchet.init_angle_arm, trebuchet.init_angle_projectile)

    # Plot lines between trebuchet points
    line_pivot = ax.plot([0.0, 0.0], [0.0, trebuchet.h_pivot], c="black")
    line_arm_projectile, = ax.plot([0.0, x_arm_projectile], [trebuchet.h_pivot, y_arm_projectile], c="red")
    line_arm_weight, = ax.plot([0.0, x_arm_weight], [trebuchet.h_pivot, y_arm_weight], c="green")
    line_weight, = ax.plot([x_arm_weight, x_weight], [y_arm_weight, y_weight], c="blue")
    line_projectile, = ax.plot([x_arm_projectile, x_projectile], [y_arm_projectile, y_projectile], c="orange")

    # Plot weight and projectile as circles
    circle_weight = plt.Circle((x_weight, y_weight), projectile.diameter, color='blue', fill=True)
    circle_projectile = plt.Circle((x_projectile, y_projectile), projectile.diameter / 2, color='orange', fill=True)
    ax.add_patch(circle_weight)
    ax.add_patch(circle_projectile)  

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trebuchet Initial Position")

    plt.show()


def animate_launch(simulation: Simulation, skip: int = 5, delay: float = 25, show=True) -> None:
    """
    Animates the trebuchet launch and projectile motion using matplotlib.
    :param simulation: Simulation instance with completed run
    :param skip: Number of frames to skip for faster animation
    :param delay: Delay between frames in milliseconds
    :param show: Whether to display the animation immediately
    :return: None
    """

    if not simulation.solved:
        raise ValueError("Simulation has not been run yet.")
    trebuchet = simulation.trebuchet
    projectile = simulation.projectile

    # Create figure
    fig, ax = plt.subplots()

    # Fetch time steps and angles
    tsteps_trebuchet = simulation.tsteps_trebuchet[::skip]
    tsteps_projectile = simulation.tsteps_projectile[::skip]

    angles_arm, angles_weight, angles_projectile = simulation.angles_trebuchet
    angles_arm = angles_arm[::skip]
    angles_weight = angles_weight[::skip]
    angles_projectile = angles_projectile[::skip]

    # Calculate trebuchet points
    x_arm_weight, y_arm_weight = trebuchet.calculate_arm_endpoint_weight(angles_arm)
    x_arm_projectile, y_arm_projectile = trebuchet.calculate_arm_endpoint_projectile(angles_arm)
    x_weight, y_weight = trebuchet.calculate_weight_point(angles_arm, angles_weight)

    # Calculate projectile trajectory
    x_projectile, y_projectile = simulation.projectile_trajectory
    x_projectile = x_projectile[::skip]
    y_projectile = y_projectile[::skip]

    # Set figure limits
    ax.set_xlim(np.min(x_projectile)-1.0, np.max(x_projectile)+1.0)
    ax.set_ylim(-1., np.max(y_projectile)+1.0)
    
    # Create line plot objects between trebuchet points
    line_pivot = ax.plot([0.0, 0.0], [0.0, trebuchet.h_pivot], c="black")
    line_arm_projectile, = ax.plot([], [], c="red")
    line_arm_weight, = ax.plot([], [], c="green")
    line_weight, = ax.plot([], [], c="blue")
    line_projectile, = ax.plot([], [], c="orange")

    # Create circles for weight and projectile
    circle_weight = plt.Circle((0.0, 0.0), projectile.diameter, color='blue', fill=True)
    circle_projectile = plt.Circle((0.0, 0.0), projectile.diameter / 2, color='orange', fill=True)
    ax.add_patch(circle_projectile)
    ax.add_patch(circle_weight)

    # Create line plot for projectile trajectory
    ax.plot(x_projectile, y_projectile, linestyle='--', color='gray', linewidth=0.5, label='Projectile Trajectory')

    def update(frame):

        if frame < tsteps_trebuchet.size:  # Animate both trebuchet and projectile
            line_arm_projectile.set_data([0.0, x_arm_projectile[frame]], [trebuchet.h_pivot, y_arm_projectile[frame]])
            line_arm_weight.set_data([0.0, x_arm_weight[frame]], [trebuchet.h_pivot, y_arm_weight[frame]])
            line_weight.set_data([x_arm_weight[frame], x_weight[frame]], [y_arm_weight[frame], y_weight[frame]])
            line_projectile.set_data([x_arm_projectile[frame], x_projectile[frame]], [y_arm_projectile[frame], y_projectile[frame]])

            circle_projectile.set_center((x_projectile[frame], y_projectile[frame]))
            circle_weight.set_center((x_weight[frame], y_weight[frame]))

        else:  # Animate only projectile (circle)
            circle_projectile.set_center((x_projectile[frame], y_projectile[frame]))

        return line_arm_weight, line_arm_projectile, line_weight, line_projectile, circle_weight, circle_projectile

    ani = animation.FuncAnimation(fig, update, frames=tsteps_projectile.size, blit=True, interval=delay)

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trebuchet Animation")
    ax.legend()

    if show:
        plt.show()

    return ani