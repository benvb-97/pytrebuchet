"""Test launch animation plotting."""

import pytest
from matplotlib.animation import FuncAnimation

from plotting.launch import animate_launch
from simulation import Simulation
from trebuchet import HingedCounterweightTrebuchet


@pytest.fixture(scope="module")
def simulation() -> Simulation:
    """Create a sample simulation for testing."""
    trebuchet = HingedCounterweightTrebuchet.default()
    simulation = Simulation(trebuchet)

    simulation.solve()
    return simulation


def test_animate_launch(simulation: Simulation) -> None:
    """Test the launch animation plotting function."""
    try:
        ani = animate_launch(simulation=simulation, skip=10, delay=50, show=False)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"animate_launch raised an exception: {e}")

    if not isinstance(ani, FuncAnimation):
        msg = f"animate_launch did not return a FuncAnimation instance. Got {ani}."
        pytest.fail(msg)
