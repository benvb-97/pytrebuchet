"""PyTrebuchet: A Python package for simulating and optimizing trebuchet designs."""

from pytrebuchet.environment import EnvironmentConfig
from pytrebuchet.optimization import DesignOptimizer
from pytrebuchet.projectile import Projectile
from pytrebuchet.simulation import Simulation, SimulationPhases
from pytrebuchet.trebuchet import Trebuchet

__all__ = [
    "DesignOptimizer",
    "EnvironmentConfig",
    "Projectile",
    "Simulation",
    "SimulationPhases",
    "Trebuchet",
]
