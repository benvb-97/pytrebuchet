"""PyTrebuchet: a Python package for simulating trebuchet mechanics."""

from pytrebuchet.environment import EnvironmentConfig
from pytrebuchet.physical_constants import PhysicalConstants
from pytrebuchet.projectile import Projectile
from pytrebuchet.simulation import Simulation, SimulationPhases
from pytrebuchet.trebuchet import HingedCounterweightTrebuchet, WhipperTrebuchet

__all__ = [
    "EnvironmentConfig",
    "HingedCounterweightTrebuchet",
    "PhysicalConstants",
    "Projectile",
    "Simulation",
    "SimulationPhases",
    "WhipperTrebuchet",
]
