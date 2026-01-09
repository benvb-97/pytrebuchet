"""PyTrebuchet: a Python package for simulating trebuchet mechanics."""

from environment import EnvironmentConfig
from physical_constants import PhysicalConstants
from projectile import Projectile
from simulation import Simulation, SimulationPhases
from trebuchet import HingedCounterweightTrebuchet, WhipperTrebuchet

__all__ = [
    "EnvironmentConfig",
    "HingedCounterweightTrebuchet",
    "PhysicalConstants",
    "Projectile",
    "Simulation",
    "SimulationPhases",
    "WhipperTrebuchet",
]
