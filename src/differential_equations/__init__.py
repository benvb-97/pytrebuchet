"""Module containing differential equations that describe trebuchet motion."""

from .ballistic_phase import ballistic_ode, projectile_hits_ground_event
from .sling_phase import SlingPhases, sling_ode, sling_terminate_event

__all__ = [
    "SlingPhases",
    "ballistic_ode",
    "projectile_hits_ground_event",
    "sling_ode",
    "sling_terminate_event",
]
