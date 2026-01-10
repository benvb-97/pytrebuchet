"""Unit tests for the Projectile class in pytrebuchet."""

from drag_coefficient import clift_grace_weber
from projectile import Projectile


def test_init_with_constant_drag_coefficient() -> None:
    """Test that a constant drag coefficient is accepted."""
    projectile = Projectile(mass=5.0, diameter=0.3, drag_coefficient=0.47)
    assert projectile.mass == 5.0
    assert projectile.diameter == 0.3
    assert projectile.drag_coefficient(1000) == 0.47
    assert projectile.drag_coefficient(5000) == 0.47


def test_init_with_callable_drag_coefficient() -> None:
    """Test that a callable drag coefficient function is accepted."""

    def custom_drag(reynolds: float) -> float:
        return 0.5 if reynolds < 1000 else 0.3

    projectile = Projectile(mass=3.0, diameter=0.25, drag_coefficient=custom_drag)
    assert projectile.mass == 3.0
    assert projectile.diameter == 0.25
    assert projectile.drag_coefficient(500) == 0.5
    assert projectile.drag_coefficient(2000) == 0.3


def test_init_with_default_drag_coefficient() -> None:
    """Test that default drag coefficient function is used when none is provided."""
    projectile = Projectile(mass=4.0, diameter=0.35, drag_coefficient=None)
    assert projectile.mass == 4.0
    assert projectile.diameter == 0.35
    assert projectile.drag_coefficient == clift_grace_weber
