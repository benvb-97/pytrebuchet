import pytest
from math import pi
from pytrebuchet.projectile import Projectile

from pytrebuchet.drag_coefficient import (
    drag_coefficient_smooth_sphere_clift_grace_weber,
)


class TestProjectile:
    def test_init_with_constant_drag_coefficient(self):
        projectile = Projectile(mass=5.0, diameter=0.3, drag_coefficient=0.47)
        assert projectile.mass == 5.0
        assert projectile.diameter == 0.3
        assert projectile.drag_coefficient(1000) == 0.47
        assert projectile.drag_coefficient(5000) == 0.47

    def test_init_with_callable_drag_coefficient(self):
        def custom_drag(reynolds):
            return 0.5 if reynolds < 1000 else 0.3

        projectile = Projectile(mass=3.0, diameter=0.25, drag_coefficient=custom_drag)
        assert projectile.mass == 3.0
        assert projectile.diameter == 0.25
        assert projectile.drag_coefficient(500) == 0.5
        assert projectile.drag_coefficient(2000) == 0.3

    def test_init_with_default_drag_coefficient(self):
        projectile = Projectile(mass=4.0, diameter=0.35, drag_coefficient=None)
        assert projectile.mass == 4.0
        assert projectile.diameter == 0.35
        assert (
            projectile.drag_coefficient
            == drag_coefficient_smooth_sphere_clift_grace_weber
        )
