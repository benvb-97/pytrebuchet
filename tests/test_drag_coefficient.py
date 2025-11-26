"""Tests for drag coefficient functions in pytrebuchet.drag_coefficient module."""

import warnings

import numpy as np
import pytest

from pytrebuchet.drag_coefficient import (
    drag_coefficient_smooth_sphere_clift_grace_weber,
    drag_coefficient_smooth_sphere_morrison,
)


class TestDragCoefficientCliftGraceWeber:
    """Tests for drag_coefficient_smooth_sphere_clift_grace_weber function."""

    def test_float_input(self) -> None:
        """Test that float input returns correct float output."""
        result = drag_coefficient_smooth_sphere_clift_grace_weber(100.0)
        assert isinstance(result, float)
        assert result == pytest.approx(1.087017, abs=1e-3)

    def test_array_input(self) -> None:
        """Test that array input returns correct array output."""
        result = drag_coefficient_smooth_sphere_clift_grace_weber(
            np.array([100.0, 200.0])
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result == pytest.approx([1.087017, 0.775634], abs=1e-3)


class TestDragCoefficientMorrison:
    """Tests for drag_coefficient_smooth_sphere_morrison function."""

    def test_float_input(self) -> None:
        """Test that float input returns correct float output."""
        result = drag_coefficient_smooth_sphere_morrison(100.0)
        assert isinstance(result, (float, np.floating))
        assert result == pytest.approx(1.0381866, abs=1e-3)

    def test_array_input_returns_array(self) -> None:
        """Test that array input returns correct array output."""
        result = drag_coefficient_smooth_sphere_morrison(np.array([100.0, 200.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result == pytest.approx([1.0381866, 0.76773156], abs=1e-3)

    def test_out_of_bounds_warning(self) -> None:
        """Test that out-of-bounds Reynolds numbers raise a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            drag_coefficient_smooth_sphere_morrison(1e8)
            drag_coefficient_smooth_sphere_morrison(5e-4)
            assert len(w) == 2
            assert issubclass(w[0].category, UserWarning)
            assert "out of range" in str(w[0].message)
            assert issubclass(w[1].category, UserWarning)
            assert "out of range" in str(w[1].message)
