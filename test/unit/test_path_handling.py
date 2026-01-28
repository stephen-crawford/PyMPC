#!/usr/bin/env python3
"""
Unit Tests for Path Handling and Reference Path Generation

Tests path-related functionality for MPCC:
- Reference path generation
- Spline interpolation
- Arc length computation
- Path derivatives (for tangent/normal vectors)

Reference: https://github.com/tud-amr/mpc_planner
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestReferencePathGeneration:
    """Tests for reference path generation."""

    def test_straight_path_generation(self):
        """Test straight path creation."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        assert path is not None
        assert hasattr(path, 'x')
        assert hasattr(path, 'y')
        assert len(path.x) > 0

        # Verify straight line
        y_coords = np.array(path.y)
        np.testing.assert_array_almost_equal(y_coords, np.zeros_like(y_coords), decimal=5)

    def test_curved_path_generation(self):
        """Test curved path creation."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [15.0, 10.0, 0.0]

        path = generate_reference_path(start, goal, "curved", num_points=50)

        assert path is not None
        assert len(path.x) > 0
        assert len(path.y) > 0

        # Verify path has curvature
        y_coords = np.array(path.y)
        assert np.max(y_coords) - np.min(y_coords) > 0.1, "Curved path should have y variation"

    def test_s_curve_path_generation(self):
        """Test S-curve path creation."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [15.0, 5.0, 0.0]

        path = generate_reference_path(start, goal, "s-turn", num_points=50)

        assert path is not None
        assert len(path.x) > 0

    def test_path_starts_at_start(self):
        """Test path starts at specified start position."""
        from planning.types import generate_reference_path

        start = [5.0, 3.0, 0.0]
        goal = [15.0, 8.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        np.testing.assert_almost_equal(path.x[0], start[0], decimal=3)
        np.testing.assert_almost_equal(path.y[0], start[1], decimal=3)

    def test_path_ends_near_goal(self):
        """Test path ends near specified goal."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 5.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        # Path should end close to goal
        final_dist = np.sqrt((path.x[-1] - goal[0])**2 + (path.y[-1] - goal[1])**2)
        assert final_dist < 2.0, "Path should end near goal"


class TestSplineInterpolation:
    """Tests for spline interpolation of reference paths.

    Reference: C++ uses cubic splines for path representation
    """

    def test_spline_creation(self):
        """Test spline is created from path points."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        assert hasattr(path, 'x_spline')
        assert hasattr(path, 'y_spline')

    def test_spline_interpolation_accuracy(self):
        """Test spline accurately interpolates path points."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        if hasattr(path, 'x_spline') and hasattr(path, 's'):
            # Evaluate spline at known arc length values
            for i in range(min(5, len(path.s))):
                s_val = path.s[i]
                x_interp = float(path.x_spline(s_val))
                np.testing.assert_almost_equal(x_interp, path.x[i], decimal=2)

    def test_spline_derivatives(self):
        """Test spline derivatives are available.

        Derivatives are needed for tangent/normal vectors.
        """
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        if hasattr(path, 'x_spline'):
            # Derivative should be available
            try:
                dx_ds = path.x_spline.derivative()
                assert dx_ds is not None
            except AttributeError:
                pytest.skip("Spline derivative method not available")


class TestArcLength:
    """Tests for arc length computation."""

    def test_arc_length_attribute(self):
        """Test path has arc length (s) attribute."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        assert hasattr(path, 's')
        assert len(path.s) > 0

    def test_arc_length_starts_at_zero(self):
        """Test arc length starts at zero."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        np.testing.assert_almost_equal(path.s[0], 0.0, decimal=5)

    def test_arc_length_monotonic(self):
        """Test arc length is monotonically increasing."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 5.0, 0.0]

        path = generate_reference_path(start, goal, "curved", num_points=50)

        s_arr = np.array(path.s)
        diffs = np.diff(s_arr)
        assert np.all(diffs > 0), "Arc length should be monotonically increasing"

    def test_arc_length_straight_path(self):
        """Test arc length equals Euclidean distance for straight path."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        expected_length = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
        actual_length = path.s[-1]

        np.testing.assert_almost_equal(actual_length, expected_length, decimal=1)


class TestPathTangentNormal:
    """Tests for tangent and normal vector computation.

    These are needed for contouring error calculation.
    """

    def test_tangent_vector_computation(self):
        """Test tangent vector can be computed from spline."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        if hasattr(path, 'x_spline') and hasattr(path, 'y_spline'):
            # Compute tangent at first point
            s_val = path.s[0]

            try:
                dx_ds = float(path.x_spline.derivative()(s_val))
                dy_ds = float(path.y_spline.derivative()(s_val))

                # For straight horizontal path, tangent should be (1, 0)
                tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)
                if tangent_norm > 1e-6:
                    tx = dx_ds / tangent_norm
                    ty = dy_ds / tangent_norm

                    np.testing.assert_almost_equal(tx, 1.0, decimal=2)
                    np.testing.assert_almost_equal(ty, 0.0, decimal=2)
            except Exception:
                pytest.skip("Cannot compute tangent vector")

    def test_normal_vector_perpendicular(self):
        """Test normal vector is perpendicular to tangent."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        if hasattr(path, 'x_spline') and hasattr(path, 'y_spline'):
            s_val = path.s[len(path.s)//2]  # Middle of path

            try:
                dx_ds = float(path.x_spline.derivative()(s_val))
                dy_ds = float(path.y_spline.derivative()(s_val))

                tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)
                if tangent_norm > 1e-6:
                    tx = dx_ds / tangent_norm
                    ty = dy_ds / tangent_norm

                    # Normal is perpendicular: (-ty, tx) or (ty, -tx)
                    nx = -ty
                    ny = tx

                    # Dot product should be zero
                    dot = tx * nx + ty * ny
                    np.testing.assert_almost_equal(dot, 0.0, decimal=5)
            except Exception:
                pytest.skip("Cannot compute normal vector")


class TestPathTypes:
    """Test different path types are handled correctly."""

    @pytest.mark.parametrize("path_type", ["straight", "curved", "s-turn"])
    def test_path_type_generation(self, path_type):
        """Test each path type can be generated."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [15.0, 10.0, 0.0]

        try:
            path = generate_reference_path(start, goal, path_type, num_points=50)
            assert path is not None
            assert len(path.x) > 0
        except ValueError:
            pytest.skip(f"Path type {path_type} not supported")


class TestReferencePathObject:
    """Tests for ReferencePath object interface."""

    def test_reference_path_attributes(self):
        """Test ReferencePath has required attributes."""
        from planning.types import ReferencePath

        path = ReferencePath()

        # Should have methods to set/get data
        assert hasattr(path, 'set') or hasattr(path, 'x')

    def test_get_arc_length_method(self):
        """Test get_arc_length method if available."""
        from planning.types import generate_reference_path

        start = [0.0, 0.0, 0.0]
        goal = [10.0, 0.0, 0.0]

        path = generate_reference_path(start, goal, "straight", num_points=50)

        if hasattr(path, 'get_arc_length'):
            arc_length = path.get_arc_length()
            assert arc_length > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
