import unittest
import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
from pathlib import Path

# Import your spline implementations
from utils.math_utils import TkSpline, Spline, TwoDimensionalSpline, FourDimensionalSpline, Clothoid2D

class TestTkSpline(unittest.TestCase):
    """Tests for the TkSpline class"""

    def setUp(self):
        # Create standard test data
        self.x = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.y = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.spline = TkSpline()
        self.spline.set_points(self.x, self.y)

    def test_initialization(self):
        """Test spline initialization"""
        self.assertEqual(len(self.spline.m_x), len(self.x))
        self.assertEqual(len(self.spline.m_y), len(self.y))
        self.assertEqual(len(self.spline.m_a), len(self.x))
        self.assertEqual(len(self.spline.m_b), len(self.x))
        self.assertEqual(len(self.spline.m_c), len(self.x))
        self.assertEqual(len(self.spline.m_d), len(self.x))


    def test_interpolation_at_knots(self):
        """Test if spline passes through the knot points"""
        for i in range(len(self.x)):
            self.assertAlmostEqual(self.spline(self.x[i]), self.y[i])

    def test_interpolation_between_knots(self):
        """Test interpolation between knot points"""
        # Test some points between knots
        x_test = 0.5
        y_interp = self.spline(x_test)
        # Value should be between the surrounding knot values
        self.assertTrue(0.0 <= y_interp <= 1.0)

    def test_extrapolation(self):
        """Test extrapolation behavior"""
        # Test point before first knot
        x_before = -1.0
        y_before = self.spline(x_before)
        # Test point after last knot
        x_after = 5.0
        y_after = self.spline(x_after)
        # Verify extrapolation works
        self.assertIsNotNone(y_before)
        self.assertIsNotNone(y_after)

    def test_first_derivative(self):
        """Test first derivative"""
        # First derivative at x=0.5 should be positive
        deriv1 = self.spline.deriv(1, 0.5)
        self.assertTrue(deriv1 > 0)

        # First derivative at x=1.5 should be negative
        deriv1 = self.spline.deriv(1, 1.5)
        self.assertTrue(deriv1 < 0)

    def test_second_derivative(self):
        """Test second derivative"""
        deriv2 = self.spline.deriv(2, 1.0)
        # At x=1.0, the curve changes from increasing to decreasing,
        # so second derivative should be negative
        self.assertTrue(deriv2 < 0)

    def test_third_derivative(self):
        """Test third derivative"""
        deriv3 = self.spline.deriv(3, 1.5)
        # Third derivative should be constant for cubic spline segments
        self.assertIsNotNone(deriv3)

    def test_higher_order_derivatives(self):
        """Test higher-order derivatives (should be zero)"""
        deriv4 = self.spline.deriv(4, 1.0)
        self.assertEqual(deriv4, 0.0)

    def test_linear_spline(self):
        """Test linear spline option"""
        linear_spline = TkSpline()
        linear_spline.set_points(self.x, self.y, cubic_spline=False)

        # For linear spline, the value at midpoint should be average of endpoints
        for i in range(len(self.x) - 1):
            mid_x = (self.x[i] + self.x[i + 1]) / 2
            expected_y = (self.y[i] + self.y[i + 1]) / 2
            self.assertAlmostEqual(linear_spline(mid_x), expected_y)

    def test_get_parameters(self):
        """Test retrieving polynomial coefficients"""
        a, b, c, d = None, None, None, None
        a, b, c, d = self.spline.get_parameters(0)

        # Verify parameters are returned
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertIsNotNone(c)
        self.assertIsNotNone(d)


class TestTwoDimensionalSpline(unittest.TestCase):
    """Tests for the TwoDimensionalSpline class"""

    def setUp(self):
        # Create a circle-like path
        self.t = np.linspace(0, 2 * pi, 20)
        self.x = np.cos(self.t)
        self.y = np.sin(self.t)
        self.spline = TwoDimensionalSpline(self.x, self.y, self.t)

    def test_initialization(self):
        """Test spline initialization"""
        self.assertEqual(len(self.spline.x), len(self.x))
        self.assertEqual(len(self.spline.y), len(self.y))
        self.assertEqual(len(self.spline.t_vector), len(self.t))
        self.assertGreater(len(self.spline.s_vector), 0)

    def test_coordinate_access(self):
        """Test individual coordinate access"""
        t_test = 1.0
        self.assertAlmostEqual(self.spline.get_x(t_test), self.spline.get_point(t_test)[0])
        self.assertAlmostEqual(self.spline.get_y(t_test), self.spline.get_point(t_test)[1])

    def test_derivatives(self):
        """Test velocity, acceleration, and jerk"""
        t_test = 1.0

        velocity = self.spline.get_velocity(t_test)
        self.assertEqual(len(velocity), 2)

        acceleration = self.spline.get_acceleration(t_test)
        self.assertEqual(len(acceleration), 2)

        jerk = self.spline.get_jerk(t_test)
        self.assertEqual(len(jerk), 2)

    def test_segment_count(self):
        self.assertEqual(self.spline.get_num_segments(), 19)

    def test_geometric_properties(self):
        """Test geometric properties of the curve"""
        t_test = pi / 2  # At t=pi/2, we're at (0,1) on the circle

        # Curvature at points on the circle should be close to 1
        curvature = self.spline.get_curvature(t_test)
        self.assertAlmostEqual(abs(curvature), 1.0, places=1)

        # Orthogonal vector at (0,1) should point toward (0,0), i.e., (0,-1)
        orthogonal = self.spline.get_orthogonal(t_test)
        self.assertAlmostEqual(orthogonal[0], 0.0, places=1)
        self.assertAlmostEqual(orthogonal[1], -1.0, places=1)

        # Path angle at (0,1) should be pi
        angle = self.spline.get_path_angle(t_test)
        self.assertAlmostEqual(angle, pi, places=1)

    def test_closest_point(self):
        """Test finding closest point on the spline"""
        # Point near the spline
        test_point = np.array([0.5, 0.5])
        segment, t_val = self.spline.find_closest_point(test_point)

        # Check that segment is valid
        self.assertTrue(0 <= segment < self.spline.get_num_segments())
        # Check that t_val is within spline parameter range
        self.assertTrue(self.spline.t_vector[0] <= t_val <= self.spline.t_vector[-1])

        # Point far from the spline
        test_point = np.array([10.0, 10.0])
        segment, t_val = self.spline.find_closest_point(test_point)

        # Check that segment is valid
        self.assertTrue(0 <= segment < self.spline.get_num_segments())
        # Check that t_val is within spline parameter range
        self.assertTrue(self.spline.t_vector[0] <= t_val <= self.spline.t_vector[-1])

    def test_sampling(self):
        """Test point sampling along the spline"""
        # Sample with a small distance
        ds = 0.1
        points, angles = self.spline.sample_points(ds)

        # Verify we have points and angles
        self.assertGreater(len(points), 1)
        self.assertEqual(len(points), len(angles))

        # Check that points are reasonably spaced
        for i in range(len(points) - 1):
            dist = np.linalg.norm(points[i + 1] - points[i])
            self.assertLessEqual(dist, ds * 1.1)  # Allow for small numerical errors

    def test_get_parameters(self):
        """Test retrieving polynomial coefficients"""
        ax, bx, cx, dx, ay, by, cy, dy = self.spline.get_parameters(0, None, None, None, None, None, None, None, None)

        # Verify parameters are returned
        self.assertIsNotNone(ax)
        self.assertIsNotNone(bx)
        self.assertIsNotNone(cx)
        self.assertIsNotNone(dx)
        self.assertIsNotNone(ay)
        self.assertIsNotNone(by)
        self.assertIsNotNone(cy)
        self.assertIsNotNone(dy)


class TestFourDimensionalSpline(unittest.TestCase):
    """Tests for the FourDimensionalSpline class"""

    def setUp(self):
        # Create a spiral-like path in 4D
        self.t = np.linspace(0, 2 * pi, 20)
        self.x = np.cos(self.t)
        self.y = np.sin(self.t)
        self.z = self.t / (2 * pi)
        self.w = np.sin(2 * self.t)
        self.spline = FourDimensionalSpline(self.x, self.y, self.z, self.w, self.t)

    def test_initialization(self):
        """Test spline initialization"""
        self.assertEqual(len(self.spline.splines), 4)

    def test_derivatives(self):
        """Test velocity, acceleration, and jerk"""
        t_test = 1.0

        velocity = self.spline.get_velocity(t_test)
        self.assertEqual(len(velocity), 2)

        acceleration = self.spline.get_acceleration(t_test)
        self.assertEqual(len(acceleration), 2)

        jerk = self.spline.get_jerk(t_test)
        self.assertEqual(len(jerk), 2)

    def test_get_parameters(self):
        """Test retrieving polynomial coefficients"""
        params = self.spline.get_parameters(
            0, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None
        )

        # Verify we get 16 parameters (a,b,c,d for each dimension)
        self.assertEqual(len(params), 16)

    def test_spline_circle_derivatives(self):
        # Create a circle
        t = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)

        # Create spline
        spline = TwoDimensionalSpline(x, y, t)

        # Test at a few points
        test_points = [0, np.pi / 4, np.pi / 2, np.pi]

        for tp in test_points:
            # Expected values for a unit circle
            expected_pos = np.array([np.cos(tp), np.sin(tp)])
            expected_vel = np.array([-np.sin(tp), np.cos(tp)])
            expected_acc = np.array([-np.cos(tp), -np.sin(tp)])
            expected_curvature = 1.0

            # Actual values from spline
            actual_pos = spline.get_point(tp)
            actual_vel = spline.get_velocity(tp)
            actual_acc = spline.get_acceleration(tp)
            actual_curvature = spline.get_curvature(tp)

            print(f"\nAt t={tp}:")
            print(f"Position: expected={expected_pos}, actual={actual_pos}")
            print(f"Velocity: expected={expected_vel}, actual={actual_vel}")
            print(f"Acceleration: expected={expected_acc}, actual={actual_acc}")
            print(f"Curvature: expected={expected_curvature}, actual={actual_curvature}")


class TestClothoid2D(unittest.TestCase):
    """Tests for the Clothoid2D class"""

    def setUp(self):
        # Create a simple clothoid
        self.waypoints_x = [0.0, 1.0, 2.0, 3.0]
        self.waypoints_y = [0.0, 0.0, 1.0, 1.0]
        self.waypoints_angle = [0.0, 0.0, pi / 2, pi / 2]
        self.sample_distance = 0.1
        self.clothoid = Clothoid2D(self.waypoints_x, self.waypoints_y, self.waypoints_angle, self.sample_distance)

    def test_initialization(self):
        """Test clothoid initialization"""
        x, y, s = self.clothoid.get_points_on_clothoid()

        # Check that we have points
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), len(s))

        # Check length is positive
        self.assertGreater(self.clothoid.get_length(), 0.0)

        # Check that s values are monotonically increasing
        for i in range(1, len(s)):
            self.assertGreaterEqual(s[i], s[i - 1])


class TestSplineVisual(unittest.TestCase):
    """Visual tests for spline functionality"""

    def setUp(self):
        # Create output directory if it doesn't exist
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)

    def test_2d_spline_visual(self):
        """Generate visual representation of 2D spline"""
        # Create figure
        plt.figure(figsize=(10, 8))

        # Create a figure-8 curve
        t = np.linspace(0, 2 * pi, 10)
        # x = np.sin(t)
        # y = np.sin(2 * t)

        x = [0, 1, 2.5, 3.6, 5, 7, 8.1, 10, 11.3, 12]
        y = np.sin(t)

        # Create spline
        spline = TwoDimensionalSpline(x, y, t)

        # Sample points along the spline
        points, angles = spline.sample_points(0.05)
        points = np.array(points)

        # Plot original points
        plt.plot(x, y, 'bo', label='Original Points')

        # Plot spline
        plt.plot(points[:, 0], points[:, 1], 'r-', label='Spline')

        # Plot tangent vectors at select points
        for i in range(0, len(points), 10):
            velocity = spline.get_velocity(spline.get_segment_start(i % spline.get_num_segments()))
            vel_normalized = velocity / np.linalg.norm(velocity) * 0.1
            plt.arrow(points[i, 0], points[i, 1], vel_normalized[0], vel_normalized[1],
                      head_width=0.05, head_length=0.1, fc='g', ec='g')

        plt.grid(True)
        plt.axis('equal')
        plt.title('2D Spline Visualization')
        plt.legend()

        # Save figure
        plt.savefig(self.output_dir / "2d_spline.png")
        plt.plot(points[:, 0], points[:, 1], 'gx', label="Control Points")
        plt.close()

    def direct_circle_curvature(self, t):
        # First derivatives for a unit circle
        x_prime = -np.sin(t)
        y_prime = np.cos(t)
        # Second derivatives
        x_double_prime = -np.cos(t)
        y_double_prime = -np.sin(t)

        numerator = y_double_prime * x_prime - x_double_prime * y_prime
        denominator = np.power(x_prime ** 2 + y_prime ** 2, 1.5)
        return numerator / denominator


    def test_curvature_visual(self):
        """Generate visual representation of spline curvature"""
        # Create figure
        plt.figure(figsize=(10, 8))

        # Create a circle-like path
        t = np.linspace(0, 2 * pi, 100)
        x = np.cos(t)
        y = np.sin(t)

        # Create spline
        spline = TwoDimensionalSpline(x, y, t)

        # Sample points and compute curvature
        t_samples = np.linspace(0, 2 * pi, 200)
        curvatures = [spline.get_curvature(t_val) for t_val in t_samples]

        for curvature in curvatures:
            print(f"Calculated curvature: {curvature}")

        # Plot curvature
        plt.plot(t_samples, curvatures, 'b-')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Expected Curvature for Circle')

        plt.grid(True)
        plt.title('Curvature of Circle Spline')
        plt.xlabel('Parameter t')
        plt.ylabel('Curvature')
        plt.legend()

        # Save figure
        plt.savefig(self.output_dir / "curvature.png")
        plt.close()

        for t in [0, np.pi / 4, np.pi / 2]:
            print(f"Direct curvature at t={t}: {self.direct_circle_curvature(t)}")



    # Test at a few points


    def test_closest_point_visual(self):
        """Generate visual representation of closest point finding"""
        # Create figure
        plt.figure(figsize=(10, 8))

        # Create a figure-8 curve
        t = np.linspace(0, 2 * pi, 10)
        # x = np.sin(t)
        # y = np.sin(2 * t)

        x = [0, 1, 2.5, 3.6, 5, 7, 8.1, 10, 11.3, 12]
        y = np.sin(t)

        # Create spline
        spline = TwoDimensionalSpline(x, y, t)

        # Sample points along the spline
        points, _ = spline.sample_points(0.05)
        points = np.array(points)
        # Plot spline
        plt.plot(points[:, 0], points[:, 1], 'r-', label='Spline')

        # Test points
        test_points = [
            np.array([0.5, 0.5]),
            np.array([-0.5, 0.5]),
            np.array([0, 0]),
            np.array([1.0, 0]),
            np.array([0, 1.0]),
            np.array([10, -6.0]),
            np.array([-8, -3.2])
        ]

        # Find closest points
        for test_point in test_points:
            segment, t_val = spline.find_closest_point(test_point)
            closest = spline.get_point(t_val)

            # Plot test point and closest point
            plt.plot(test_point[0], test_point[1], 'bo')
            plt.plot(closest[0], closest[1], 'go')

            # Draw line from test point to closest point
            plt.plot([test_point[0], closest[0]], [test_point[1], closest[1]], 'k--')

        plt.grid(True)
        plt.axis('equal')
        plt.title('Closest Point Finding')
        plt.legend(['Spline', 'Test Points', 'Closest Points'])

        # Save figure
        plt.savefig(self.output_dir / "closest_point.png")
        plt.close()


class TestSplinePerformance(unittest.TestCase):
    """Performance tests for spline operations"""

    def test_2d_spline_sample_performance(self):
        """Test performance of 2D spline sampling"""
        import time

        # Create a complex path
        t = np.linspace(0, 10 * pi, 1000)
        x = np.sin(t) * t / 10
        y = np.cos(t) * t / 10

        # Create spline
        start_time = time.time()
        spline = TwoDimensionalSpline(x, y, t)
        construction_time = time.time() - start_time

        # Sample points
        start_time = time.time()
        points, _ = spline.sample_points(0.01)
        sampling_time = time.time() - start_time

        # Find closest points
        test_points = [np.array([i, i]) for i in np.linspace(-5, 5, 100)]
        start_time = time.time()
        for point in test_points:
            spline.find_closest_point(point)
        closest_time = time.time() - start_time

        # Print performance metrics
        print(f"\nPerformance Metrics:")
        print(f"  Spline Construction Time: {construction_time:.4f} seconds")
        print(f"  Point Sampling Time ({len(points)} points): {sampling_time:.4f} seconds")
        print(f"  Closest Point Finding Time ({len(test_points)} queries): {closest_time:.4f} seconds")

        # Assert reasonable performance
        self.assertLess(construction_time, 1.0, "Spline construction is too slow")
        self.assertLess(sampling_time, 10.0, "Point sampling is too slow")
        self.assertLess(closest_time, 5.0, "Closest point finding is too slow")


def generate_visualization():
    """Generate comprehensive visualization of spline functionality"""
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(15, 10))

    # Create test data for a complex path
    t = np.linspace(0, 4 * pi, 50)
    x = np.sin(t) * (1 + t / 10)
    y = np.cos(t) * (1 + t / 10)

    # Create 2D spline
    spline = TwoDimensionalSpline(x, y, t)

    # Sample points along the spline
    sample_points, sample_angles = spline.sample_points(0.1)
    sample_points = np.array(sample_points)

    # Plot original vs interpolated
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'bo', label='Original Points')
    plt.plot(sample_points[:, 0], sample_points[:, 1], 'r-', label='Spline')
    plt.grid(True)
    plt.axis('equal')
    plt.title('Spline Interpolation')
    plt.legend()

    # Plot curvature
    plt.subplot(2, 2, 2)
    t_samples = np.linspace(0, 4 * pi, 200)
    curvatures = [spline.get_curvature(t_val) for t_val in t_samples]
    plt.plot(t_samples, curvatures, 'g-')
    plt.grid(True)
    plt.title('Curvature Along Spline')
    plt.xlabel('Parameter t')
    plt.ylabel('Curvature')

    # Plot velocity magnitude
    plt.subplot(2, 2, 3)
    velocities = [np.linalg.norm(spline.get_velocity(t_val)) for t_val in t_samples]
    plt.plot(t_samples, velocities, 'b-')
    plt.grid(True)
    plt.title('Velocity Magnitude Along Spline')
    plt.xlabel('Parameter t')
    plt.ylabel('Velocity Magnitude')

    # Plot path with tangent and normal vectors
    plt.subplot(2, 2, 4)
    plt.plot(sample_points[:, 0], sample_points[:, 1], 'r-', label='Spline')

    # Add tangent and normal vectors at select points
    for i in range(0, len(sample_points), 5):
        t_val = t[min(i // 5, len(t) - 1)]
        velocity = spline.get_velocity(t_val)
        vel_normalized = velocity / np.linalg.norm(velocity) * 0.2
        normal = spline.get_orthogonal(t_val) * 0.2

        plt.arrow(sample_points[i, 0], sample_points[i, 1],
                  vel_normalized[0], vel_normalized[1],
                  head_width=0.05, head_length=0.1, fc='g', ec='g')
        plt.arrow(sample_points[i, 0], sample_points[i, 1],
                  normal[0], normal[1],
                  head_width=0.05, head_length=0.1, fc='b', ec='b')

    plt.grid(True)
    plt.axis('equal')
    plt.title('Tangent (green) and Normal (blue) Vectors')

    plt.tight_layout()
    plt.savefig(output_dir / "spline_comprehensive.png")
    plt.close()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

    # Generate visualization
    generate_visualization()