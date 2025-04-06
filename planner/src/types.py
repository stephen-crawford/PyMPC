import numpy as np
from math import sqrt, atan2
import logging
from bisect import bisect_right
from enum import Enum
from typing import List, Optional


logger = logging.getLogger(__name__)

class Disc:
    def __init__(self, offset_: float, radius_: float):
        self.offset = offset_
        self.radius = radius_

    def get_position(self, robot_position: np.ndarray, angle: float) -> np.ndarray:
        """Get disc position from robot position and angle"""
        # Implementation would calculate position based on robot center and angle
        # This is a placeholder for the actual implementation
        return np.zeros(2)  # Replace with actual implementation

    def to_robot_center(self, disc_position: np.ndarray, angle: float) -> np.ndarray:
        """Convert disc position to robot center position"""
        # Implementation would calculate robot center based on disc position and angle
        # This is a placeholder for the actual implementation
        return np.zeros(2)  # Replace with actual implementation

class Halfspace:
    def __init__(self, A: np.ndarray, b: float):
        """Halfspace defined by Ax <= b"""
        self.A = A
        self.b = b

# Type definition for StaticObstacle
# In Python, we can use a simple type hint
# StaticObstacle = List[Halfspace]

class PredictionType(Enum):
    DETERMINISTIC = 0
    GAUSSIAN = 1
    NONGAUSSIAN = 2
    NONE = 3

class PredictionStep:
    def __init__(self, position: np.ndarray, angle: float, major_radius: float, minor_radius: float):
        # Mean
        self.position = position
        self.angle = angle
        # Covariance
        self.major_radius = major_radius
        self.minor_radius = minor_radius

# Type definition for Mode
# Mode = List[PredictionStep]

class Prediction:
    def __init__(self, type_=None):
        self.type = type_
        self.modes = []
        self.probabilities = []

    def empty(self) -> bool:
        return len(self.modes) == 0

class ObstacleType(Enum):
    STATIC = 0
    DYNAMIC = 1

class DynamicObstacle:
    def __init__(self, index: int, position: np.ndarray, angle: float, radius: float,
                 _type: 'ObstacleType' = ObstacleType.DYNAMIC):
        self.index = index
        self.position = position
        self.angle = angle
        self.radius = radius
        self.type = _type
        self.prediction = Prediction()

class ReferencePath:
    def __init__(self, length: int = 10):
        self.x = []
        self.y = []
        self.psi = []
        self.v = []
        self.s = []

    def clear(self):
        self.x = []
        self.y = []
        self.psi = []
        self.v = []
        self.s = []

    def point_in_path(self, point_num: int, other_x: float, other_y: float) -> bool:
        # Implementation would check if point is in path
        # This is a placeholder for the actual implementation
        return False  # Replace with actual implementation

    def empty(self) -> bool:
        return len(self.x) == 0

    def has_velocity(self) -> bool:
        return len(self.v) > 0

    def has_distance(self) -> bool:
        return len(self.s) > 0

# Type definition for Boundary
# Boundary = ReferencePath

class Trajectory:
    def __init__(self, dt: float = 0.0, length: int = 10):
        self.dt = dt
        self.positions = []

    def add(self, p_or_x, y=None):
        if y is None:
            # p is an ndarray
            p = p_or_x
            self.positions.append(p.copy())
        else:
            # p_or_x is x, y is y
            self.positions.append(np.array([p_or_x, y]))

class FixedSizeTrajectory:
    def __init__(self, size: int = 50):
        self._size = size
        self.positions = []

    def add(self, p: np.ndarray):
        self.positions.append(p.copy())
        if len(self.positions) > self._size:
            self.positions.pop(0)


class BandMatrix:
    """Band matrix implementation with LU decomposition for solving linear systems."""

    def __init__(self, dim=0, n_upper=0, n_lower=0):
        """Initialize a band matrix with given dimensions.

        Args:
            dim (int): Matrix dimension
            n_upper (int): Number of upper diagonals
            n_lower (int): Number of lower diagonals
        """
        if dim > 0:
            self.resize(dim, n_upper, n_lower)
        else:
            self.m_upper = []
            self.m_lower = []

    def resize(self, dim, n_upper, n_lower):
        """Resize the band matrix.

        Args:
            dim (int): Matrix dimension
            n_upper (int): Number of upper diagonals
            n_lower (int): Number of lower diagonals
        """
        assert dim > 0
        assert n_upper >= 0
        assert n_lower >= 0

        self.m_upper = [np.zeros(dim) for _ in range(n_upper + 1)]
        self.m_lower = [np.zeros(dim) for _ in range(n_lower + 1)]

    def dim(self):
        """Get the dimension of the matrix."""
        if len(self.m_upper) > 0:
            return len(self.m_upper[0])
        else:
            return 0

    def num_upper(self):
        """Get the number of upper diagonals."""
        return len(self.m_upper) - 1

    def num_lower(self):
        """Get the number of lower diagonals."""
        return len(self.m_lower) - 1

    def get_element(self, i, j):
        """Get the element at position (i, j)."""
        k = j - i  # which band is the entry
        assert 0 <= i < self.dim() and 0 <= j < self.dim()
        assert -self.num_lower() <= k <= self.num_upper()

        # k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if k >= 0:
            return self.m_upper[k][i]
        else:
            return self.m_lower[-k][i]

    def set_element(self, i, j, value):
        """Set the element at position (i, j)."""
        k = j - i  # which band is the entry
        assert 0 <= i < self.dim() and 0 <= j < self.dim()
        assert -self.num_lower() <= k <= self.num_upper()

        # k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if k >= 0:
            self.m_upper[k][i] = value
        else:
            self.m_lower[-k][i] = value

    def get_saved_diag(self, i):
        """Get the saved diagonal element."""
        assert 0 <= i < self.dim()
        return self.m_lower[0][i]

    def set_saved_diag(self, i, value):
        """Set the saved diagonal element."""
        assert 0 <= i < self.dim()
        self.m_lower[0][i] = value

    def lu_decompose(self):
        """LU decomposition of a band matrix."""
        n = self.dim()

        # Preconditioning: normalize column i so that a_ii=1
        for i in range(n):
            assert self.get_element(i, i) != 0.0
            self.set_saved_diag(i, 1.0 / self.get_element(i, i))

            j_min = max(0, i - self.num_lower())
            j_max = min(n - 1, i + self.num_upper())

            for j in range(j_min, j_max + 1):
                self.set_element(i, j, self.get_element(i, j) * self.get_saved_diag(i))

            self.set_element(i, i, 1.0)  # prevents rounding errors

        # Gauss LR-Decomposition
        for k in range(n):
            i_max = min(n - 1, k + self.num_lower())

            for i in range(k + 1, i_max + 1):
                assert self.get_element(k, k) != 0.0
                x = -self.get_element(i, k) / self.get_element(k, k)
                self.set_element(i, k, -x)  # assembly part of L

                j_max = min(n - 1, k + self.num_upper())
                for j in range(k + 1, j_max + 1):
                    # assembly part of R
                    new_val = self.get_element(i, j) + x * self.get_element(k, j)
                    self.set_element(i, j, new_val)

    def l_solve(self, b):
        """Solve Ly=b."""
        assert self.dim() == len(b)
        n = self.dim()
        x = np.zeros(n)

        for i in range(n):
            sum_val = 0
            j_start = max(0, i - self.num_lower())

            for j in range(j_start, i):
                sum_val += self.get_element(i, j) * x[j]

            x[i] = (b[i] * self.get_saved_diag(i)) - sum_val

        return x

    def r_solve(self, b):
        """Solve Rx=y."""
        assert self.dim() == len(b)
        n = self.dim()
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            sum_val = 0
            j_stop = min(n - 1, i + self.num_upper())

            for j in range(i + 1, j_stop + 1):
                sum_val += self.get_element(i, j) * x[j]

            x[i] = (b[i] - sum_val) / self.get_element(i, i)

        return x

    def lu_solve(self, b, is_lu_decomposed=False):
        """Solve Ax=b using LU decomposition."""
        assert self.dim() == len(b)

        if not is_lu_decomposed:
            self.lu_decompose()

        y = self.l_solve(b)
        x = self.r_solve(y)

        return x


class TkSpline:
    """Cubic spline interpolation class."""

    # Boundary condition types
    FIRST_DERIV = 1
    SECOND_DERIV = 2

    def __init__(self):
        """Initialize spline with default parameters."""
        self.m_x = []
        self.m_y = []
        self.m_x_ = []  # Copy of x coordinates as in original code
        self.m_y_ = []  # Copy of y coordinates as in original code
        self.m_a = []
        self.m_b = []
        self.m_c = []
        self.m_d = []
        self.m_b0 = 0.0
        self.m_c0 = 0.0
        self.m_left = self.SECOND_DERIV
        self.m_right = self.SECOND_DERIV
        self.m_left_value = 0.0
        self.m_right_value = 0.0
        self.m_force_linear_extrapolation = False

    def set_boundary(self, left, left_value, right, right_value, force_linear_extrapolation=False):
        """Set boundary conditions.

        Args:
            left: Boundary condition type for left boundary
            left_value: Value for left boundary condition
            right: Boundary condition type for right boundary
            right_value: Value for right boundary condition
            force_linear_extrapolation: Force linear extrapolation
        """
        assert len(self.m_x) == 0  # set_points() must not have happened yet
        self.m_left = left
        self.m_right = right
        self.m_left_value = left_value
        self.m_right_value = right_value
        self.m_force_linear_extrapolation = force_linear_extrapolation

    def set_points(self, x, y, cubic_spline=True):
        """Set data points for interpolation.

        Args:
            x: x coordinates
            y: y coordinates
            cubic_spline: Use cubic spline if True, linear interpolation if False
        """


        assert len(x) == len(y)
        assert len(x) > 2

        self.m_x = list(x)
        self.m_y = list(y)
        self.m_x_ = list(x)  # Store copies as in original code
        self.m_y_ = list(y)  # Store copies as in original code

        n = len(x)

        # Check that x is strictly increasing
        for i in range(n - 1):
            assert self.m_x[i] < self.m_x[i + 1]

        if cubic_spline:
            # Cubic spline interpolation
            # Setting up the matrix and right hand side of the equation system
            # for the parameters b[]
            A = BandMatrix(n, 1, 1)
            rhs = np.zeros(n)

            for i in range(1, n - 1):
                A.set_element(i, i - 1, 1.0 / 3.0 * (x[i] - x[i - 1]))
                A.set_element(i, i, 2.0 / 3.0 * (x[i + 1] - x[i - 1]))
                A.set_element(i, i + 1, 1.0 / 3.0 * (x[i + 1] - x[i]))
                rhs[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])

            # Boundary conditions
            if self.m_left == self.FIRST_DERIV:
                # c[0] = f', needs to be re-expressed in terms of b:
                # (2b[0]+b[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
                A.set_element(0, 0, 2.0 * (x[1] - x[0]))
                A.set_element(0, 1, 1.0 * (x[1] - x[0]))
                rhs[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0]) - self.m_left_value)
            elif self.m_left == self.SECOND_DERIV:
                # 2*b[0] = f''
                A.set_element(0, 0, 2.0)
                A.set_element(0, 1, 0.0)
                rhs[0] = self.m_left_value
            else:
                assert False

            if self.m_right == self.FIRST_DERIV:
                # c[n-1] = f', needs to be re-expressed in terms of b:
                # (b[n-2]+2b[n-1])(x[n-1]-x[n-2])
                # = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
                A.set_element(n - 1, n - 1, 2.0 * (x[n - 1] - x[n - 2]))
                A.set_element(n - 1, n - 2, 1.0 * (x[n - 1] - x[n - 2]))
                rhs[n - 1] = 3.0 * (self.m_right_value - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
            elif self.m_right == self.SECOND_DERIV:
                # 2*b[n-1] = f''
                A.set_element(n - 1, n - 1, 2.0)
                A.set_element(n - 1, n - 2, 0.0)
                rhs[n - 1] = self.m_right_value
            else:
                assert False

            # Solve the equation system to obtain the parameters b[]
            self.m_b = A.lu_solve(rhs)

            # Calculate parameters a[] and c[] based on b[]
            self.m_a = np.zeros(n)
            self.m_c = np.zeros(n)
            self.m_d = y

            for i in range(n - 1):
                self.m_a[i] = 1.0 / 3.0 * (self.m_b[i + 1] - self.m_b[i]) / (x[i + 1] - x[i])
                self.m_c[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - 1.0 / 3.0 * (
                            2.0 * self.m_b[i] + self.m_b[i + 1]) * (x[i + 1] - x[i])
        else:
            # Linear interpolation
            self.m_a = np.zeros(n)
            self.m_b = np.zeros(n)
            self.m_c = np.zeros(n)
            self.m_d = y

            for i in range(n - 1):
                self.m_a[i] = 0.0
                self.m_b[i] = 0.0
                self.m_c[i] = (self.m_y[i + 1] - self.m_y[i]) / (self.m_x[i + 1] - self.m_x[i])

        # For left extrapolation coefficients
        self.m_b0 = 0.0 if self.m_force_linear_extrapolation else self.m_b[0]
        self.m_c0 = self.m_c[0]

        # For the right extrapolation coefficients
        # f_{n-1}(x) = b*(x-x_{n-1})^2 + c*(x-x_{n-1}) + y_{n-1}
        h = x[n - 1] - x[n - 2]
        # m_b[n-1] is determined by the boundary condition
        self.m_a[n - 1] = 0.0
        self.m_c[n - 1] = 3.0 * self.m_a[n - 2] * h * h + 2.0 * self.m_b[n - 2] * h + self.m_c[
            n - 2]  # = f'_{n-2}(x_{n-1})

        if self.m_force_linear_extrapolation:
            self.m_b[n - 1] = 0.0

    def __call__(self, x):
        """Evaluate the spline at point x.

        Args:
            x: Point to evaluate at

        Returns:
            Interpolated value
        """
        n = len(self.m_x)

        # Find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
        idx = max(bisect_right(self.m_x, x) - 1, 0)

        h = x - self.m_x[idx]

        # handle periodicity
        if x > self.m_x[len(self.m_x) - 1]:
            x = self.m_x[0] + ((x - self.m_x[0]) % (self.m_x[-1] - self.m_x[0]))
        if x < self.m_x[0]:
            # Extrapolation to the left
            return (self.m_b0 * h + self.m_c0) * h + self.m_y[0]
        elif x > self.m_x[n - 1]:
            # Extrapolation to the right

            return (self.m_b[n - 1] * h + self.m_c[n - 1]) * h + self.m_y[n - 1]
        else:
            # Interpolation
            return ((self.m_a[idx] * h + self.m_b[idx]) * h + self.m_c[idx]) * h + self.m_y[idx]

    def deriv(self, order, x):
        """Compute derivative of the spline.

        Args:
            order: Order of derivative (1, 2, or 3)
            x: Point to evaluate at

        Returns:
            Value of derivative
        """
        assert order > 0

        n = len(self.m_x)

        # Find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
        idx = max(bisect_right(self.m_x, x) - 1, 0)

        h = x - self.m_x[idx]

        if x < self.m_x[0]:
            # Extrapolation to the left
            if order == 1:
                return 2.0 * self.m_b0 * h + self.m_c0
            elif order == 2:
                return 2.0 * self.m_b0
            else:
                return 0.0
        elif x > self.m_x[n - 1]:
            # Extrapolation to the right
            if order == 1:
                return 2.0 * self.m_b[n - 1] * h + self.m_c[n - 1]
            elif order == 2:
                return 2.0 * self.m_b[n - 1]
            else:
                return 0.0
        else:
            # Interpolation
            if order == 1:
                return (3.0 * self.m_a[idx] * h + 2.0 * self.m_b[idx]) * h + self.m_c[idx]
            elif order == 2:
                return 6.0 * self.m_a[idx] * h + 2.0 * self.m_b[idx]
            elif order == 3:
                return 6.0 * self.m_a[idx]
            else:
                return 0.0

    def get_parameters(self, index):
        """Get spline parameters for segment index.

        Args:
            index: Segment index

        Returns:
            Tuple of (a, b, c, d) parameters for the segment
        """
        assert index <= len(self.m_a) - 1
        return self.m_a[index], self.m_b[index], self.m_c[index], self.m_d[index]

    def get_spline_start(self, index):
        """Get starting x value for segment index.

        Args:
            index: Segment index

        Returns:
            Starting x value for the segment
        """
        assert index >= 0
        assert index < len(self.m_x) - 1
        return self.m_x[index]

    def get_spline_end(self, index):
        """Get ending x value for segment index.

        Args:
            index: Segment index

        Returns:
            Ending x value for the segment
        """
        assert index >= 0
        assert index < len(self.m_x) - 1
        return self.m_x[index + 1]


class Spline:
    """Base spline class for multiple dimensions"""

    def __init__(self, points=None, t_vector=None, start_velocity=None):
        self.splines = []
        self.t_vector = []
        self.s_vector = []
        self.closest_segment = -1
        self.prev_query_point = None

        if points is not None and t_vector is not None:
            self.initialize(points, t_vector)

    def initialize(self, points, t_vector):
        """Initialize splines with points and parameter vector"""
        self.t_vector = list(t_vector)
        self.compute_distance_vector(points, self.s_vector)

        # Initialize splines for each dimension
        dim = len(points)
        self.splines = []
        for i in range(dim):
            spline = TkSpline()
            spline.set_points(points[i])
            self.splines.append(spline)

    def compute_distance_vector(self, points, out):
        """Compute the distance vector between points"""
        dim = len(points)
        n_points = len(points[0])

        out.clear() if hasattr(out, 'clear') else None
        out.append(0.0)

        for i in range(1, n_points):
            a = np.array([points[d][i - 1] for d in range(dim)])
            b = np.array([points[d][i] for d in range(dim)])
            dist = np.linalg.norm(b - a)
            out.append(out[-1] + dist)

    def get_point(self, t):
        """Get point at parameter value t"""
        return np.array([spline(t) for spline in self.splines])

    def get_coordinate(self, t, coordinate):
        """Get specific coordinate at parameter value t"""
        return self.splines[coordinate](t)

    def get_velocity(self, t):
        """Get velocity (first derivative) at parameter value t"""
        return np.array([spline.deriv(1, t) for spline in self.splines])

    def get_acceleration(self, t):
        """Get acceleration (second derivative) at parameter value t"""
        return np.array([spline.deriv(2, t) for spline in self.splines])

    def get_jerk(self, t):
        """Get jerk (third derivative) at parameter value t"""
        return np.array([spline.deriv(3, t) for spline in self.splines])

    def get_orthogonal(self, t):
        """Get vector orthogonal to curve at parameter value t"""
        # For 2D implementation, override in subclass
        raise NotImplementedError("Implement in subclass")

    def get_num_segments(self):
        """Get number of segments in the spline"""
        if not self.splines:
            return 0
        return len(self.splines[0].m_x) - 1

    def get_segment_start(self, index):
        """Get starting parameter value for a segment"""
        if index > self.get_num_segments() - 1:
            return self.t_vector[-1]
        else:
            return self.t_vector[index]

    def get_length(self):
        """Get total length of the spline"""
        return self.s_vector[-1] if self.s_vector else 0.0

    def get_parameter_length(self):
        """Get total parameter length"""
        return self.t_vector[-1] if self.t_vector else 0.0

    def get_parameters(self, segment_index, *args):
        """Get polynomial coefficients for a segment (to be implemented in subclasses)"""
        raise NotImplementedError("Implement in subclass")

    def initialize_closest_point(self, point):
        """Initialize search for closest point to the spline"""
        min_dist = float('inf')
        local_segment_out = -1
        local_t_out = -1.0

        # Check each segment
        for i in range(len(self.t_vector) - 1):
            cur_t = self.find_closest_s_recursively(point, self.t_vector[i], self.t_vector[i + 1], 10)
            cur_dist = np.linalg.norm(self.get_point(cur_t) - point)

            if cur_dist < min_dist:
                min_dist = cur_dist
                local_t_out = cur_t
                local_segment_out = i

        if local_segment_out == -1:
            raise ValueError("Could not find a closest point on the spline")

        self.closest_segment = local_segment_out
        return local_segment_out, local_t_out

    def find_closest_point(self, point, range_val=2):
        """Find the closest point on the spline to the given point"""
        # If not initialized or point is far from previous query
        if self.closest_segment == -1 or (self.prev_query_point is not None and
                                          np.linalg.norm(self.prev_query_point - point) > 5.0):
            segment_out, t_out = self.initialize_closest_point(point)
            self.prev_query_point = point.copy() if hasattr(point, 'copy') else point
            return segment_out, t_out

        self.prev_query_point = point.copy() if hasattr(point, 'copy') else point

        # Search locally
        first_segment = max(0, self.closest_segment - range_val)
        last_segment = min(len(self.t_vector) - 1, self.closest_segment + range_val)

        t_out = self.find_closest_s_recursively(point, self.t_vector[first_segment],
                                                self.t_vector[last_segment], 0)

        # Find which segment this parameter falls into
        for i in range(first_segment, last_segment):
            if t_out > self.t_vector[i] and t_out < self.t_vector[i + 1]:
                self.closest_segment = i
                return i, t_out

        self.closest_segment = last_segment
        return last_segment, t_out

    def find_closest_s_recursively(self, point, low, high, num_recursions=0):
        """Recursively find the closest parameter on the spline to the given point"""
        if abs(high - low) <= 1e-4 or num_recursions > 40:
            if num_recursions > 40:
                logger.warning("Recursion count exceeded.")
            return (low + high) / 2.0

        # Compute middle parameter value
        mid = (low + high) / 2.0

        # Compute distances to spline at low and high
        value_low = np.linalg.norm(self.get_point(low) - point)
        value_high = np.linalg.norm(self.get_point(high) - point)

        # Recurse on the closer half
        if value_low < value_high:
            return self.find_closest_s_recursively(point, low, mid, num_recursions + 1)
        else:
            return self.find_closest_s_recursively(point, mid, high, num_recursions + 1)


class TwoDimensionalSpline(Spline):
    """2D spline implementation"""

    def __init__(self, x=None, y=None, t_vector=None):
        super().__init__()
        if x is not None and y is not None and t_vector is not None:
            self.x = list(x)
            self.y = list(y)
            self.t_vector = list(t_vector)
            self.s_vector = []

            # Initialize distance vector
            self.compute_distance_vector([[x_val for x_val in x],
                                          [y_val for y_val in y]], self.s_vector)

            # Initialize splines
            self.x_spline = TkSpline()
            self.y_spline = TkSpline()
            self.x_spline.set_points(self.t_vector, x)
            self.y_spline.set_points(self.t_vector, y)
            self.splines = [self.x_spline, self.y_spline]
            self.closest_segment = -1
            self.prev_query_point = None

    def get_point(self, t):
        """Get 2D point at parameter value t"""
        return np.array([self.x_spline(t), self.y_spline(t)])

    def get_x(self, t):
        """Get x coordinate at parameter value t"""
        return self.x_spline(t)

    def get_y(self, t):
        """Get y coordinate at parameter value t"""
        return self.y_spline(t)

    def get_velocity(self, t):
        """Get velocity vector at parameter value t"""
        vel = np.array([self.x_spline.deriv(1, t), self.y_spline.deriv(1, t)])
        return vel

    def get_acceleration(self, t):
        """Get acceleration vector at parameter value t"""
        accel = np.array([self.x_spline.deriv(2, t), self.y_spline.deriv(2, t)])
        return accel

    def get_jerk(self, t):
        """Get jerk vector at parameter value t"""
        return np.array([self.x_spline.deriv(3, t), self.y_spline.deriv(3, t)])

    def get_curvature(self, t):
        """Get curvature at parameter value t"""
        first_deriv = self.get_velocity(t)
        second_deriv = self.get_acceleration(t)

        # k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        numerator = first_deriv[0] * second_deriv[1] - first_deriv[1] * second_deriv[0]
        denominator = np.power(first_deriv[0] ** 2 + first_deriv[1] ** 2, 1.5)

        return numerator / denominator

    def get_curvature_deriv(self, t):
        """Get derivative of curvature at parameter value t"""
        first_deriv = self.get_velocity(t)
        second_deriv = self.get_acceleration(t)
        third_deriv = self.get_jerk(t)

        # Complex derivative of curvature calculation
        z = first_deriv[0] * second_deriv[1] - second_deriv[0] * first_deriv[1]
        n = np.power(first_deriv[0] ** 2 + first_deriv[1] ** 2, 1.5)
        z_d = first_deriv[0] * third_deriv[1] - third_deriv[0] * first_deriv[1]
        n_d = 1.5 * np.sqrt(first_deriv[0] ** 2 + first_deriv[1] ** 2) * 2 * (
                first_deriv[0] * second_deriv[0] + first_deriv[1] * second_deriv[1]) * 1.5

        k_d = (z_d * n - z * n_d) / (n ** 2)
        return k_d

    def get_orthogonal(self, t):
        """Get unit vector orthogonal to curve at parameter value t"""
        vel = self.get_velocity(t)
        # Orthogonal vector in 2D: (-y', x')
        orth = np.array([-vel[1], vel[0]])
        # Normalize
        norm = np.linalg.norm(orth)
        if norm > 1e-10:  # Prevent division by zero
            return orth / norm
        return orth

    def get_path_angle(self, t):
        """Get tangent angle at parameter value t"""
        return np.atan2(self.y_spline.deriv(1, t), self.x_spline.deriv(1, t))

    def sample_points(self, ds):
        """Sample points and angles along the spline with distance ds"""
        points = []
        angles = []
        length = self.get_length()

        spline_sample_dist = min(ds, length)
        n_spline_pts = int(np.ceil(length / spline_sample_dist))

        s_cur = 0.0
        for _ in range(n_spline_pts):
            vel = self.get_velocity(s_cur)
            point = self.get_point(s_cur)
            angle = np.arctan2(vel[1], vel[0])
            points.append(point)
            angles.append(angle)

            s_cur += spline_sample_dist

        # Ensure we capture the final point
        error = np.linalg.norm(points[-1] - self.get_point(length))
        if error > 0.01:
            vel = self.get_velocity(length)
            points.append(self.get_point(length))
            angles.append(np.arctan2(vel[1], vel[0]))

        return points, angles

    def get_parameters(self, segment_index, ax=None, bx=None, cx=None, dx=None,
                       ay=None, by=None, cy=None, dy=None):
        """Get polynomial coefficients for a segment"""
        # Extrapolate with constant path based on last segment if needed
        if segment_index > self.get_num_segments() - 1:
            ax, bx, cx, dx, ay, by, cy, dy = self.get_parameters(
                self.get_num_segments() - 1, ax, bx, cx, dx, ay, by, cy, dy)

            ax, bx, cx = 0.0, 0.0, 0.0
            ay, by, cy = 0.0, 0.0, 0.0

            return ax, bx, cx, dx, ay, by, cy, dy

        ax_val, bx_val, cx_val, dx_val = self.x_spline.get_parameters(segment_index)
        ay_val, by_val, cy_val, dy_val = self.y_spline.get_parameters(segment_index)

        return ax_val, bx_val, cx_val, dx_val, ay_val, by_val, cy_val, dy_val


class FourDimensionalSpline(TwoDimensionalSpline):
    """4D spline implementation extending 2D spline"""

    def __init__(self, x=None, y=None, z=None, w=None, t_vector=None):
        super().__init__(x, y, t_vector)
        if z is not None and w is not None:
            self.z = list(z)
            self.w = list(w)

            # Initialize z and w splines
            self.z_spline = TkSpline()
            self.w_spline = TkSpline()
            self.z_spline.set_points(t_vector, z)
            self.w_spline.set_points(t_vector, w)

            self.splines = [self.x_spline, self.y_spline, self.z_spline, self.w_spline]

    def get_point(self, t):
        """Get 4D point at parameter value t"""
        return np.array([self.x_spline(t), self.y_spline(t),
                         self.z_spline(t), self.w_spline(t)])

    def get_parameters(self, segment_index, ax=None, bx=None, cx=None, dx=None,
                       ay=None, by=None, cy=None, dy=None,
                       az=None, bz=None, cz=None, dz=None,
                       aw=None, bw=None, cw=None, dw=None):
        """Get polynomial coefficients for a segment for all 4 dimensions"""
        # Get 2D coefficients first
        ax, bx, cx, dx, ay, by, cy, dy = super().get_parameters(
            segment_index, ax, bx, cx, dx, ay, by, cy, dy)

        # Get z and w coefficients
        az_val, bz_val, cz_val, dz_val = self.z_spline.get_parameters(segment_index)
        aw_val, bw_val, cw_val, dw_val = self.w_spline.get_parameters(segment_index)

        return (ax, bx, cx, dx, ay, by, cy, dy,
                az_val, bz_val, cz_val, dz_val,
                aw_val, bw_val, cw_val, dw_val)


class Clothoid2D:
    """2D clothoid implementation"""

    def __init__(self, waypoints_x, waypoints_y, waypoints_angle, sample_distance):
        self._length = 0.0
        self._x = []
        self._y = []
        self._s = []
        self.fit_clothoid(waypoints_x, waypoints_y, waypoints_angle, sample_distance)

    def get_points_on_clothoid(self):
        """Get points on the clothoid"""
        return self._x, self._y, self._s

    def get_length(self):
        """Get total length of the clothoid"""
        return self._length

    def fit_clothoid(self, waypoints_x, waypoints_y, waypoints_angle, sample_distance):
        """Fit a clothoid to the waypoints"""
        # Note: This is a simplified implementation
        # In a full implementation, you would need the actual Clothoid::buildClothoid function
        # which involves solving for clothoid parameters (curvature, curvature rate)

        self._s = [0.0]

        n_waypoints = len(waypoints_x)

        for i in range(n_waypoints - 1):
            # Here we would normally build a clothoid
            # For now, linear approximation (this is not a true clothoid)
            x1, y1, angle1 = waypoints_x[i], waypoints_y[i], waypoints_angle[i]
            x2, y2, angle2 = waypoints_x[i + 1], waypoints_y[i + 1], waypoints_angle[i + 1]

            # Calculate segment length (simplified)
            dx, dy = x2 - x1, y2 - y1
            L = np.sqrt(dx * dx + dy * dy)

            # Sample points along this segment
            n_clothoid = max(int(np.ceil(L / sample_distance)), 2)
            X = []
            Y = []

            for j in range(n_clothoid):
                t = j / (n_clothoid - 1)
                # Linear interpolation (this is NOT a true clothoid)
                X.append(x1 + t * dx)
                Y.append(y1 + t * dy)

            self._length += L

            # Add points to the overall clothoid
            if i == 0:
                # Add all points for the first segment
                self._x.extend(X)
                self._y.extend(Y)
            else:
                # Skip duplicate initial point for later segments
                self._x.extend(X[1:])
                self._y.extend(Y[1:])

            # Add distances
            for j in range(1, n_clothoid):
                self._s.append(self._s[-1] + L / (n_clothoid - 1))


