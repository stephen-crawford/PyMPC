import math
import random
from bisect import bisect_right

from typing import List

from casadi import atan2

import numpy as np
import casadi as cd
import math

from utils.utils import LOG_WARN


# Math utilities

def distance(a, b):
    """
Calculate Euclidean distance between two points.
"""
    a_array = np.array(a)
    b_array = np.array(b)
    return np.sqrt(np.sum((a_array - b_array) ** 2))


def exponential_quantile(lambda_param, p):
    """
Find the exponential CDF value at probability p.
"""
    return -math.log(1 - p) / lambda_param


def linspace(start, end, num):
    """
Create an evenly spaced array of numbers.
"""
    if num == 0:
        return []
    if num == 1:
        return [end]
    if num == 2:
        return [start, end]

    delta = (end - start) / (num - 1)
    result = [start]
    for i in range(1, num - 1):
        result.append(start + delta * i)
    result.append(end)
    return result


def rotation_matrix_from_heading(heading):
    """
Create a 2D rotation matrix from a heading angle.
"""
    c = math.cos(heading)
    s = math.sin(heading)
    return np.array([[c, s], [-s, c]])


def rotation_matrix(angle):
    return np.array([[cd.cos(angle), -cd.sin(angle)],
                    [cd.sin(angle), cd.cos(angle)]])


def angular_difference(angle1, angle2):
    """
Calculate the shortest angular difference between two angles.
"""
    diff = math.fmod(angle2 - angle1, 2 * math.pi)

    if diff > math.pi:
        diff -= 2 * math.pi  # Shift difference to be within [-pi, pi]
    elif diff < -math.pi:
        diff += 2 * math.pi

    return diff


def bisection(low, high, func, tol):
    """
Find a root of a function using the bisection method.
"""
    if low > high:
        raise RuntimeError("Bisection low value was higher than the high value!")

    value_low = func(low)

    for iterations in range(1000):
        mid = (low + high) / 2.0
        value_mid = func(mid)

        if abs(value_mid) < tol or (high - low) / 2.0 < tol:
            return mid

        # Extra check for integers
        if high - low == 1:
            return high

        if np.sign(value_mid) == np.sign(value_low):
            low = mid
            value_low = value_mid
        else:
            high = mid

    raise RuntimeError("Bisection failed!")


def sgn(val):
    """
Return the sign of a value.
"""
    return 1 if val > 0 else -1 if val < 0 else 0


def haar_difference_without_abs(angle1, angle2):
    return math.fmod(angle1 - angle2 + np.pi, 2*np.pi) - np.pi



# Conversion utilities

def angle_to_quaternion(angle):
    """
Convert a yaw angle to a quaternion.
"""
    # In Python, we can use scipy or transformations library, but here's a basic implementation
    cy = math.cos(angle * 0.5)
    sy = math.sin(angle * 0.5)
    result = {
        'x': 0.0,
        'y': 0.0,
        'z': sy,
        'w': cy
    }
    return result


def quaternion_to_angle(pose_or_quaternion):
    """
Extract yaw angle from a quaternion or pose with quaternion.
"""
    # Determine if input is a pose or quaternion
    if hasattr(pose_or_quaternion, 'orientation'):  # It's a pose
        q = pose_or_quaternion.orientation
    else:  # It's a quaternion
        q = pose_or_quaternion

    # Extract quaternion components
    if hasattr(q, 'y'):  # Object with attributes
        ysqr = q.y * q.y
        t3 = 2.0 * (q.w * q.z + q.x * q.y)
        t4 = 1.0 - 2.0 * (ysqr + q.z * q.z)
    else:  # Dictionary-like
        ysqr = q['y'] * q['y']
        t3 = 2.0 * (q['w'] * q['z'] + q['x'] * q['y'])
        t4 = 1.0 - 2.0 * (ysqr + q['z'] * q['z'])

    return math.atan2(t3, t4)

class Halfspace:
    def __init__(self, A: np.ndarray, b: float):
        """Halfspace defined by Ax <= b"""
        self.A = A
        self.b = b

# Type definition for StaticObstacle
# In Python, we can use a simple type hint
# StaticObstacle = List[Halfspace]

# Random number generation
class RandomGenerator:
    def __init__(self, seed=-1):
        if seed == -1:
            # Use random seeds
            self.rng__ = random.Random()
            self.rng_int_ = random.Random()
            self.rng_gaussian_ = random.Random()
        else:
            # Use fixed seeds
            self.rng__ = random.Random(seed)
            self.rng_int_ = random.Random(seed)
            self.rng_gaussian_ = random.Random(seed)

        self.epsilon_ = np.finfo(float).eps

    def random(self):
        """Generate a random number between 0 and 1."""
        return self.rng__.random()

    def int(self, max_val):
        """Generate a random integer between 0 and max."""
        return self.rng_int_.randint(0, max_val)

    def gaussian(self, mean, stddev):
        """Generate a random number from a Gaussian distribution."""
        return self.rng_gaussian_.gauss(mean, stddev)

    def uniform_to_gaussian_2d(self, uniform_variables):
        """Convert uniform random variables to Gaussian via Box-Muller."""
        # Temporarily save the first variable
        temp_u1 = uniform_variables[0]

        # Convert the uniform variables to gaussian via Box-Muller
        uniform_variables[0] = math.sqrt(-2 * math.log(temp_u1)) * math.cos(2 * math.pi * uniform_variables[1])
        uniform_variables[1] = math.sqrt(-2 * math.log(temp_u1)) * math.sin(2 * math.pi * uniform_variables[1])

        return uniform_variables

    def bivariate_gaussian(self, mean, major_axis, minor_axis, angle):
        """Generate a random point from a bivariate Gaussian distribution."""
        # Get the rotation matrix
        R = rotation_matrix_from_heading(angle)

        # Generate uniform random numbers in 2D
        u1 = 0
        while u1 <= self.epsilon_:
            u1 = self.rng_gaussian_.random()
        u2 = self.rng_gaussian_.random()
        uniform_samples = np.array([u1, u2])

        # Convert them to a Gaussian
        uniform_samples = self.uniform_to_gaussian_2d(uniform_samples)

        # Convert the semi axes back to gaussians
        SVD = np.array([[major_axis ** 2, 0.0], [0.0, minor_axis ** 2]])

        # Compute Sigma and Cholesky decomposition
        Sigma = R @ SVD @ R.T
        A = np.linalg.cholesky(Sigma)  # Matrix square root

        # Apply transformation to uniform Gaussian samples
        result = A @ uniform_samples + np.array(mean)
        return result


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
        print("Creating tk spline")
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

        print("X contains: {}".format(x))
        print("Y contains: {}".format(y))
        assert len(x) == len(y)
        assert len(x) > 2

        self.m_x = list(x)
        self.m_y = list(y)
        self.m_x_ = list(x)  # Store copies as in original code
        self.m_y_ = list(y)  # Store copies as in original code

        n = len(x)

        # # Check that x is strictly increasing
        # for i in range(n - 1):
        #     is_increasing = self.m_x[i] < self.m_x[i + 1]
        #     print("Is points strictly increasing: {}".format(is_increasing))
        #     assert self.m_x[i] < self.m_x[i + 1]

        if cubic_spline:
            print("Building cubic spline")
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
                # 2*b[n-1]x contains = f''
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

        self.splines = []
        spline = TkSpline()
        spline.set_points(points[0], points[1])
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

    def deriv_normalized(self, t):
        # Get velocity (first derivative) vector
        velocity = self.get_velocity(t)

        # Extract dx and dy components
        dx, dy = velocity[0], velocity[1]

        # Calculate norm (using numpy instead of cd)
        path_norm = np.linalg.norm(velocity)  # or np.sqrt(dx*dx + dy*dy)

        # Return normalized components
        return dx / path_norm, dy / path_norm

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
        #TODO: This needs to be tested
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
            if self.t_vector[i] < t_out < self.t_vector[i + 1]:
                self.closest_segment = i
                return i, t_out

        self.closest_segment = last_segment
        return last_segment, t_out

    def find_closest_s_recursively(self, point, low, high, num_recursions=0):
        """Recursively find the closest parameter on the spline to the given point"""
        #TODO: This needs to be tested
        if abs(high - low) <= 1e-4 or num_recursions > 40:
            if num_recursions > 40:
                LOG_WARN("Recursion count exceeded.")
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

    def get_t_vector(self):
        return self.t_vector

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
        return atan2(self.y_spline.deriv(1, t), self.x_spline.deriv(1, t))

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

class SplineAdapter:
    """Adapter class that provides the old Spline interface using the new Spline implementation"""

    def __init__(self, params, name, num_segments, s):
        """
        Initialize the 1D spline adapter

        Args:
            params: Parameter object with get method
            name: Base name for the spline parameters (e.g., "spline_v")
            num_segments: Number of spline segments
            s: Current spline parameter value (not used in initialization)
        """
        # Create points arrays for the new spline format
        points = [[]]  # Single dimension array for 1D spline
        t_vector = []

        # Debug information
        print(f"Initializing SplineAdapter with {num_segments} segments, name: {name}")

        # Extract all the segment parameters from the params object
        for i in range(num_segments):
            s_start = params.get(f"spline_{i}_start")
            a = params.get(f"{name}{i}_a")
            b = params.get(f"{name}{i}_b")
            c = params.get(f"{name}{i}_c")
            d = params.get(f"{name}{i}_d")

            print(f"Segment {i}: s_start={s_start}, a={a}, b={b}, c={c}, d={d}")

            # Add the start point of this segment
            t_vector.append(s_start)
            points[0].append(d)  # At s=0, only d coefficient matters

            # For the last segment, add the endpoint
            if i == num_segments - 1:
                # Choose a reasonable end point - using 1.0 as a default segment length
                s_end = s_start + 1.0
                # Calculate the value at the endpoint using the polynomial
                s_param = s_end - s_start
                end_value = a * s_param ** 3 + b * s_param ** 2 + c * s_param + d
                t_vector.append(s_end)
                points[0].append(end_value)

        print(f"t_vector: {t_vector}")
        print(f"points: {points}")

        # Initialize the new spline

        self.spline = Spline(points, t_vector)


    def at(self, s):
        """Evaluate the spline at parameter s"""
        return self.spline.get_coordinate(s, 0)

    def deriv(self, s):
        """Get the first derivative at parameter s"""
        velocity = self.spline.get_velocity(s)
        return velocity[0]

    def deriv2(self, s):
        """Get the second derivative at parameter s"""
        acceleration = self.spline.get_acceleration(s)
        return acceleration[0]


class Spline2DAdapter:
    """Adapter class that provides the old Spline2D interface using the new Spline implementation"""

    def __init__(self, params, num_segments, s):
        """
        Initialize the 2D spline adapter

        Args:
            params: Parameter object with get method
            num_segments: Number of spline segments
            s: Current spline parameter value (not used in initialization)
        """
        # Create arrays for x and y points
        points_x = []
        points_y = []
        t_vector = []

        print(f"Initializing Spline2DAdapter with {num_segments} segments")

        # Extract all segment parameters from the params object
        for i in range(num_segments):
            s_start = params.get(f"spline_{i}_start")

            # X spline coefficients
            ax = params.get(f"spline_{i}_ax")
            bx = params.get(f"spline_{i}_bx")
            cx = params.get(f"spline_{i}_cx")
            dx = params.get(f"spline_{i}_dx")

            # Y spline coefficients
            ay = params.get(f"spline_{i}_ay")
            by = params.get(f"spline_{i}_by")
            cy = params.get(f"spline_{i}_cy")
            dy = params.get(f"spline_{i}_dy")

            print(f"Segment {i}: s_start={s_start}")
            print(f"X coeffs: ax={ax}, bx={bx}, cx={cx}, dx={dx}")
            print(f"Y coeffs: ay={ay}, by={by}, cy={cy}, dy={dy}")

            # Add the start point of this segment
            t_vector.append(s_start)
            points_x.append(dx)  # At s=0, only d coefficient matters
            points_y.append(dy)

            # For the last segment, add the endpoint
            if i == num_segments - 1:
                # Choose a reasonable end point - using 1.0 as a default segment length
                s_end = s_start + 1.0
                s_param = s_end - s_start

                # Calculate end values using the polynomial
                x_end = ax * s_param ** 3 + bx * s_param ** 2 + cx * s_param + dx
                y_end = ay * s_param ** 3 + by * s_param ** 2 + cy * s_param + dy

                t_vector.append(s_end)
                points_x.append(x_end)
                points_y.append(y_end)

        print(f"t_vector: {t_vector}")
        print(f"points_x: {points_x}")
        print(f"points_y: {points_y}")

        # Try using TwoDimensionalSpline first (preferred)
        if hasattr(TwoDimensionalSpline, '__init__') and callable(getattr(TwoDimensionalSpline, '__init__')):
            self.spline = TwoDimensionalSpline(points_x, points_y, t_vector)
        else:
            # Fall back to using the base Spline class with 2D points
            self.spline = Spline([points_x, points_y], t_vector)


    def at(self, s):
        """Get the 2D point at parameter s"""
        point = self.spline.get_point(s)
        return point[0], point[1]

    def deriv(self, s):
        """Get the velocity vector at parameter s"""
        velocity = self.spline.get_velocity(s)
        return velocity[0], velocity[1]

    def deriv_normalized(self, s):
        """Get the normalized velocity vector at parameter s"""
        velocity = self.spline.get_velocity(s)
        dx, dy = velocity[0], velocity[1]
        path_norm = cd.sqrt(dx * dx + dy * dy)  # Using casadi's sqrt as in original code

        # Avoid division by zero
        if isinstance(path_norm, (int, float)) and path_norm < 1e-10:
            path_norm = 1e-10

        return dx / path_norm, dy / path_norm

    def deriv2(self, s):
        """Get the acceleration vector at parameter s"""
        acceleration = self.spline.get_acceleration(s)
        return acceleration[0], acceleration[1]

    def get_curvature(self, s):
        """Get the curvature at parameter s"""
        acceleration = self.spline.get_acceleration(s)
        path_x_deriv2, path_y_deriv2 = acceleration[0], acceleration[1]
        return cd.sqrt(path_x_deriv2 * path_x_deriv2 + path_y_deriv2 * path_y_deriv2)


# Create a module import wrapper to handle import errors gracefully
def create_spline_adapter(params, name, num_segments, s):
    try:
        return SplineAdapter(params, name, num_segments, s)
    except Exception as e:
        print(f"Error creating SplineAdapter: {e}")
        raise


def create_spline2d_adapter(params, num_segments, s):
    try:
        return Spline2DAdapter(params, num_segments, s)
    except Exception as e:
        print(f"Error creating Spline2DAdapter: {e}")
        raise


import casadi as cd
import numpy as np


class CasadiSpline2D:
    """CasADi-compatible 2D spline class designed for optimization problems"""

    def __init__(self, params, num_segments):
        """Initialize with polynomial coefficients for each segment

        Args:
            params: Parameter object with get method
            num_segments: Number of spline segments
        """
        self.num_segments = num_segments

        # Store segment data
        self.segments = []
        for i in range(num_segments):
            s_start = params.get(f"spline_{i}_start")

            # X spline coefficients
            ax = params.get(f"spline_{i}_ax")
            bx = params.get(f"spline_{i}_bx")
            cx = params.get(f"spline_{i}_cx")
            dx = params.get(f"spline_{i}_dx")

            # Y spline coefficients
            ay = params.get(f"spline_{i}_ay")
            by = params.get(f"spline_{i}_by")
            cy = params.get(f"spline_{i}_cy")
            dy = params.get(f"spline_{i}_dy")

            # Store segment data
            self.segments.append({
                's_start': s_start,
                'ax': ax, 'bx': bx, 'cx': cx, 'dx': dx,
                'ay': ay, 'by': by, 'cy': cy, 'dy': dy
            })

            print(f"Segment {i}: s_start={s_start}")
            print(f"X coeffs: ax={ax}, bx={bx}, cx={cx}, dx={dx}")
            print(f"Y coeffs: ay={ay}, by={by}, cy={cy}, dy={dy}")

        # Compute s_end for each segment (used only for visualization/debugging)
        for i in range(num_segments - 1):
            self.segments[i]['s_end'] = self.segments[i + 1]['s_start']

        # For the last segment, set a default end value
        if num_segments > 0:
            last_segment = self.segments[-1]
            last_segment['s_end'] = last_segment['s_start'] + 1.0  # Default

    def find_segment(self, s):
        """Find which segment contains parameter s

        Important: This function returns symbolic expressions for use with CasADi.
        It uses a technique that avoids branching which CasADi can't handle.

        Args:
            s: Spline parameter value

        Returns:
            segment_idx: Symbolic expression that evaluates to segment index
            local_s: Local s parameter within the segment
        """
        # Initialize with values for outside the spline range
        segment_idx = 0
        local_s = s - self.segments[0]['s_start']

        # For each segment, create indicators if s is in that segment
        for i in range(self.num_segments):
            s_start = self.segments[i]['s_start']
            s_end = self.segments[i]['s_end'] if i < self.num_segments - 1 else float('inf')

            # Calculate local_s for this segment
            segment_local_s = s - s_start

            # Create indicator (1 if s is in this segment, 0 otherwise)
            # We use smoothed conditionals that are differentiable
            in_segment = smooth_conditional(s >= s_start) * smooth_conditional(s < s_end)

            # Update the segment index and local_s using the indicator
            segment_idx = segment_idx * (1 - in_segment) + i * in_segment
            local_s = local_s * (1 - in_segment) + segment_local_s * in_segment

        return segment_idx, local_s

    def at(self, s):
        """Evaluate the spline at parameter s

        Args:
            s: Spline parameter value

        Returns:
            (x, y): Coordinates at parameter s
        """
        # Initialize result variables
        x_result = 0
        y_result = 0

        # For each segment, evaluate and use conditional logic to select the right one
        for i in range(self.num_segments):
            segment = self.segments[i]
            s_start = segment['s_start']
            s_end = segment['s_end'] if i < self.num_segments - 1 else float('inf')

            # Calculate local s (parameter relative to segment start)
            local_s = s - s_start

            # Evaluate polynomial for this segment
            x_segment = segment['ax'] * local_s ** 3 + segment['bx'] * local_s ** 2 + segment['cx'] * local_s + segment[
                'dx']
            y_segment = segment['ay'] * local_s ** 3 + segment['by'] * local_s ** 2 + segment['cy'] * local_s + segment[
                'dy']

            # Indicator for whether s is in this segment
            in_segment = smooth_conditional(s >= s_start) * smooth_conditional(s < s_end)

            # Update result using the indicator
            x_result = x_result * (1 - in_segment) + x_segment * in_segment
            y_result = y_result * (1 - in_segment) + y_segment * in_segment

        return x_result, y_result

    def deriv(self, s):
        """Get the first derivatives at parameter s

        Args:
            s: Spline parameter value

        Returns:
            (dx/ds, dy/ds): Derivatives at parameter s
        """
        # Initialize result variables
        dx_result = 0
        dy_result = 0

        # For each segment, evaluate and use conditional logic to select the right one
        for i in range(self.num_segments):
            segment = self.segments[i]
            s_start = segment['s_start']
            s_end = segment['s_end'] if i < self.num_segments - 1 else float('inf')

            # Calculate local s (parameter relative to segment start)
            local_s = s - s_start

            # Evaluate derivative of polynomial for this segment
            dx_segment = 3 * segment['ax'] * local_s ** 2 + 2 * segment['bx'] * local_s + segment['cx']
            dy_segment = 3 * segment['ay'] * local_s ** 2 + 2 * segment['by'] * local_s + segment['cy']

            # Indicator for whether s is in this segment
            in_segment = smooth_conditional(s >= s_start) * smooth_conditional(s < s_end)

            # Update result using the indicator
            dx_result = dx_result * (1 - in_segment) + dx_segment * in_segment
            dy_result = dy_result * (1 - in_segment) + dy_segment * in_segment

        return dx_result, dy_result

    def deriv_normalized(self, s):
        """Get the normalized derivatives at parameter s

        Args:
            s: Spline parameter value

        Returns:
            (dx/ds, dy/ds): Normalized derivatives at parameter s
        """
        dx, dy = self.deriv(s)
        norm = cd.sqrt(dx ** 2 + dy ** 2)

        # Avoid division by zero
        safe_norm = cd.fmax(norm, 1e-10)

        return dx / safe_norm, dy / safe_norm

    def deriv2(self, s):
        """Get the second derivatives at parameter s

        Args:
            s: Spline parameter value

        Returns:
            (d²x/ds², d²y/ds²): Second derivatives at parameter s
        """
        # Initialize result variables
        d2x_result = 0
        d2y_result = 0

        # For each segment, evaluate and use conditional logic to select the right one
        for i in range(self.num_segments):
            segment = self.segments[i]
            s_start = segment['s_start']
            s_end = segment['s_end'] if i < self.num_segments - 1 else float('inf')

            # Calculate local s (parameter relative to segment start)
            local_s = s - s_start

            # Evaluate second derivative of polynomial for this segment
            d2x_segment = 6 * segment['ax'] * local_s + 2 * segment['bx']
            d2y_segment = 6 * segment['ay'] * local_s + 2 * segment['by']

            # Indicator for whether s is in this segment
            in_segment = smooth_conditional(s >= s_start) * smooth_conditional(s < s_end)

            # Update result using the indicator
            d2x_result = d2x_result * (1 - in_segment) + d2x_segment * in_segment
            d2y_result = d2y_result * (1 - in_segment) + d2y_segment * in_segment

        return d2x_result, d2y_result


class CasadiSpline:
    """CasADi-compatible 1D spline class designed for optimization problems"""

    def __init__(self, params, name, num_segments):
        """Initialize with polynomial coefficients for each segment

        Args:
            params: Parameter object with get method
            name: Base name for the spline parameters (e.g., "width_left")
            num_segments: Number of spline segments
        """
        self.num_segments = num_segments

        # Store segment data
        self.segments = []
        for i in range(num_segments):
            s_start = params.get(f"spline_{i}_start")
            a = params.get(f"{name}_{i}_a")
            b = params.get(f"{name}_{i}_b")
            c = params.get(f"{name}_{i}_c")
            d = params.get(f"{name}_{i}_d")

            # Store segment data
            self.segments.append({
                's_start': s_start,
                'a': a, 'b': b, 'c': c, 'd': d
            })

            print(f"Segment {i} ({name}): s_start={s_start}, a={a}, b={b}, c={c}, d={d}")

        # Compute s_end for each segment (used only for visualization/debugging)
        for i in range(num_segments - 1):
            self.segments[i]['s_end'] = self.segments[i + 1]['s_start']

        # For the last segment, set a default end value
        if num_segments > 0:
            last_segment = self.segments[-1]
            last_segment['s_end'] = last_segment['s_start'] + 1.0  # Default

    def at(self, s):
        """Evaluate the spline at parameter s

        Args:
            s: Spline parameter value

        Returns:
            Value at parameter s
        """
        # Initialize result variable
        result = 0

        # For each segment, evaluate and use conditional logic to select the right one
        for i in range(self.num_segments):
            segment = self.segments[i]
            s_start = segment['s_start']
            s_end = segment['s_end'] if i < self.num_segments - 1 else float('inf')

            # Calculate local s (parameter relative to segment start)
            local_s = s - s_start

            # Evaluate polynomial for this segment
            segment_value = segment['a'] * local_s ** 3 + segment['b'] * local_s ** 2 + segment['c'] * local_s + \
                            segment['d']

            # Indicator for whether s is in this segment
            in_segment = smooth_conditional(s >= s_start) * smooth_conditional(s < s_end)

            # Update result using the indicator
            result = result * (1 - in_segment) + segment_value * in_segment

        return result

    def deriv(self, s):
        """Get the first derivative at parameter s

        Args:
            s: Spline parameter value

        Returns:
            First derivative at parameter s
        """
        # Initialize result variable
        result = 0

        # For each segment, evaluate and use conditional logic to select the right one
        for i in range(self.num_segments):
            segment = self.segments[i]
            s_start = segment['s_start']
            s_end = segment['s_end'] if i < self.num_segments - 1 else float('inf')

            # Calculate local s (parameter relative to segment start)
            local_s = s - s_start

            # Evaluate derivative of polynomial for this segment
            segment_value = 3 * segment['a'] * local_s ** 2 + 2 * segment['b'] * local_s + segment['c']

            # Indicator for whether s is in this segment
            in_segment = smooth_conditional(s >= s_start) * smooth_conditional(s < s_end)

            # Update result using the indicator
            result = result * (1 - in_segment) + segment_value * in_segment

        return result

    def deriv2(self, s):
        """Get the second derivative at parameter s

        Args:
            s: Spline parameter value

        Returns:
            Second derivative at parameter s
        """
        # Initialize result variable
        result = 0

        # For each segment, evaluate and use conditional logic to select the right one
        for i in range(self.num_segments):
            segment = self.segments[i]
            s_start = segment['s_start']
            s_end = segment['s_end'] if i < self.num_segments - 1 else float('inf')

            # Calculate local s (parameter relative to segment start)
            local_s = s - s_start

            # Evaluate second derivative of polynomial for this segment
            segment_value = 6 * segment['a'] * local_s + 2 * segment['b']

            # Indicator for whether s is in this segment
            in_segment = smooth_conditional(s >= s_start) * smooth_conditional(s < s_end)

            # Update result using the indicator
            result = result * (1 - in_segment) + segment_value * in_segment

        return result


def smooth_conditional(condition):
    """A smooth approximation of an indicator function for CasADi

    This is a differentiable approximation of the step function:
    - Returns values close to 1 when condition > 0
    - Returns values close to 0 when condition < 0
    - Smoothly transitions around condition = 0

    Args:
        condition: A CasADi expression

    Returns:
        A smooth approximation of the step function applied to the condition
    """
    # Sigmoid function with scaling for steepness
    steepness = 1000  # Higher values make transition sharper but may cause numerical issues
    return 1 / (1 + cd.exp(-steepness * condition))


def haar_difference_without_abs(a, b):
    """Calculate the smallest angle difference between two angles

    Args:
        a, b: Angles in radians

    Returns:
        Smallest signed angle difference from b to a
    """
    diff = a - b
    # Normalize to [-π, π]
    return cd.fmod(diff + cd.pi, 2 * cd.pi) - cd.pi


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

class Hyperplane2D:
    """Class representing a 2D hyperplane (line) with a point and normal vector."""

    def __init__(self, point: np.ndarray, normal: np.ndarray):
        """
        Initialize hyperplane with a point and normal vector.

        Args:
            point: A point on the hyperplane
            normal: Normal vector to the hyperplane
        """
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize

    def distance(self, point: np.ndarray) -> float:
        """Calculate signed distance from point to hyperplane."""
        return np.dot(self.normal, point - self.point)


class Polyhedron:
    """Class representing a polyhedron (polygon in 2D) as a set of hyperplanes."""

    def __init__(self, dim: int = 2):
        """Initialize empty polyhedron."""
        self.dim = dim
        self.hyperplanes_list = []

    def add(self, hp: Hyperplane2D) -> None:
        """Add a hyperplane to the polyhedron."""
        self.hyperplanes_list.append(hp)

    def hyperplanes(self) -> List[Hyperplane2D]:
        """Get the list of hyperplanes."""
        return self.hyperplanes_list

    @property
    def vertices(self) -> List[np.ndarray]:
        """Calculate vertices of the polyhedron (in 2D)."""
        if self.dim != 2 or len(self.hyperplanes_list) < 3:
            return []

        # Find vertices by intersecting adjacent hyperplanes
        vertices = []
        n = len(self.hyperplanes_list)

        for i in range(n):
            h1 = self.hyperplanes_list[i]
            h2 = self.hyperplanes_list[(i + 1) % n]

            # Find intersection of two lines
            # Using linear algebra to solve:
            # p = p1 + t1 * (normal1 rotated 90°)
            # p = p2 + t2 * (normal2 rotated 90°)

            # Rotate normals by 90°
            dir1 = np.array([-h1.normal[1], h1.normal[0]])
            dir2 = np.array([-h2.normal[1], h2.normal[0]])

            # Set up system of equations
            A = np.column_stack((dir1, -dir2))
            if np.linalg.det(A) == 0:  # Parallel lines
                continue

            b = h2.point - h1.point
            t = np.linalg.solve(A, b)

            vertex = h1.point + t[0] * dir1
            vertices.append(vertex)

        return vertices


class Ellipsoid:
    """Class representing an ellipsoid."""

    def __init__(self, center: np.ndarray = None, axes: np.ndarray = None, rotation: np.ndarray = None):
        """
        Initialize ellipsoid with center, semi-axes lengths, and rotation.

        Args:
            center: Center point of the ellipsoid
            axes: Semi-axes lengths
            rotation: Rotation matrix for the ellipsoid
        """
        self.center = np.zeros(2) if center is None else np.array(center)
        self.axes = np.ones(2) if axes is None else np.array(axes)
        self.rotation = np.eye(2) if rotation is None else np.array(rotation)

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point is inside the ellipsoid."""
        p_centered = point - self.center
        p_rotated = self.rotation.T @ p_centered

        # Check if point satisfies ellipsoid equation
        return sum((p_rotated / self.axes) ** 2) <= 1


class LineSegment:
    """Class representing a line segment."""

    def __init__(self, p1: np.ndarray, p2: np.ndarray):
        """
        Initialize line segment with two endpoints.

        Args:
            p1: First endpoint
            p2: Second endpoint
        """
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.dim = len(p1)
        self.local_bbox = np.zeros(self.dim)
        self.obs = []
        self.ellipsoid = None
        self.polyhedron = None

    def set_local_bbox(self, bbox: np.ndarray) -> None:
        """Set local bounding box dimensions."""
        self.local_bbox = np.array(bbox)

    def set_obs(self, obs: List[np.ndarray]) -> None:
        """Set obstacle points."""
        self.obs = [np.array(o) for o in obs]

    def dilate(self, offset_x: float = 0.0) -> None:
        """
        Dilate the line segment to create ellipsoid and polyhedron.

        Args:
            offset_x: Offset added to the long semi-axis
        """
        # Calculate ellipsoid parameters
        center = (self.p1 + self.p2) / 2
        direction = self.p2 - self.p1
        length = np.linalg.norm(direction)

        if length < 1e-6:  # Avoid division by zero
            direction = np.array([1.0, 0.0])
            length = 1.0

        direction = direction / length  # Normalize

        # Semi-major axis is half the length plus offset
        a = length / 2 + offset_x

        # Find closest obstacle to determine semi-minor axis
        b = float('inf')
        for obs_point in self.obs:
            # Skip obstacles outside local bounding box
            if any(abs(obs_point[i] - center[i]) > self.local_bbox[i] for i in range(self.dim)):
                continue

            # Project obstacle onto line
            t = np.dot(obs_point - self.p1, direction)
            t = max(0, min(length, t))  # Clamp to line segment
            proj = self.p1 + t * direction

            # Distance from obstacle to line
            dist = np.linalg.norm(obs_point - proj)
            b = min(b, dist)

        # If no close obstacles found, use default value
        if b == float('inf'):
            b = a  # Default to circle if no obstacles

        # Create rotation matrix (2D rotation)
        rot = np.column_stack((direction, np.array([-direction[1], direction[0]])))

        # Create ellipsoid
        self.ellipsoid = Ellipsoid(center, np.array([a, b]), rot)

        # Create polyhedron from ellipsoid
        self.polyhedron = self._generate_polyhedron(center, a, b, rot)

    def _generate_polyhedron(self, center, a, b, rot, num_sides=8):
        """
        Generate polyhedron approximation of the ellipsoid.

        Args:
            center: Center of ellipsoid
            a: Semi-major axis length
            b: Semi-minor axis length
            rot: Rotation matrix
            num_sides: Number of sides for approximation

        Returns:
            Polyhedron object
        """
        poly = Polyhedron(self.dim)

        # Generate approximating hyperplanes around ellipsoid
        for i in range(num_sides):
            angle = 2 * np.pi * i / num_sides
            # Point on ellipsoid
            p_ellipse = np.array([a * np.cos(angle), b * np.sin(angle)])
            p_rotated = rot @ p_ellipse
            point = center + p_rotated

            # Normal to ellipsoid at this point (gradient of ellipsoid function)
            normal = np.array([np.cos(angle) / a, np.sin(angle) / b])
            normal = rot @ normal
            normal = normal / np.linalg.norm(normal)  # Normalize

            poly.add(Hyperplane2D(point, normal))

        return poly

    def get_ellipsoid(self) -> Ellipsoid:
        """Get the ellipsoid."""
        return self.ellipsoid

    def get_polyhedron(self) -> Polyhedron:
        """Get the polyhedron."""
        return self.polyhedron


class LinearConstraint:
    """Class representing linear constraints Ax ≤ b."""

    def __init__(self, point: np.ndarray = None, hyperplanes: List[Hyperplane2D] = None):
        """
        Initialize linear constraints.

        Args:
            point: Reference point
            hyperplanes: List of hyperplanes
        """
        self.point = np.zeros(2) if point is None else np.array(point)
        self.A_ = np.zeros((0, 2))  # Empty matrix initially
        self.b_ = np.zeros(0)  # Empty vector initially

        if hyperplanes:
            self._setup_constraints(hyperplanes)

    def _setup_constraints(self, hyperplanes: List[Hyperplane2D]) -> None:
        """Setup A and b matrices from hyperplanes."""
        n = len(hyperplanes)
        self.A_ = np.zeros((n, 2))
        self.b_ = np.zeros(n)

        for i, hp in enumerate(hyperplanes):
            self.A_[i] = hp.normal
            self.b_[i] = np.dot(hp.normal, hp.point)


class EllipsoidDecomp:
    """
    EllipsoidDecomp takes input as a given path and finds the Safe Flight Corridor
    around it using Ellipsoids.
    """

    def __init__(self, origin: np.ndarray = None, dim: np.ndarray = None):
        """
        Initialize EllipsoidDecomp.

        Args:
            origin: The origin of the global bounding box
            dim: The dimension of the global bounding box
        """
        self.obs_ = []
        self.path_ = []
        self.lines_ = []
        self.ellipsoids_ = []
        self.polyhedrons_ = []
        self.local_bbox_ = np.zeros(2)

        # Global bounding box
        self.global_bbox_min_ = np.zeros(2) if origin is None else np.array(origin)
        self.global_bbox_max_ = np.zeros(2) if origin is None or dim is None else np.array(origin) + np.array(dim)

    def set_obs(self, obs: List[np.ndarray]) -> None:
        """Set obstacle points."""
        print("At top of set obs")
        self.obs_ = [np.array(o) for o in obs]

    def set_local_bbox(self, bbox: np.ndarray) -> None:
        """Set dimension of local bounding box."""
        self.local_bbox_ = np.array(bbox)

    def get_path(self) -> List[np.ndarray]:
        """Get the path that is used for dilation."""
        return self.path_

    def get_polyhedrons(self) -> List[Polyhedron]:
        """Get the Safe Flight Corridor."""
        return self.polyhedrons_

    def get_ellipsoids(self) -> List[Ellipsoid]:
        """Get the ellipsoids."""
        return self.ellipsoids_

    def get_constraints(self) -> List[LinearConstraint]:
        """
        Get the constraints of SFC as Ax ≤ b.

        Returns:
            List of LinearConstraint objects
        """
        constraints = []
        for i in range(len(self.polyhedrons_)):
            if i + 1 >= len(self.path_):
                continue

            pt = (self.path_[i] + self.path_[i + 1]) / 2
            constraints.append(LinearConstraint(pt, self.polyhedrons_[i].hyperplanes()))

        return constraints

    def set_constraints(self, constraints_out: list, offset: float = 0.0) -> None:
        """
        Set constraints, primarily for compatibility with original code.

        Args:
            constraints_out: Output list to store constraints
            offset: Offset value
        """
        constraints = self.get_constraints()

        # Make sure the output list has enough space
        while len(constraints_out) < len(constraints):
            constraints_out.append(None)

        # Copy constraints to output
        for i, constraint in enumerate(constraints):
            constraints_out[i] = constraint

    def dilate(self, path: List[np.ndarray], offset_x: float = 0.0, safe_check: bool = True) -> None:
        """
        Decomposition thread.

        Args:
            path: The path to dilate
            offset_x: Offset added to the long semi-axis
            safe_check: Safety check flag (not used in this implementation)
        """
        if len(path) <= 1:
            return

        N = len(path) - 1
        self.lines_ = []
        self.ellipsoids_ = []
        self.polyhedrons_ = []

        for i in range(N):
            line = LineSegment(path[i], path[i + 1])
            line.set_local_bbox(self.local_bbox_)
            line.set_obs(self.obs_)
            line.dilate(offset_x)

            self.lines_.append(line)
            self.ellipsoids_.append(line.get_ellipsoid())
            self.polyhedrons_.append(line.get_polyhedron())

            # Add global bounding box if defined
            if np.linalg.norm(self.global_bbox_min_) != 0 or np.linalg.norm(self.global_bbox_max_) != 0:
                self._add_global_bbox(self.polyhedrons_[-1])

        self.path_ = [np.array(p) for p in path]

    def _add_global_bbox(self, poly: Polyhedron) -> None:
        """
        Add global bounding box constraints to polyhedron.

        Args:
            poly: Polyhedron to modify
        """
        # Add bound along X axis
        poly.add(Hyperplane2D(np.array([self.global_bbox_max_[0], 0]), np.array([1, 0])))
        poly.add(Hyperplane2D(np.array([self.global_bbox_min_[0], 0]), np.array([-1, 0])))

        # Add bound along Y axis
        poly.add(Hyperplane2D(np.array([0, self.global_bbox_max_[1]]), np.array([0, 1])))
        poly.add(Hyperplane2D(np.array([0, self.global_bbox_min_[1]]), np.array([0, -1])))


# Define the 2D version explicitly
class EllipsoidDecomp2D(EllipsoidDecomp):
    """2D version of EllipsoidDecomp."""

    def __init__(self, origin=None, dim=None):
        super().__init__(origin, dim)


class SplineSegment:

    def __init__(self, param, name, spline_nr):
        print("Init spline segment", name, spline_nr)
        # Retrieve spline values from the parameters (stored as multi parameter by name)
        self.a = param.get(f"{name}_{spline_nr}a")
        self.b = param.get(f"{name}_{spline_nr}b")
        self.c = param.get(f"{name}_{spline_nr}c")
        self.d = param.get(f"{name}_{spline_nr}d")

        self.s_start = param.get(f"spline_{spline_nr}_start")

    def at(self, spline_index):
        s = spline_index - self.s_start
        return self.a * s * s * s + self.b * s * s + self.c * s + self.d

    def deriv(self, spline_index):
        s = spline_index - self.s_start
        return 3 * self.a * s * s + 2 * self.b * s + self.c

    def deriv2(self, spline_index):
        s = spline_index - self.s_start
        return 6 * self.a * s + 2 * self.b


class SplineBySegment:
    def __init__(self, params, name, get_num_segments, s):
        self.splines = []  # Classes containing the splines
        self.lambdas = []  # Merges splines
        for i in range(get_num_segments):
            self.splines.append(SplineSegment(params, f"{name}", i))

            # No lambda for the first segment (it is not glued to anything prior)
            if i > 0:
                self.lambdas.append(1.0 / (1.0 + np.exp((s - self.splines[-1].s_start + 0.02) / 0.1)))  # Sigmoid

    def at(self, s):
        # Iteratively glue segments together
        value = self.splines[-1].at(s)
        for k in range(len(self.splines) - 1, 0, -1):
            value = self.lambdas[k - 1] * self.splines[k - 1].at(s) + (1.0 - self.lambdas[k - 1]) * value
        return value

    def deriv(self, s):
        value = self.splines[-1].deriv(s)
        for k in range(len(self.splines) - 1, 0, -1):
            value = self.lambdas[k - 1] * self.splines[k - 1].deriv(s) + (1.0 - self.lambdas[k - 1]) * value
        return value

    def deriv2(self, s):
        value = self.splines[-1].deriv2(s)
        for k in range(len(self.splines) - 1, 0, -1):
            value = self.lambdas[k - 1] * self.splines[k - 1].deriv2(s) + (1.0 - self.lambdas[k - 1]) * value
        return value


class Spline2DBySegment:

    def __init__(self, params, get_num_segments, s):
        self.spline_x = SplineBySegment(params, "spline_x", get_num_segments, s)
        self.spline_y = SplineBySegment(params, "spline_y", get_num_segments, s)

    def at(self, s):
        return self.spline_x.at(s), self.spline_y.at(s)

    def deriv(self, s):
        return self.spline_x.deriv(s), self.spline_y.deriv(s)

    def deriv_normalized(self, s):
        dx = self.spline_x.deriv(s)
        dy = self.spline_y.deriv(s)
        path_norm = cd.sqrt(dx * dx + dy * dy)

        return dx / path_norm, dy / path_norm

    def deriv2(self, s):
        return self.spline_x.deriv2(s), self.spline_y.deriv2(s)

    def get_curvature(self, s):
        path_x_deriv2 = self.spline_x.deriv2(s)
        path_y_deriv2 = self.spline_y.deriv2(s)

        return cd.sqrt(path_x_deriv2 * path_x_deriv2 + path_y_deriv2 * path_y_deriv2)  # + 0.0000000001) # Max = 1e2


