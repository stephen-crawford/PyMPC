import random

import casadi as cd
import numpy as np
from scipy.stats import chi2

from utils.utils import LOG_WARN, LOG_DEBUG


###### General math utilities ######

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

def safe_norm(x, y):
    norm = cd.sqrt(x ** 2 + y ** 2)
    # Add a small epsilon to prevent division by zero
    safe_norm = cd.fmax(norm, 1e-6)
    return safe_norm


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

def casadi_rotation_matrix(theta):
    return cd.vertcat(
        cd.horzcat(cd.cos(theta), -cd.sin(theta)),
        cd.horzcat(cd.sin(theta),  cd.cos(theta))
    )

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


def chi_square_quantile(dof: int, alpha: float) -> float:
    """Compute chi-square quantile for given degrees of freedom and confidence level."""
    return chi2.ppf(alpha, dof)


def haar_difference_without_abs(a, b):
    """Calculate the smallest angle between two angles in a way compatible with CasADi symbolics."""
    diff = a - b
    # Use casadi's mod function for symbolic compatibility instead of % operator
    return diff - 2 * cd.pi * cd.floor((diff + cd.pi) / (2 * cd.pi))

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

##### Geometry classes #####

class Halfspace:
    def __init__(self, A: np.ndarray, b: float):
        """Halfspace defined by Ax <= b"""
        self.A = A
        self.b = b


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




###### Decomposition Utils ######

"""
Decomposition Library - Python Port
A geometric decomposition library for path planning using ellipsoids and polyhedra.
Converted from C++ to Python using NumPy for linear algebra operations.
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Constants and type aliases
EPSILON = 1e-10


# ANSI color codes for printing
class Colors:
    RED = "\x1b[1;31m"
    GREEN = "\x1b[1;32m"
    YELLOW = "\x1b[1;33m"
    BLUE = "\x1b[1;34m"
    MAGENTA = "\x1b[1;35m"
    CYAN = "\x1b[1;36m"
    RESET = "\x1b[0m"


# Type aliases for clarity
Vec2f = np.ndarray  # shape (2,)
Vec3f = np.ndarray  # shape (3,)
Mat2f = np.ndarray  # shape (2, 2)
Mat3f = np.ndarray  # shape (3, 3)
Quatf = np.ndarray  # quaternion as [w, x, y, z]


class DataUtils:
    """Utility functions for data manipulation"""

    @staticmethod
    def transform_vec(vectors: List[np.ndarray], transform: np.ndarray) -> List[np.ndarray]:
        """Transform a list of vectors using given transformation matrix"""
        return [transform @ v for v in vectors]

    @staticmethod
    def total_distance(vectors: List[np.ndarray]) -> float:
        """Calculate total distance along a path"""
        if len(vectors) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(1, len(vectors)):
            total_dist += np.linalg.norm(vectors[i] - vectors[i - 1])
        return total_dist


@dataclass
class Hyperplane:
    """Hyperplane class defined by a point and normal vector"""
    p: np.ndarray  # Point on the plane
    n: np.ndarray  # Normal vector (directional)

    def signed_dist(self, pt: np.ndarray) -> float:
        """Calculate signed distance from point to hyperplane"""
        return np.dot(self.n, pt - self.p)

    def dist(self, pt: np.ndarray) -> float:
        """Calculate absolute distance from point to hyperplane"""
        return abs(self.signed_dist(pt))


class Polyhedron:
    """Polyhedron class defined by hyperplanes"""

    def __init__(self, hyperplanes: Optional[List[Hyperplane]] = None):
        self.hyperplanes_ = hyperplanes or []

    def add(self, hyperplane: Hyperplane):
        """Add a hyperplane to the polyhedron"""
        self.hyperplanes_.append(hyperplane)

    def inside(self, pt: np.ndarray) -> bool:
        """Check if point is inside polyhedron (non-exclusive)"""
        for hp in self.hyperplanes_:
            if hp.signed_dist(pt) > EPSILON:
                return False
        return True

    def points_inside(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Filter points that are inside the polyhedron"""
        return [pt for pt in points if self.inside(pt)]

    def hyperplanes(self) -> List[Hyperplane]:
        """Get the hyperplane array"""
        return self.hyperplanes_

    def cal_normals(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Calculate normals for visualization"""
        return [(hp.p, hp.n) for hp in self.hyperplanes_]

    def __str__(self):
        return f"Polyhedron with {str(self.hyperplanes())}"

    def __repr__(self):
        return self.__str__()


class LinearConstraint:
    """Linear constraint representation as Ax <= b"""

    def __init__(self, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None,
                 p0: Optional[np.ndarray] = None, hyperplanes: Optional[List[Hyperplane]] = None):
        if A is not None and b is not None:
            self.A_ = A
            self.b_ = b
        elif p0 is not None and hyperplanes is not None:
            self._construct_from_hyperplanes(p0, hyperplanes)
        else:
            self.A_ = np.array([])
            self.b_ = np.array([])

    def _construct_from_hyperplanes(self, p0: np.ndarray, hyperplanes: List[Hyperplane]):
        """Construct from inside point and hyperplanes"""
        size = len(hyperplanes)
        dim = len(p0)
        A = np.zeros((size, dim))
        b = np.zeros(size)

        for i, hp in enumerate(hyperplanes):
            n = hp.n.copy()
            c = np.dot(hp.p, n)
            if np.dot(n, p0) - c > 0:
                n = -n
                c = -c
            A[i] = n
            b[i] = c

        self.A_ = A
        self.b_ = b

    def inside(self, pt: np.ndarray) -> bool:
        """Check if point satisfies constraints"""
        d = self.A_ @ pt - self.b_
        return np.all(d <= 0)

    def A(self) -> np.ndarray:
        return self.A_

    def b(self) -> np.ndarray:
        return self.b_

    def __str__(self):
        return f"LinearConstraint with A: {self.A_}, b: {self.b_}"

    def __repr__(self):
        return self.__str__()


class Ellipsoid:
    """Ellipsoid class defined by matrix C and center d"""

    def __init__(self, C: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None):
        self.C_ = C if C is not None else np.eye(2)
        self.d_ = d if d is not None else np.zeros(2)

    def dist(self, pt: np.ndarray) -> float:
        """Calculate distance to center (normalized by ellipsoid)"""
        LOG_DEBUG("Trying to calculate distance for ellipsoid with C: {}, and d: {}".format(self.C_, self.d_))
        try:
            C_inv = np.linalg.inv(self.C_)
        except np.linalg.LinAlgError:
            return 100

        return np.linalg.norm(C_inv @ (pt - self.d_))

    def inside(self, pt: np.ndarray) -> bool:
        """Check if point is inside ellipsoid (non-exclusive)"""
        return self.dist(pt) <= 1.0

    def points_inside(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """Filter points inside ellipsoid"""
        return [pt for pt in points if self.inside(pt)]

    def closest_point(self, points: List[np.ndarray]) -> np.ndarray:
        """Find closest point from a list"""
        if not points:
            return np.zeros_like(self.d_)

        min_dist = float('inf')
        closest_pt = points[0]

        for pt in points:
            d = self.dist(pt)
            if d < min_dist:
                min_dist = d
                closest_pt = pt

        return closest_pt

    def closest_hyperplane(self, points: List[np.ndarray]) -> Hyperplane:
        """Find closest hyperplane from closest point"""
        closest_pt = self.closest_point(points)
        C_inv = np.linalg.inv(self.C_)
        n = C_inv @ C_inv.T @ (closest_pt - self.d_)
        n_normalized = n / np.linalg.norm(n)
        return Hyperplane(closest_pt, n_normalized)

    def sample_2d(self, num: int) -> List[np.ndarray]:
        """Sample points along 2D ellipsoid contour"""
        if len(self.d_) != 2:
            raise ValueError("sample_2d only works for 2D ellipsoids")

        points = []
        dyaw = 2 * math.pi / num

        for i in range(num):
            yaw = i * dyaw
            pt = np.array([math.cos(yaw), math.sin(yaw)])
            points.append(self.C_ @ pt + self.d_)

        return points

    def volume(self) -> float:
        """Get ellipsoid volume (determinant of C)"""
        return np.linalg.det(self.C_)

    def C(self) -> np.ndarray:
        return self.C_

    def d(self) -> np.ndarray:
        return self.d_

    def __str__(self):
        return f"Ellipsoid with C: {self.C_}, d: {self.d_}"

    def __repr__(self):
        return self.__str__()

class GeometricUtils:
    """Geometric utility functions"""

    @staticmethod
    def eigen_values(A: np.ndarray) -> np.ndarray:
        """Calculate eigenvalues of matrix"""
        return np.linalg.eigvals(A)

    @staticmethod
    def vec2_to_rotation(v: np.ndarray) -> np.ndarray:
        """Calculate 2D rotation matrix from vector"""
        yaw = math.atan2(v[1], v[0])
        return np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])

    @staticmethod
    def vec3_to_rotation(v: np.ndarray) -> np.ndarray:
        """Calculate 3D rotation matrix from vector (zero roll)"""
        rpy = np.array([0,
                        math.atan2(-v[2], np.linalg.norm(v[:2])),
                        math.atan2(v[1], v[0])])

        # Convert RPY to rotation matrix
        roll, pitch, yaw = rpy

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])

        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])

        return Rz @ Ry @ Rx

    @staticmethod
    def sort_pts_2d(pts: List[np.ndarray]) -> List[np.ndarray]:
        """Sort 2D points in counter-clockwise order"""
        if not pts:
            return pts

        # Calculate center
        avg = np.mean(pts, axis=0)

        # Sort by angle
        def angle_key(pt):
            return math.atan2(pt[1] - avg[1], pt[0] - avg[0])

        return sorted(pts, key=angle_key)

    @staticmethod
    def line_intersect_2d(line1: Tuple[np.ndarray, np.ndarray],
                          line2: Tuple[np.ndarray, np.ndarray]) -> Tuple[bool, np.ndarray]:
        """Find intersection between two 2D lines"""
        v1, p1 = line1
        v2, p2 = line2

        a1 = -v1[1]
        b1 = v1[0]
        c1 = a1 * p1[0] + b1 * p1[1]

        a2 = -v2[1]
        b2 = v2[0]
        c2 = a2 * p2[0] + b2 * p2[1]

        det = a1 * b2 - a2 * b1
        if abs(det) < EPSILON:
            return False, np.zeros(2)

        x = (c1 * b2 - c2 * b1) / det
        y = (c1 * a2 - c2 * a1) / det

        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return False, np.zeros(2)

        return True, np.array([x, y])


class DecompBase(ABC):
    """Base class for decomposition algorithms"""

    def __init__(self):
        self.obs_ = []
        self.ellipsoid_ = Ellipsoid()
        self.polyhedron_ = Polyhedron()
        self.local_bbox_ = np.zeros(2)  # Will be resized based on dimension

    def set_local_bbox(self, bbox: np.ndarray):
        """Set local bounding box"""
        self.local_bbox_ = bbox.copy()

    def set_obs(self, obs: List[np.ndarray]):
        """Set obstacle points (filtered by local bbox)"""
        # Create polyhedron from local bbox and filter points
        poly = Polyhedron()
        self.add_local_bbox(poly)
        self.obs_ = poly.points_inside(obs)

    def get_obs(self) -> List[np.ndarray]:
        return self.obs_

    def get_ellipsoid(self) -> Ellipsoid:
        return self.ellipsoid_

    def get_polyhedron(self) -> Polyhedron:
        return self.polyhedron_

    @abstractmethod
    def dilate(self, radius: float = 0):
        """Inflate the shape"""
        pass

    @abstractmethod
    def add_local_bbox(self, polyhedron: Polyhedron):
        """Add local bounding box constraints"""
        pass

    def find_polyhedron(self):
        """Find polyhedron from ellipsoid and obstacles"""
        poly = Polyhedron()
        obs_remain = self.obs_.copy()

        while obs_remain:
            hyperplane = self.ellipsoid_.closest_hyperplane(obs_remain)
            poly.add(hyperplane)

            # Filter points behind the hyperplane
            obs_tmp = []
            for pt in obs_remain:
                if hyperplane.signed_dist(pt) < 0:
                    obs_tmp.append(pt)
            obs_remain = obs_tmp

        self.polyhedron_ = poly


class LineSegment(DecompBase):
    """Line segment decomposition"""

    def __init__(self, p1: Optional[np.ndarray] = None, p2: Optional[np.ndarray] = None):
        super().__init__()
        self.p1_ = p1 if p1 is not None else np.zeros(2)
        self.p2_ = p2 if p2 is not None else np.zeros(2)
        self.local_bbox_ = np.zeros(len(self.p1_))

    def dilate(self, radius: float = 0):
        """Dilate the line segment"""

        if np.linalg.norm(self.p1_ - self.p2_) < 1e-4:
            # Degenerate segment: use dummy ellipsoid and polyhedron
            dummy_C = np.zeros((2, 2))
            dummy_d = self.p1_.copy()
            self.ellipsoid_ = Ellipsoid(dummy_C, dummy_d)

            dummy_poly = Polyhedron()
            dummy_poly.add(Hyperplane(self.p1_, np.array([1.0, 0.0])))  # a1 = 1
            dummy_poly.add(Hyperplane(self.p1_, np.array([-1.0, 0.0])))  # a1 = -1
            dummy_poly.add(Hyperplane(self.p1_, np.array([0.0, 1.0])))  # a2 = 1
            dummy_poly.add(Hyperplane(self.p1_, np.array([0.0, -1.0])))  # a2 = -1
            self.polyhedron_ = dummy_poly
            return

        self._find_ellipsoid(radius)
        self.find_polyhedron()
        self.add_local_bbox(self.polyhedron_)

    def get_line_segment(self) -> List[np.ndarray]:
        """Get the line segment endpoints"""
        return [self.p1_, self.p2_]

    def add_local_bbox(self, polyhedron: Polyhedron):
        """Add local bounding box constraints"""
        if np.linalg.norm(self.local_bbox_) == 0:
            return

        # Direction along the line
        direction = self.p2_ - self.p1_
        if np.linalg.norm(direction) == 0:
            return
        direction = direction / np.linalg.norm(direction)

        # Perpendicular direction
        if len(self.p1_) == 2:
            dir_h = np.array([-direction[1], direction[0]])
        else:  # 3D case
            # Find perpendicular vector
            if abs(direction[2]) < 0.9:
                dir_h = np.cross(direction, np.array([0, 0, 1]))
            else:
                dir_h = np.cross(direction, np.array([1, 0, 0]))
            dir_h = dir_h / np.linalg.norm(dir_h)

        # Add constraints along width (perpendicular to line)
        if len(self.local_bbox_) > 1:
            pp1 = self.p1_ + dir_h * self.local_bbox_[1]
            pp2 = self.p1_ - dir_h * self.local_bbox_[1]
            polyhedron.add(Hyperplane(pp1, dir_h))
            polyhedron.add(Hyperplane(pp2, -dir_h))

        # Add constraints along length
        if len(self.local_bbox_) > 0:
            pp3 = self.p2_ + direction * self.local_bbox_[0]
            pp4 = self.p1_ - direction * self.local_bbox_[0]
            polyhedron.add(Hyperplane(pp3, direction))
            polyhedron.add(Hyperplane(pp4, -direction))

        # Add constraints along height (3D only)
        if len(self.p1_) == 3 and len(self.local_bbox_) > 2:
            dir_v = np.cross(direction, dir_h)
            pp5 = self.p1_ + dir_v * self.local_bbox_[2]
            pp6 = self.p1_ - dir_v * self.local_bbox_[2]
            polyhedron.add(Hyperplane(pp5, dir_v))
            polyhedron.add(Hyperplane(pp6, -dir_v))

    def _find_ellipsoid(self, offset_x: float):
        """Find ellipsoid for line segment"""
        f = np.linalg.norm(self.p1_ - self.p2_) / 2
        dim = len(self.p1_)

        # Initial ellipsoid
        C = f * np.eye(dim)
        axes = np.full(dim, f)
        C[0, 0] += offset_x
        axes[0] += offset_x

        # Scale if needed
        if axes[0] > 0:
            ratio = axes[1] / axes[0]
            axes *= ratio
            C *= ratio

        # Rotation matrix
        if dim == 2:
            R = GeometricUtils.vec2_to_rotation(self.p2_ - self.p1_)
        else:
            R = GeometricUtils.vec3_to_rotation(self.p2_ - self.p1_)

        C = R @ C @ R.T
        center = (self.p1_ + self.p2_) / 2

        ellipsoid = Ellipsoid(C, center)
        obs_inside = ellipsoid.points_inside(self.obs_)

        # Iteratively adjust ellipsoid to avoid obstacles
        while obs_inside:
            pw = ellipsoid.closest_point(obs_inside)
            p = R.T @ (pw - ellipsoid.d())  # Transform to ellipsoid frame

            # Adjust short axes
            if p[0] < axes[0]:
                axes[1] = abs(p[1]) / math.sqrt(1 - (p[0] / axes[0]) ** 2)

            # Update ellipsoid
            new_C = np.diag(axes)
            ellipsoid.C_ = R @ new_C @ R.T

            # Filter remaining obstacles
            obs_new = []
            for pt in obs_inside:
                if 1 - ellipsoid.dist(pt) > EPSILON:
                    obs_new.append(pt)
            obs_inside = obs_new

        self.ellipsoid_ = ellipsoid


class SeedDecomp(DecompBase):
    """Seed point decomposition"""

    def __init__(self, p: Optional[np.ndarray] = None):
        super().__init__()
        self.p_ = p if p is not None else np.zeros(2)
        self.local_bbox_ = np.zeros(len(self.p_))

    def dilate(self, radius: float = 0):
        """Dilate around seed point with sphere"""
        dim = len(self.p_)
        self.ellipsoid_ = Ellipsoid(radius * np.eye(dim), self.p_)
        self.find_polyhedron()
        self.add_local_bbox(self.polyhedron_)

    def get_seed(self) -> np.ndarray:
        return self.p_

    def add_local_bbox(self, polyhedron: Polyhedron):
        """Add bounding box constraints around seed"""
        if np.linalg.norm(self.local_bbox_) == 0:
            return

        dim = len(self.p_)

        # X direction
        if len(self.local_bbox_) > 0:
            dir_x = np.zeros(dim)
            dir_x[0] = 1
            pp1 = self.p_ + dir_x * self.local_bbox_[0]
            pp2 = self.p_ - dir_x * self.local_bbox_[0]
            polyhedron.add(Hyperplane(pp1, dir_x))
            polyhedron.add(Hyperplane(pp2, -dir_x))

        # Y direction
        if len(self.local_bbox_) > 1:
            dir_y = np.zeros(dim)
            dir_y[1] = 1
            pp3 = self.p_ + dir_y * self.local_bbox_[1]
            pp4 = self.p_ - dir_y * self.local_bbox_[1]
            polyhedron.add(Hyperplane(pp3, dir_y))
            polyhedron.add(Hyperplane(pp4, -dir_y))

        # Z direction (3D only)
        if dim == 3 and len(self.local_bbox_) > 2:
            dir_z = np.zeros(dim)
            dir_z[2] = 1
            pp5 = self.p_ + dir_z * self.local_bbox_[2]
            pp6 = self.p_ - dir_z * self.local_bbox_[2]
            polyhedron.add(Hyperplane(pp5, dir_z))
            polyhedron.add(Hyperplane(pp6, -dir_z))


class EllipsoidDecomp:
    """Ellipsoid decomposition for path planning"""

    def __init__(self, origin: Optional[np.ndarray] = None, dim: Optional[np.ndarray] = None):
        if origin is not None and dim is not None:
            self.global_bbox_min_ = origin.copy()
            self.global_bbox_max_ = origin + dim
        else:
            self.global_bbox_min_ = np.zeros(2)
            self.global_bbox_max_ = np.zeros(2)

        self.obs_ = []
        self.local_bbox_ = np.zeros(2)
        self.path_ = []
        self.ellipsoids_ = []
        self.polyhedrons_ = []
        self.lines_ = []

    def set_obs(self, obs: List[np.ndarray]):
        """Set obstacle points"""
        self.obs_ = obs

    def set_local_bbox(self, bbox: np.ndarray):
        """Set local bounding box"""
        self.local_bbox_ = bbox.copy()

    def get_path(self) -> List[np.ndarray]:
        return self.path_

    def get_polyhedrons(self) -> List[Polyhedron]:
        return self.polyhedrons_

    def get_ellipsoids(self) -> List[Ellipsoid]:
        return self.ellipsoids_

    def get_constraints(self) -> List[LinearConstraint]:
        """Get linear constraints for Safe Flight Corridor"""
        constraints = []
        for i in range(len(self.polyhedrons_)):
            pt = (self.path_[i] + self.path_[i + 1]) / 2
            constraint = LinearConstraint(p0=pt, hyperplanes=self.polyhedrons_[i].hyperplanes())
            if np.linalg.norm(self.path_[i] - self.path_[i + 1]) < 1e-4:
                size = len(self.polyhedrons_[i].hyperplanes())
                dim = len(pt)
                A = np.zeros((size, dim))
                b = np.zeros(size)
                for i, hp in enumerate(self.polyhedrons_[i].hyperplanes()):
                    n = hp.n.copy()
                    c = np.dot(hp.p, n)
                    if np.dot(n, pt) - c > 0:
                        n = -n
                    A[i] = n
                    b[i] = 100.0
                LOG_DEBUG("Norm is small so using dummy constraint")
                constraint = LinearConstraint(A, b)
            constraints.append(constraint)
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

    def dilate(self, path: List[np.ndarray], offset_x: float = 0):
        """Perform decomposition along path"""
        if len(path) < 2:
            return

        N = len(path) - 1
        self.lines_ = []
        self.ellipsoids_ = []
        self.polyhedrons_ = []

        # Create line segments and decompose
        for i in range(N):

            line = LineSegment(path[i], path[i + 1])
            line.set_local_bbox(self.local_bbox_)
            line.set_obs(self.obs_)
            line.dilate(offset_x)

            self.lines_.append(line)
            self.ellipsoids_.append(line.get_ellipsoid())
            self.polyhedrons_.append(line.get_polyhedron())

        self.path_ = path

        # Add global bounding box if specified
        if (np.linalg.norm(self.global_bbox_min_) != 0 or
                np.linalg.norm(self.global_bbox_max_) != 0):
            for polyhedron in self.polyhedrons_:
                self._add_global_bbox(polyhedron)

    def _add_global_bbox(self, polyhedron: Polyhedron):
        """Add global bounding box constraints"""
        dim = len(self.global_bbox_min_)

        if dim == 2:
            # X bounds
            polyhedron.add(Hyperplane(np.array([self.global_bbox_max_[0], 0]), np.array([1, 0])))
            polyhedron.add(Hyperplane(np.array([self.global_bbox_min_[0], 0]), np.array([-1, 0])))
            # Y bounds
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_max_[1]]), np.array([0, 1])))
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_min_[1]]), np.array([0, -1])))

        elif dim == 3:
            # X bounds
            polyhedron.add(Hyperplane(np.array([self.global_bbox_max_[0], 0, 0]), np.array([1, 0, 0])))
            polyhedron.add(Hyperplane(np.array([self.global_bbox_min_[0], 0, 0]), np.array([-1, 0, 0])))
            # Y bounds
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_max_[1], 0]), np.array([0, 1, 0])))
            polyhedron.add(Hyperplane(np.array([0, self.global_bbox_min_[1], 0]), np.array([0, -1, 0])))
            # Z bounds
            polyhedron.add(Hyperplane(np.array([0, 0, self.global_bbox_max_[2]]), np.array([0, 0, 1])))
            polyhedron.add(Hyperplane(np.array([0, 0, self.global_bbox_min_[2]]), np.array([0, 0, -1])))


class IterativeDecomp(EllipsoidDecomp):
    """Iterative decomposition for safer corridors"""

    def dilate_iter(self, path_raw: List[np.ndarray], iter_num: int = 5,
                    res: float = 0, offset_x: float = 0):
        """Iterative decomposition with path simplification"""
        # Downsample if resolution specified
        path = self._downsample(path_raw, res) if res > 0 else path_raw

        # Initial decomposition
        self.dilate(path, offset_x)
        new_path = self._simplify(path)

        # Iterate
        for i in range(iter_num):
            if len(new_path) == len(path):
                break
            path = new_path
            self.dilate(path, offset_x)
            new_path = self._simplify(path)

    def _downsample(self, path: List[np.ndarray], d: float) -> List[np.ndarray]:
        """Uniformly sample path into segments of length d"""
        if len(path) < 2:
            return path

        new_path = []
        for i in range(1, len(path)):
            dist = np.linalg.norm(path[i] - path[i - 1])
            cnt = int(np.ceil(dist / d))
            for j in range(cnt):
                new_path.append(path[i - 1] + j * (path[i] - path[i - 1]) / cnt)

        new_path.append(path[-1])
        return new_path

    def _cal_closest_dist(self, pt: np.ndarray, polyhedron: Polyhedron) -> float:
        """Calculate closest distance to polyhedron"""
        min_dist = float('inf')
        for hp in polyhedron.hyperplanes():
            d = abs(np.dot(hp.n, pt - hp.p))
            if d < min_dist:
                min_dist = d
        return min_dist

    def _simplify(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Remove redundant waypoints"""
        if len(path) <= 2:
            return path

        ref_pt = path[0]
        new_path = [ref_pt]

        for i in range(2, len(path)):
            if (self.polyhedrons_[i - 1].inside(ref_pt) and
                    self._cal_closest_dist(ref_pt, self.polyhedrons_[i - 1]) > 0.1):
                continue
            else:
                ref_pt = path[i - 1]
                new_path.append(ref_pt)

        new_path.append(path[-1])
        return new_path

###### Spline Utils ######


class SplineSegment:

    def __init__(self, param, name, spline_nr):
        # Retrieve spline values from the parameters (stored as multi parameter by name)
        self.a = param.get(f"{name}_{spline_nr}_a")
        self.b = param.get(f"{name}_{spline_nr}_b")
        self.c = param.get(f"{name}_{spline_nr}_c")
        self.d = param.get(f"{name}_{spline_nr}_d")

        self.s_start = param.get(f"path_{spline_nr}_start")

    def at(self, spline_index):
        s = spline_index - self.s_start
        return self.a * s * s * s + self.b * s * s + self.c * s + self.d

    def deriv(self, spline_index):
        s = spline_index - self.s_start
        return 3 * self.a * s * s + 2 * self.b * s + self.c

    def deriv2(self, spline_index):
        s = spline_index - self.s_start
        return 6 * self.a * s + 2 * self.b


class Spline:
    def __init__(self, params, name, num_segments, s):
        self.splines = []  # Classes containing the splines
        self.lambdas = []  # Merges splines
        for i in range(num_segments):
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


class Spline2D:

    def __init__(self, params, num_segments, s):
        self.spline_x = Spline(params, "path_x", num_segments, s)
        self.spline_y = Spline(params, "path_y", num_segments, s)

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


class DouglasRachford:
    """
    Python implementation of the Douglas-Rachford projection method
    for collision avoidance in motion planning.
    """

    def douglas_rachford_projection(self, position, delta, anchor, r, starting_pose):
        """
        Performs Douglas-Rachford projection.

        Args:
            position (np.ndarray): Point to project (modified in-place)
            delta (np.ndarray): Obstacle position
            anchor (np.ndarray): Anchor position (reference obstacle)
            r (float): Collision radius
            starting_pose (np.ndarray): Starting pose for projection
        """
        position[:] = (position + self._reflect(self._reflect(position, anchor, r, position), delta, r, starting_pose)) / 2.0

    def _project(self, position, delta, radius, starting_pose):
        """
        Projects point `position` onto a circle of radius `radius` centered at `delta`,
        in the direction from `starting_pose` to `delta`, if it's within the radius.
        """
        diff = position - delta
        dist = np.linalg.norm(diff)

        if dist < radius:
            direction = (delta - starting_pose)
            norm_dir = np.linalg.norm(direction)
            if norm_dir == 0:
                return position.copy()  # Avoid division by zero
            return delta - direction / norm_dir * radius
        else:
            return position.copy()

    def _reflect(self, position, delta, r, starting_pose):
        """
        Reflects point p across the projection onto the collision-free region.

        Args:
            position (np.ndarray): Point to reflect
            delta (np.ndarray): Obstacle center
            r (float): Collision radius
            starting_pose (np.ndarray): Starting pose for projection

        Returns:
            np.ndarray: Reflected point
        """
        return 2.0 * self._project(position, delta, r, starting_pose) - position