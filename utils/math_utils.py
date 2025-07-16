import math
import random
from typing import List

import casadi as cd
import numpy as np


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