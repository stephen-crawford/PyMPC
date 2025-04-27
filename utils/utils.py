import datetime
import inspect
import logging
import os
import random
import threading
import time
from contextlib import contextmanager

import math
import numpy as np
import yaml

# Initialize logger
logger = logging.getLogger(__name__)

def read_config_file():
    print("Reading config file")
    config_path = os.path.join(os.path.dirname(__file__), "../../PyMPC/utils/CONFIG.yml")
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.ymlError as e:
            print(f"Error reading YAML file: {e}")
            return None


def get_config_dotted(config, dotted_key, default=None):
    keys = dotted_key.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if default is not None:
                return default
            raise KeyError(f"Key path '{dotted_key}' not found in config.")
    return value


CONFIG = read_config_file()
SAVE_FOLDER = CONFIG["recording"]["folder"]
SAVE_FILE = CONFIG["recording"]["file"]

MOCKED_CONFIG = {
    "max_obstacle_distance": 50.0,
    "max_obstacles": 3,
    "N": 5,
    "integrator_step": 0.1,
    "probabilistic": {"enable": False}
}

def set_complete_mocked_config(data):
    global MOCKED_CONFIG
    MOCKED_CONFIG = data

def set_mocked_config_entry(entry, value):
    global MOCKED_CONFIG
    MOCKED_CONFIG[entry] = value


# Original Python utilities - keeping these intact
@contextmanager
def PROFILE_SCOPE(name):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.debug(f"{name} took {elapsed_time:.6f} seconds")

def LOG_DEBUG(message):
    logging.basicConfig(level=logging.DEBUG)  # Configure logging level
    logger = logging.getLogger(__name__)
    logger.debug(message)


def LOG_WARN(message):
    logging.basicConfig(level=logging.WARN)  # Configure logging level
    logger = logging.getLogger(__name__)
    logger.debug(message)


def PYMPC_ASSERT(expr, msg):
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)

    if not expr:
        frame = inspect.currentframe().f_back  # Get the caller's frame
        file = frame.f_code.co_filename
        line = frame.f_lineno
        expr_str = frame.f_globals.get("__name__", "Unknown")  # Expression string not available directly

        logger.error(f"Assert failed:\t{msg}\n"
                     f"Expected:\t{expr_str}\n"
                     f"Source:\t\t{file}, line {line}\n")
        raise AssertionError(msg)


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


# Profiling tools
class Benchmarker:
    def __init__(self, name):
        self.name_ = name
        self.running_ = False
        self.total_duration_ = 0.0
        self.max_duration_ = -1.0
        self.min_duration_ = 99999.0
        self.last_ = -1.0
        self.total_runs_ = 0

    def print(self):
        """Print results."""
        average_run_time = self.total_duration_ / float(self.total_runs_) * 1000.0

        print("-" * 50)
        print(f"Timing of: {self.name_}")
        print(f"Average (ms): {average_run_time}")
        print(f"Max (ms): {self.max_duration_ * 1000.0}")

    def start(self):
        """Start the timer."""
        self.running_ = True
        self.start_time_ = datetime.datetime.now()

    def cancel(self):
        """Cancel the timer."""
        self.running_ = False

    def stop(self):
        """Stop the timer and record results."""
        if not self.running_:
            return 0.0

        end_time = datetime.datetime.now()
        time_diff = (end_time - self.start_time_).total_seconds()

        if time_diff < self.min_duration_:
            self.min_duration_ = time_diff

        if time_diff > self.max_duration_:
            self.max_duration_ = time_diff

        self.total_duration_ += time_diff
        self.total_runs_ += 1

        self.last_ = time_diff
        self.running_ = False
        return self.last_

    def get_last(self):
        """Get the last recorded time."""
        return self.last_

    def get_total_duration(self):
        """Get the total duration."""
        return self.total_duration_

    def reset(self):
        """Reset the benchmarker."""
        self.total_duration_ = 0.0
        self.max_duration_ = -1.0
        self.min_duration_ = 99999.0
        self.last_ = -1.0
        self.total_runs_ = 0
        self.running_ = False

    def is_running(self):
        """Check if the timer is running."""
        return self.running_


class Timer:
    def __init__(self, duration):
        self.duration_ = duration
        self.start_time = None

    def set_duration(self, duration):
        """Set the timer duration."""
        self.duration_ = duration

    def start(self):
        """Start the timer."""
        self.start_time = datetime.datetime.now()

    def current_duration(self):
        """Get the current duration."""
        end_time = datetime.datetime.now()
        return (end_time - self.start_time).total_seconds()

    def has_finished(self):
        """Check if the timer has finished."""
        duration = self.current_duration()
        return duration >= self.duration_


# Chrome trace event format profiler
class ProfileResult:
    def __init__(self, name, start, end, thread_id):
        self.Name = name
        self.Start = start
        self.End = end
        self.ThreadID = thread_id


class InstrumentationSession:
    def __init__(self, name):
        self.Name = name


class Instrumentor:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.m_CurrentSession = None
        self.m_OutputStream = None
        self.m_ProfileCount = 0
        self.m_lock = threading.Lock()

    def begin_session(self, name, filepath):
        """Begin a profiling session."""
        import os
        # Get package path would need to be implemented based on your ROS setup
        full_filepath = os.path.join(self._get_package_path(name), filepath)
        print(f"Profiling Path: {full_filepath}")
        self.m_OutputStream = open(full_filepath, 'w')
        self._write_header()
        self.m_CurrentSession = InstrumentationSession(name)

    def end_session(self):
        """End the profiling session."""
        self._write_footer()
        self.m_OutputStream.close()
        self.m_CurrentSession = None
        self.m_ProfileCount = 0

    def write_profile(self, result):
        """Write a profile result."""
        with self.m_lock:
            if self.m_ProfileCount > 0:
                self.m_OutputStream.write(",")

            name = result.Name.replace('"', "'")

            self.m_OutputStream.write("{")
            self.m_OutputStream.write(f'"cat":"function",')
            self.m_OutputStream.write(f'"dur":{result.End - result.Start},')
            self.m_OutputStream.write(f'"name":"{name}",')
            self.m_OutputStream.write(f'"ph":"X",')
            self.m_OutputStream.write(f'"pid":0,')
            self.m_OutputStream.write(f'"tid":{result.ThreadID},')
            self.m_OutputStream.write(f'"ts":{result.Start}')
            self.m_OutputStream.write("}")

            self.m_OutputStream.flush()
            self.m_ProfileCount += 1

    def _write_header(self):
        """Write the header of the profiling file."""
        self.m_OutputStream.write('{"otherData": {},"traceEvents":[')
        self.m_OutputStream.flush()

    def _write_footer(self):
        """Write the footer of the profiling file."""
        self.m_OutputStream.write("]}")
        self.m_OutputStream.flush()

    def _get_package_path(self, name):
        """Get ROS package path - simplified implementation."""
        import os
        # This would need to be properly implemented based on your ROS setup
        return os.getcwd()


class InstrumentationTimer:
    def __init__(self, name):
        self.m_Name = name
        self.m_Stopped = False
        self.m_StartTimepoint = datetime.datetime.now()

    def __del__(self):
        if not self.m_Stopped:
            self.stop()

    def stop(self):
        """Stop the timer and write a profile result."""
        end_timepoint = datetime.datetime.now()

        start = int(self.m_StartTimepoint.timestamp() * 1000000)
        end = int(end_timepoint.timestamp() * 1000000)

        thread_id = hash(threading.current_thread().ident) & 0xFFFFFFFF
        Instrumentor.get().write_profile(ProfileResult(self.m_Name, start, end, thread_id))

        self.m_Stopped = True


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


class DataSaver:
    def __init__(self):
        self.data = {}
        self.add_timestamp = False

    def set_add_timestamp(self, value):
        self.add_timestamp = value

    def add_data(self, key, value):
        self.data[key] = value

    def save_data(self, folder, file):
        with open(f"{folder}/{file}", "w") as f:
            f.write(str(self.data))  # Example: Save as a string dictionary

    def get_file_path(self, folder, file, flag):
        return f"{folder}/{file}"


class ExperimentManager:
    def __init__(self):
        self.SAVE_FOLDER = CONFIG["recording"]["folder"]
        self.SAVE_FILE = CONFIG["recording"]["file"]

        self.data_saver = DataSaver()
        self.data_saver.set_add_timestamp(CONFIG["recording"]["timestamp"])

        if CONFIG["recording"]["enable"]:
            logger.info(
                f"Planner Save File: {self.data_saver.get_file_path(self.SAVE_FOLDER, self.SAVE_FILE, False)}")

        self.control_iteration = 0
        self.iteration_at_last_reset = 0
        self.experiment_counter = 0

    def update(self, state, solver, data):
        logger.info("planner.util.save_data()")

        if len(data.dynamic_obstacles) == 0:
            logger.info("Not exporting data: Obstacles not yet received.")
            return

        # Save vehicle data
        self.data_saver.add_data("vehicle_pose", state.get_pose())
        self.data_saver.add_data("vehicle_orientation", state.get("psi"))

        # Save planned trajectory
        for k in range(CONFIG["N"]):
            self.data_saver.add_data(f"vehicle_plan_{k}", solver.get_ego_prediction_position(k))

        # Save obstacle data
        for v, obstacle in enumerate(data.dynamic_obstacles):
            if obstacle.index is not None:
                self.data_saver.add_data(f"obstacle_map_{v}", obstacle.index)
                self.data_saver.add_data(f"obstacle_{v}_pose", obstacle.position)
                self.data_saver.add_data(f"obstacle_{v}_orientation", obstacle.angle)

            # Save disc obstacle (assume only one disc)
            self.data_saver.add_data("disc_0_pose", obstacle.position)
            self.data_saver.add_data("disc_0_radius", obstacle.radius)
            self.data_saver.add_data("disc_0_obstacle", v)

        self.data_saver.add_data("max_intrusion", data.intrusion)
        self.data_saver.add_data("metric_collisions", int(data.intrusion > 0.0))

        # Time keeping
        self.data_saver.add_data("iteration", self.control_iteration)
        self.control_iteration += 1

    def export_data(self):
        # Use the class variables instead of requiring parameters
        self.data_saver.save_data(self.SAVE_FOLDER, self.SAVE_FILE)

    def on_task_complete(self, objective_reached):

        self.data_saver.add_data("reset", self.control_iteration)
        self.data_saver.add_data(
            "metric_duration",
            (self.control_iteration - self.iteration_at_last_reset) * (1.0 / float(CONFIG["control_frequency"]))
        )
        self.data_saver.add_data("metric_completed", int(objective_reached))

        self.iteration_at_last_reset = self.control_iteration  # Fixed: was using _control_iteration
        self.experiment_counter += 1

        num_experiments = int(CONFIG["recording"]["num_experiments"])
        if self.experiment_counter % num_experiments == 0 and self.experiment_counter > 0:
            self.export_data()

        if self.experiment_counter >= num_experiments:
            logger.info(f"Completed {num_experiments} experiments.")
        else:
            logger.info(f"Starting experiment {self.experiment_counter + 1} / {num_experiments}")

        assert self.experiment_counter < num_experiments, "Stopping the planner."

    def set_start_experiment(self):
        self.iteration_at_last_reset = self.control_iteration


import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


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