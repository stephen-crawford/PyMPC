import math
import time
import yaml
import logging
import random
import numpy as np
import threading
import datetime
import functools
import inspect
from contextlib import contextmanager

import logging
from utils.utils import read_config_file

# Initialize logger
logger = logging.getLogger(__name__)

# Read configuration
CONFIG = read_config_file()

SAVE_FOLDER = CONFIG["recording"]["folder"]
SAVE_FILE = CONFIG["recording"]["file"]


# Initialize logger
logger = logging.getLogger(__name__)

# Read configuration
CONFIG = read_config_file()




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


def read_config_file():
    with open("CONFIG.yml", 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None


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
            self.export_data()  # Fixed: was passing arguments incorrectly

        if self.experiment_counter >= num_experiments:
            logger.info(f"Completed {num_experiments} experiments.")
        else:
            logger.info(f"Starting experiment {self.experiment_counter + 1} / {num_experiments}")

        assert self.experiment_counter < num_experiments, "Stopping the planner."

    def set_start_experiment(self):
        self.iteration_at_last_reset = self.control_iteration
