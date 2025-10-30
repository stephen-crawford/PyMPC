import datetime
import inspect
import json
import logging
import sys
import time
from contextlib import contextmanager

import numpy as np

# Initialize logger

import logging

# Configure logging once at module level

import os
import yaml
from typing import List, Tuple, Dict, Union, Any


######## CONFIGURATION MANAGEMENT

def load_yaml(path, target_dict):
    """Load YAML configuration file into the target dictionary."""
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            target_dict.update(data)
    except Exception as e:
        LOG_DEBUG(f"Error loading config from {path}: {e}")


def read_config_file():
    print("Reading config file")
    config_path = os.path.join(os.path.dirname(__file__), "../../PyMPC/config/CONFIG.yml")
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

def write_to_config(key, value):
    config_path = os.path.join(os.path.dirname(__file__), "../../PyMPC/config/CONFIG.yml")
    config_path = os.path.abspath(config_path)

    # Load current config
    try:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file) or {}
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return False

    # Update config
    config_data[key] = value

    # Write updated config
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config_data, file, default_flow_style=False)
        print(f"Updated {key} in config file.")
        return True
    except Exception as e:
        print(f"Error writing to config file: {e}")
        return False

def get_config_dotted(config, dotted_key, default=None):
    print("Trying to parse dotted key from config")
    keys = dotted_key.split('.')
    value = config
    for key in keys:
        print("Looking for " + str(key))
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

##### LOGGING MANAGEMENT
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

# Optional named logger for your module
logger = logging.getLogger("PyMPC")

# Create a handler that outputs to the console (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the handler to DEBUG level

# Create a simple format for the log messages
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

file_handler = logging.FileHandler('debug.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)

# Export short-hand debug logging
def LOG_DEBUG(msg):
    logger.debug(msg)


def LOG_INFO(msg):
    logger.info(msg)

def LOG_WARN(msg):
    logger.warning(msg)

def LOG_ERROR(msg):
    logger.error(msg)

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


class bcolors:
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    HEADER = BOLD

def print_value(name, value, tab=False, **kwargs):
    if tab:
        string = " "
    else:
        string = ""

    print(string + bcolors.BOLD + bcolors.UNDERLINE + f"{name}" + bcolors.ENDC + f": {value}", **kwargs)


def print_path(name, value, tab=False, **kwargs):
    print_value(name, os.path.abspath(value), tab, **kwargs)


def print_success(msg):
    print(bcolors.BOLD + bcolors.OKGREEN + f"{msg}" + bcolors.ENDC)


def print_warning(msg, no_tab=False):
    if no_tab:
        print(bcolors.BOLD + bcolors.WARNING + f"Warning: {msg}" + bcolors.ENDC)
    else:
        print("\t" + bcolors.BOLD + bcolors.WARNING + f"Warning: {msg}" + bcolors.ENDC)


def print_header(msg):
    print("==============================================")
    print("\t" + bcolors.HEADER + f"{msg}" + bcolors.ENDC)
    print("==============================================")



#### TIME TRACKING
class TimeTracker:

    def __init__(self, name):
        self.name = name
        self._times = []

    def add(self, timing):
        self._times.append(timing)

    def get_stats(self):
        return np.mean(self._times), np.max(self._times), len(self._times)

    def print_stats(self):
        print(f"--- Computation Times {self.name} ---")
        print_value("Mean", f"{np.mean(self._times):.1f} ms", tab=True)
        print_value("Max", f"{np.max(self._times):.1f} ms", tab=True)
        print_value("Number of calls", len(self._times), tab=True)

def get_base_path():
  return os.path.dirname(os.path.realpath('../../utils'))


def get_current_package():
  return os.path.dirname(os.path.realpath(sys.argv[0])).split("/")[-2]


def get_package_path(package_name):
  return os.path.join(os.path.dirname(__file__), f"../../{package_name}")


def get_solver_package_path():
  return get_package_path("mpc_planner_solver")


def save_config_path():
  config_path = os.path.join(get_base_path(), "utils/")
  os.makedirs(os.path.dirname(config_path), exist_ok=True)
  return config_path


def load_config_path():
  print("Joining base path of " + str(get_base_path()) + " and /utils")
  util_path = str(get_base_path())+"/utils"
  print("Returning path of " + str(util_path))
  return util_path


def load_settings_path(setting_file_name="CONFIG"):
  return str(load_config_path()) + f"/{setting_file_name}.yml"


def load_settings(setting_file_name="CONFIG"):
  path = load_settings_path(setting_file_name)
  print_path("Settings", path, end="")
  with open(path, "r") as stream:
    settings = yaml.safe_load(stream)
  print_success(f" -> loaded")
  return settings


def load_test_settings(setting_file_name="settings"):
  path = f"{get_package_path('mpc_planner')}/config/{setting_file_name}.yml"
  print_path("Settings", path, end="")
  with open(path, "r") as stream:
    settings = yaml.safe_load(stream)
  print_success(f" -> loaded")
  return settings


def default_solver_path(settings):
  return os.path.join(os.getcwd(), f"{solver_name(settings)}")


def solver_path(settings):
  return os.path.join(get_solver_package_path(), f"{solver_name(settings)}")


def default_casadi_solver_path(settings):
  return os.path.join(get_package_path("solver_generator"), f"casadi")


def casadi_solver_path(settings):
  return os.path.join(get_solver_package_path(), f"casadi")


def parameter_map_path():
  return os.path.join(save_config_path(), f"parameter_map.yml")


def model_map_path():
  return os.path.join(save_config_path(), f"model_map.yml")


def solver_settings_path():
  return os.path.join(save_config_path(), f"solver_settings.yml")

def generated_src_file(settings):
  return os.path.join(solver_path(settings), f"mpc_planner_generated.py")

def planner_path():
  return get_package_path("mpc_planner")

def solver_name(settings):
  return "Solver"

def write_to_yaml(filename, data):
  with open(filename, "w") as outfile:
    yaml.dump(data, outfile, default_flow_style=False)


def define_robot_area(length: float, width: float, n_discs: int) -> list[Disc]:
    """Define the robot area using discs."""
    # Where is the center w.r.t. the back of the vehicle
    center_offset = length / 2.  # Could become a parameter
    radius = width / 2.

    robot_area = []
    assert n_discs > 0, "Trying to create a collision region with less than a disc"

    if n_discs == 1:
        robot_area.append(Disc(0., radius))
    else:
        for i in range(n_discs):
            if i == 0:
                # First disc at the back of the car
                robot_area.append(Disc(-center_offset + radius, radius))
            elif i == n_discs - 1:
                # Last disc at the front of the car
                robot_area.append(Disc(-center_offset + length - radius, radius))
            else:
                # Other discs in between
                offset = -center_offset + radius + (i * (length - 2. * radius) / (n_discs - 1.))
                robot_area.append(Disc(offset, radius))

            logging.debug(f"Disc {i}: offset {robot_area[-1].offset}, radius {robot_area[-1].radius}")

    return robot_area


def propagate_obstacles(data, dt=0.1, horizon=10, speed=0, sigma_pos=0.2):
  if data.dynamic_obstacles is None:
      return

  for obstacle in data.dynamic_obstacles:
      pred = obstacle.prediction
      path = getattr(pred, "path", None)

      # Fallback: constant velocity if no path
      if path is None:
          velocity = np.array([np.cos(obstacle.angle), np.sin(obstacle.angle)]) * speed
          obstacle.prediction = get_constant_velocity_prediction(obstacle.position, velocity, dt, horizon)
          continue

      total_length = path.s[-1]

      # Initialize progress in arc length
      if not hasattr(obstacle, "s"):
          obstacle.s = 0.0
      obstacle.s += speed * dt

      # If reached end of current path → generate new path
      if obstacle.s >= total_length:
          start = [path.x[-1], path.y[-1], path.z[-1] if hasattr(path, "z") else 0.0]

          # Choose new goal
          if hasattr(obstacle, "road_to_follow"):
              road = obstacle.road_to_follow
              s_offset = np.random.uniform(0, road.length)
              x_center = road.x_spline(s_offset)
              y_center = road.y_spline(s_offset)

              # Tangent & lateral offset to simulate lane variation
              dx = road.x_spline(min(s_offset + 0.1, road.length)) - x_center
              dy = road.y_spline(min(s_offset + 0.1, road.length)) - y_center
              tangent = np.array([dx, dy]) / np.linalg.norm([dx, dy])
              normal = np.array([-tangent[1], tangent[0]])
              lateral_offset = np.random.uniform(-3.0, 3.0)
              goal = [x_center + lateral_offset * normal[0], y_center + lateral_offset * normal[1], 0.0]
          else:
              goal = [
                  np.random.uniform(start[0] - 20, start[0] + 20),
                  np.random.uniform(start[1] - 20, start[1] + 20),
                  0.0
              ]

          # Generate new reference path
          new_path = generate_reference_path(
              start, goal, path_type=np.random.choice(["straight", "curved", "s-turn", "circle"]),
              num_points=10
          )

          obstacle.prediction.path = new_path
          obstacle.s = 0.0
          path = new_path
          total_length = path.s[-1]

      # ✅ Compute position using arc-length splines
      s_now = min(obstacle.s, total_length)
      x = path.x_spline(s_now)
      y = path.y_spline(s_now)
      z = path.z_spline(s_now) if path.z_spline else 0.0
      obstacle.position = np.array([x, y, z])

      # ✅ Compute heading
      ds = 0.1
      s_next = min(s_now + ds, total_length)
      dx = path.x_spline(s_next) - x
      dy = path.y_spline(s_next) - y
      obstacle.angle = np.arctan2(dy, dx)

      # ✅ Build prediction horizon
      pred_steps = []
      s_future = s_now
      for _ in range(horizon):
          s_future = min(s_future + speed * dt, total_length)
          px = path.x_spline(s_future)
          py = path.y_spline(s_future)
          pz = path.z_spline(s_future) if path.z_spline else 0.0

          s_next = min(s_future + ds, total_length)
          dx_f = path.x_spline(s_next) - px
          dy_f = path.y_spline(s_next) - py
          angle = np.arctan2(dy_f, dx_f)

          pos = np.array([px, py, pz])

          # Add noise & uncertainty
          if pred.type == PredictionType.GAUSSIAN:
              pos += np.random.normal(0, sigma_pos, size=pos.shape)
              major_r, minor_r = sigma_pos * 2, sigma_pos
          elif pred.type == PredictionType.NONGAUSSIAN:
              pos += np.random.standard_t(df=3, size=pos.shape) * sigma_pos
              major_r, minor_r = sigma_pos * 3, sigma_pos * 1.5
          else:
              major_r, minor_r = 0.1, 0.1

          pred_steps.append(PredictionStep(pos, angle, major_r, minor_r))

      pred.steps = pred_steps


def remove_distant_obstacles(obstacles: list[DynamicObstacle], state: 'State') -> None:
    """Remove obstacles that are far from the current state."""
    nearby_obstacles = []

    pos = state.get_position()
    for obstacle in obstacles:
        if distance(pos, obstacle.position) < CONFIG["max_obstacle_distance"]:
            nearby_obstacles.append(obstacle)

    obstacles.clear()
    obstacles.extend(nearby_obstacles)

def filter_distant_obstacles(obstacles: list[DynamicObstacle], state: 'State', distance_limit = None):
    """Remove obstacles that are far from the current state."""
    nearby_obstacles = []

    dist = 0
    if distance_limit is not None:
        dist = distance_limit
    else:
        dist = CONFIG["max_obstacle_distance"]
    pos = state.get_position()
    for obstacle in obstacles:
        if distance(pos, obstacle.position) < dist:
            nearby_obstacles.append(obstacle)

    return nearby_obstacles


def ensure_obstacle_size(obstacles: list[DynamicObstacle], state: 'State') -> None:
    """Ensure that the number of obstacles matches the configured maximum."""
    max_obstacles = CONFIG["max_obstacles"]
    # Create an index list
    indices = list(range(len(obstacles)))

    # If more, we sort and retrieve the closest obstacles
    if len(obstacles) > max_obstacles:
        distances = []
        logging.debug(f"Received {len(obstacles)} > {max_obstacles} obstacles. Keeping the closest.")

        for obstacle in obstacles:
            min_dist = 1e5
            direction = np.array([np.cos(state.get("psi")), np.sin(state.get("psi"))])

            for k in range(CONFIG["horizon"]):
                # Linearly scaled
                dist = (k + 1) * 0.6 * distance(
                    obstacle.prediction.steps[k].position,
                    state.get_position() + state.get("v") * k * direction
                )

                if dist < min_dist:
                    min_dist = dist

            distances.append(min_dist)

        # Sort obstacles on distance
        indices.sort(key=lambda j: distances[j])

        # Keep the closest obstacles
        processed_obstacles = []

        for v in range(max_obstacles):
            processed_obstacles.append(obstacles[indices[v]])

        # Assign sequential IDs
        for i in range(len(processed_obstacles)):
            processed_obstacles[i].index = i

        obstacles.clear()
        obstacles.extend(processed_obstacles)

    elif len(obstacles) < max_obstacles:
        logging.debug(f"Received {len(obstacles)} < {max_obstacles} obstacles. Adding dummies.")

        for cur_size in range(len(obstacles), max_obstacles):
            dummy = get_dummy_obstacle(state)
            dummy.prediction = get_constant_velocity_prediction(
                dummy.position,
                np.array([0., 0.]),
                CONFIG["integrator_step"],
                CONFIG["horizon"]
            )
            obstacles.append(dummy)

    logging.debug(f"Obstacle size (after processing) is: {len(obstacles)}")


def propagate_prediction_uncertainty(prediction: Prediction) -> None:
    """Propagate uncertainty through the prediction steps."""
    if prediction.type != PredictionType.GAUSSIAN:
        return

    dt = CONFIG["integrator_step"]
    major = 0.
    minor = 0.

    for k in range(10):
        major = np.sqrt(major ** 2 + (prediction.steps[k].major_radius * dt) ** 2)
        minor = np.sqrt(minor ** 2 + (prediction.steps[k].minor_radius * dt) ** 2)
        prediction.steps[k].major_radius += major # This was originally straight assignment not addition
        prediction.steps[k].minor_radius += minor

def propagate_prediction_uncertainty_for_obstacles(obstacles: list[DynamicObstacle]) -> None:
    """Propagate uncertainty for all obstacles."""
    for obstacle in obstacles:
        propagate_prediction_uncertainty(obstacle.prediction)