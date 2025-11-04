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
    """Get a nested config value using dotted key notation (e.g., 'planner.horizon').
    
    Args:
        config: The config dictionary
        dotted_key: Dot-separated key path (e.g., 'planner.horizon')
        default: Default value if key not found
        
    Returns:
        The config value or default if not found
    """
    if config is None:
        return default
    
    keys = dotted_key.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
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

# Forward reference for type hints - Disc is defined in planning.types
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from planning.types import Disc

def define_robot_area(length: float, width: float, n_discs: int):
    """Define robot area with discs. Implementation moved to planning.types."""
    from planning.types import define_robot_area as _define_robot_area, Disc
    return _define_robot_area(length, width, n_discs)

