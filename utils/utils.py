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

from datetime import datetime

class Timer:
    def __init__(self, hard_stop):
        self.max_runtime = hard_stop
        self.start_time = None
        self.stop_time = None

    def set_max_runtime(self, duration):
        """Set the maximum runtime in seconds."""
        self.max_runtime = duration

    def start(self):
        """Start the timer."""
        self.start_time = datetime.now()
        self.stop_time = None  # In case it's a restart

    def current_runtime(self):
        """Get the current duration in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.stop_time or datetime.now()
        return (end_time - self.start_time).total_seconds()

    def has_finished(self):
        """Check if the timer has finished."""
        return self.current_runtime() >= self.max_runtime

    def stop(self):
        """Stop the timer."""
        if self.start_time is not None:
            self.stop_time = datetime.now()

    def reset(self):
        """Reset the timer."""
        self.stop_time = None
        self.start_time = None

# Chrome trace event format profiler
class ProfileResult:
    def __init__(self, name, start, end, thread_id):
        self.Name = name
        self.Start = start
        self.End = end
        self.ThreadID = thread_id


class DataSaver:
    def __init__(self):
        self.data: Dict[str, List[Union[float, int, str]]] = {}
        self.add_timestamp = False

    def set_add_timestamp(self, value: bool):
        """Enable/disable automatic timestamp addition"""
        self.add_timestamp = value

    def clear(self):
        """Clear all stored data"""
        self.data.clear()

    def add_data(self, key: str, value: Union[float, int, str, List]):
        """
        Add data to the specified key. Values are stored as lists to support multiple entries.

        Args:
            key: The data key/name
            value: The value to store (will be converted to list if not already)
        """
        if key not in self.data:
            self.data[key] = []

        # Handle different value types
        if isinstance(value, list):
            self.data[key].extend(value)
        else:
            self.data[key].append(value)

        # Add timestamp if enabled
        if self.add_timestamp:
            timestamp_key = f"{key}_timestamp"
            if timestamp_key not in self.data:
                self.data[timestamp_key] = []
            self.data[timestamp_key].append(datetime.now().isoformat())

    def save_data(self, filename: str, folder: str = ".", use_json: bool = True):
        """
        Save data to file

        Args:
            filename: Name of the file (without extension)
            folder: Folder to save in (default: current directory)
            use_json: If True, save as JSON; if False, save as plain text
        """
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        if use_json:
            filepath = os.path.join(folder, f"{filename}.json")
            with open(filepath, "w") as f:
                json.dump(self.data, f, indent=2)
        else:
            filepath = os.path.join(folder, f"{filename}.txt")
            with open(filepath, "w") as f:
                for key, values in self.data.items():
                    f.write(f"{key}: {values}\n")

    def load_data(self, filename: str, folder: str = ".", use_json: bool = True) -> Tuple[bool, Dict[str, List]]:
        """
        Load data from file

        Args:
            filename: Name of the file (without extension)
            folder: Folder to load from (default: current directory)
            use_json: If True, expect JSON format; if False, expect plain text

        Returns:
            Tuple of (success: bool, data: Dict[str, List])
        """
        try:
            if use_json:
                filepath = os.path.join(folder, f"{filename}.json")
                with open(filepath, "r") as f:
                    loaded_data = json.load(f)
            else:
                # Simple text format parsing (basic implementation)
                filepath = os.path.join(folder, f"{filename}.txt")
                loaded_data = {}
                with open(filepath, "r") as f:
                    for line in f:
                        if ": " in line:
                            key, value_str = line.strip().split(": ", 1)
                            try:
                                # Try to parse as list
                                loaded_data[key] = eval(value_str)
                            except:
                                loaded_data[key] = [value_str]

            self.data = loaded_data
            return True, loaded_data
        except FileNotFoundError:
            return False, {}
        except Exception as e:
            print(f"Error loading data: {e}")
            return False, {}

    def get_file_path(self, folder: str, filename: str, extension: str = "json") -> str:
        """
        Get the full file path

        Args:
            folder: Folder path
            filename: Filename (without extension)
            extension: File extension (default: json)

        Returns:
            Full file path
        """
        return os.path.join(folder, f"{filename}.{extension}")

    def get_data(self, key: str) -> List:
        """Get data for a specific key"""
        return self.data.get(key, [])

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the data"""
        return key in self.data

    def get_keys(self) -> List[str]:
        """Get all data keys"""
        return list(self.data.keys())

    def remove_key(self, key: str):
        """Remove a key and its data"""
        if key in self.data:
            del self.data[key]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of stored data"""
        summary = {}
        for key, values in self.data.items():
            summary[key] = {
                'count': len(values),
                'type': type(values[0]).__name__ if values else 'empty',
                'sample_values': values[:3] if len(values) > 3 else values
            }
        return summary

    def print_summary(self):
        """Print a summary of stored data"""
        print("DataSaver Summary:")
        print("-" * 40)
        for key, info in self.get_summary().items():
            print(f"{key}: {info['count']} items ({info['type']})")
            if info['sample_values']:
                print(f"  Sample: {info['sample_values']}")
        print("-" * 40)


class ExperimentManager:
    def __init__(self):
        print("CONFIG is " + str(CONFIG))
        self.SAVE_FOLDER = CONFIG["recording"]["folder"]
        self.SAVE_FILE = CONFIG["recording"]["file"]
        self.timer = Timer(1.0)
        self.data_saver = DataSaver()
        self.data_saver.set_add_timestamp(CONFIG["recording"]["timestamp"])

        if CONFIG["recording"]["enable"]:
            logger.info(
                f"Planner Save File: {self.data_saver.get_file_path(self.SAVE_FOLDER, self.SAVE_FILE, False)}")

        self.control_iteration = 0
        self.iteration_at_last_reset = 0
        self.experiment_counter = 0

    def update(self, state, solver, data):
        logger.info("planning.util.save_data()")

        if len(data.dynamic_obstacles) == 0:
            logger.info("Not exporting data: Obstacles not yet received.")
            return

        # Save vehicle data
        self.data_saver.add_data("vehicle_pose", state.get_pose())
        self.data_saver.add_data("vehicle_orientation", state.get("psi"))

        # Save planned trajectory
        for k in range(CONFIG["horizon"]):
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

        assert self.experiment_counter < num_experiments, "Stopping the planning."

    def set_start_experiment(self, duration=1):
        self.iteration_at_last_reset = self.control_iteration
        self.set_timer(duration)

    def set_timer(self, duration):
        self.timer = Timer(duration)

    def start_timer(self):
        self.timer.start()

    def stop_timer(self):
        duration = self.timer.current_duration()
        self.timer = Timer(1.0)
        return duration

    def get_data_saver(self):
        return self.data_saver