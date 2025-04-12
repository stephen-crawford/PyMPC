import os, sys
import yaml

from solver_generator.util.logging import print_success, print_value, print_path

def get_base_path():
  return os.path.dirname(os.path.realpath('../../utils'))


def get_current_package():
  return os.path.dirname(os.path.realpath(sys.argv[0])).split("/")[-2]


def get_package_path(package_name):
  return os.path.join(os.path.dirname(__file__), f"../../{package_name}")


def getsolver_package_path():
  return get_package_path("mpc_plannersolver")


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


def defaultsolver_path(settings):
  return os.path.join(os.getcwd(), f"{solver_name(settings)}")


def solver_path(settings):
  return os.path.join(getsolver_package_path(), f"{solver_name(settings)}")


def default_casadisolver_path(settings):
  return os.path.join(get_package_path("solver_generator"), f"casadi")


def casadisolver_path(settings):
  return os.path.join(getsolver_package_path(), f"casadi")


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
