import copy
import os

import numpy as np
from utils.utils import CONFIG, load_settings, print_header, print_value, parameter_map_path, write_to_yaml, \
    get_current_package, get_package_path, print_path, print_success

# Python real-time
class RealTimeParameters:

  def __init__(self, settings):
    self._map = load_settings("parameter_map")
    self._settings = settings

    self._num_p = self._map['num parameters']
    self.params = np.zeros((settings["N"], self._num_p))

  def set(self, k, parameter, value):
    if parameter in self._map.keys():
      self.params[k, self._map[parameter]] = value
      # print(f"{parameter} set to {value} | map value: {self._map[parameter]} check: {self.params[self._map[parameter]]}")

  def get(self, k, parameter):
    return self.params[k, self._map[parameter]]

  def get_solver_params(self):
    out = []
    for k in range(self._settings["N"]):
      for i in range(self._num_p):
        out.append(self.params[k, i])
    return out

  def get_solver_params_for_stage(self, k):
    out = []
    for i in range(self._num_p):
      out.append(self.params[k, i])
    return out

  def get_num_par(self):
    return self._num_p


# Python real-time
class RealTimeModel:
  def __init__(self, settings, solver_settings):
    self._map = load_settings("model_map")
    self._settings = settings

    self._N = settings["N"]
    self._nu = solver_settings["nu"]
    self._nx = solver_settings["nx"]
    self._nvar = self._nu + self._nx
    self._vars = np.zeros((settings["N"], self._nvar))

  def get(self, k, var_name):
    map_value = self._map[var_name]
    return self._vars[k, map_value[1]]

# Python real-time
class ForcesRealTimeModel(RealTimeModel):


  def __init__(self, settings, solver_settings):
    super().__init__(settings, solver_settings)

  def load(self, forces_output):
    for k in range(self._N):
      for var in range(self._nu):
        if k + 1< 10:
          self._vars[k, var] = forces_output[f"x0{k+1}"][var]
        else:
          self._vars[k, var] = forces_output[f"x{k+1}"][var]
      for var in range(self._nu, self._nvar):
        if k + 1< 10:
          self._vars[k, var] = forces_output[f"x0{k+1}"][var - self._nu]
        else:
          self._vars[k, var] = forces_output[f"x{k+1}"][var - self._nu]

  def get_trajectory(self, forces_output, mpc_x_plan, mpc_u_plan):

    for k in range(self._N):
      for i in range(self._nu):
        if k + 1 < 10:
          mpc_u_plan[i, k] = forces_output[f"x0{k+1}"][i]
        else:
          mpc_u_plan[i, k] = forces_output[f"x{k+1}"][i]
      for i in range(self._nu, self._nvar):
        if k + 1 < 10:
          mpc_x_plan[i - self._nu, k] = forces_output[f"x0{k+1}"][i]
        else:
          mpc_x_plan[i - self._nu, k] = forces_output[f"x{k+1}"][i]

    return np.concatenate([mpc_u_plan, mpc_x_plan])



def generate_rqt_reconfigure(settings):
    """Generate configuration files for RQT Reconfigure."""
    current_package = get_current_package()
    system_name = "".join(current_package.split("_")[2:])
    path = f"{get_package_path(current_package)}/cfg/"
    os.makedirs(path, exist_ok=True)
    path += f"{system_name}.cfg"
    print_path("RQT Reconfigure", path, end="", tab=True)

    with open(path, "w") as rqt_file:
        rqt_file.write("#!/usr/bin/env python\n")
        rqt_file.write(f'PACKAGE = "{current_package}"\n')
        rqt_file.write("from dynamic_reconfigure.parameter_generator_catkin import *\n")
        rqt_file.write("gen = ParameterGenerator()\n\n")

        rqt_file.write('weight_params = gen.add_group("Weights", "Weights")\n')
        rqt_params = settings["params"].rqt_params
        for idx, param in enumerate(rqt_params):
            rqt_file.write(
                f'weight_params.add("{param}", double_t, 1, "{param}", 1.0, '
                f'{settings["params"].rqt_param_min_values[idx]}, '
                f'{settings["params"].rqt_param_max_values[idx]})\n'
            )
        rqt_file.write(f'exit(gen.generate(PACKAGE, "{current_package}", "{system_name}"))\n')

    print_success(" -> generated")


def get_parameter_bundle_values(settings):
    """Get parameter bundle values from settings."""
    parameter_bundles = {}

    for key, indices in settings["params"].parameter_bundles.items():
        function_name = key.replace("_", " ").title().replace(" ", "")
        parameter_bundles[function_name] = indices

    return parameter_bundles


def set_solver_parameters(k, params, parameter_name, value, index=0):
    """Set solver parameters based on parameter name."""
    parameter_bundles = get_parameter_bundle_values(CONFIG)

    if parameter_name in parameter_bundles:
        indices = parameter_bundles[parameter_name]
        if len(indices) == 1:
            params.all_parameters[k * CONFIG['params'].length() + indices[0]] = value
        else:
            if 0 <= index < len(indices):
                params.all_parameters[k * CONFIG['params'].length() + indices[index]] = value
    else:
        raise ValueError(f"Unknown parameter: {parameter_name}")