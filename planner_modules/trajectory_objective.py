import logging

from planner_modules.base_objective import BaseObjective
from utils.const import OBJECTIVE
from utils.utils import read_config_file

CONFIG = read_config_file()

class TrajectoryObjective(BaseObjective):

 def __init__(self, solver):
  super().__init__(solver)
  self.solver = solver
  self.module_type = OBJECTIVE
  self.name = "mpc_base"

  self._weights = []

  self._weights_per_function = []
  self._variables_per_function = []
  self._cost_functions = []
  self._kwarg_list = []

  self.weight_names = WEIGHT_PARAMS

 def update(self, state, data, module_data):
  return

 def define_parameters(self, params):
        for idx, param in enumerate(self._weights):
            params.add(param, add_to_rqt_reconfigure=True, **self._kwarg_list[idx])

        return params

    # Weights w are a parameter vector
    # Only add weights if they are not also parameters!
    def add(self, variable_to_weight, weight_names, cost_function=lambda x, w: w[0] * x**2, **kwargs):

        # # Make sure it's a list if it isn't yet
        if type(weight_names) != list:
            weight_names = [weight_names]

        # # Add all weights in the list
        for weight_name in weight_names:
            self._weights.append(weight_name)
            self._kwarg_list.append(kwargs)

        self._weights_per_function.append(weight_names)
        self._variables_per_function.append(variable_to_weight)
        self._cost_functions.append(cost_function)

    def get_value(self, model, params, settings, stage_idx):
        cost = 0.0
        for idx, cost_function in enumerate(self._cost_functions):
            weights = []
            for cost_weight in self._weights_per_function[idx]:  # Retrieve the weight parameters for this cost function!
                weights.append(params.get(cost_weight))

            variable = model.get(self._variables_per_function[idx])  # Retrieve the state / input to be weighted
            # _, _, var_range = model.get_bounds(self._variables_per_function[idx])

            # Add to the cost
            cost += cost_function(variable, weights)

        return cost

    def get_weights(self) -> list:
        return self._weights


 def set_parameters(self, data, module_data, k):

  if (k == 0):
   logging.DEBUG("TrajectoryObjective.set_parameters()")

  for weight in self.weight_names:

   solver.set_parameter(k, weight, CONFIG["weights"][weight])
