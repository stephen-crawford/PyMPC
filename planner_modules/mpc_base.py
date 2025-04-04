from planner.src.data_prep import logger
from utils.const import OBJECTIVE


from utils.utils import read_config_file

CONFIG = read_config_file()


class MPCBaseModule:

 def __init__(self, solver):
  self.solver = solver
  self.module_type = OBJECTIVE
  self.name = "mpc_base"
  self.weight_names = WEIGHT_PARAMS

 def update(self, state, data, module_data):
  return

 def set_parameters(self, data, module_data, k):

  if (k == 0):
   logger.log(10, "MPCBaseModule.set_parameters()")

  for weight in self.weight_names:

   _solver.set_parameter(k, weight, CONFIG["weights"][weight])
