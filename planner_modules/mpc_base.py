import logging

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
   logging.DEBUG("MPCBaseModule.set_parameters()")

  for weight in self.weight_names:

   solver.set_parameter(k, weight, CONFIG["weights"][weight])
