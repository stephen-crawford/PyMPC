from planner_modules.contouring import Contouring
from solver.solver_interface import set_solver_parameter
from utils.utils import read_config_file
CONFIG = read_config_file()

class CurvatureAwareContouring:

  def __init__(self, solver):
    self.solver = solver
    self.dynamic_velocity_reference = None
    self.contouring = Contouring(solver)

  def set_parameters(self, data, module_data, k):

    if k == 0:

      contouring_weight = CONFIG["weights"]["contour"]

      terminal_angle_weight = CONFIG["weights"]["terminal_angle"]
      terminal_contouring_weight = CONFIG["weights"]["terminal_contouring"]

      if self.dynamic_velocity_reference:
        velocity_weight = CONFIG["weights"]["velocity"]
        reference_velocity = CONFIG["weights"]["reference_velocity"]

      set_solver_parameter(self.solver.params, "contour", contouring_weight, k, settings=CONFIG)
      set_solver_parameter(self.solver.params, "terminal_angle", terminal_angle_weight, settings=CONFIG)

      if self.dynamic_velocity_reference:
        set_solver_parameter(self.solver.params, "velocity", velocity_weight, settings=CONFIG)
        set_solver_parameter(self.solver.params, "reference_velocity", reference_velocity, settings=CONFIG)

    Contouring.set_spline_parameters(k)