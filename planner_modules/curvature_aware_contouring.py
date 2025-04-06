
from utils.utils import read_config_file
CONFIG = read_config_file()

class CurvatureAwareContouring:

  def __init__(self, solver):
    self.solver = solver

  def set_parameters(self, data, module_data, k):

    if k == 0:

      contouring_weight = CONFIG["weights"]["contour"]

      terminal_angle_weight = CONFIG["weights"]["terminal_angle"]
      terminal_contouring_weight = CONFIG["weights"]["terminal_contouring"]

      if _dynamic_velocity_reference:
        velocity_weight = CONFIG["weights"]["velocity"]
        reference_velocity = CONFIG["weights"]["reference_velocity"]


      set_solver_parameter_contour(k, self.solver._params, contouring_weight)

      set_solver_parameter_terminal_angle(k, self.solver._params, terminal_angle_weight)
      set_solver_parameter_terminal_contouring(k, self.solver._params, terminal_contouring_weight)

      if _dynamic_velocity_reference:
        set_solver_parameter_velocity(k, self.solver._params, velocity_weight)
        set_solver_parameter_reference_velocity(k, self.solver._params, reference_velocity)

    setspline_parameters(k)