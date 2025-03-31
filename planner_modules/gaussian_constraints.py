from numpy import sqrt

from planner.data_prep import logger
from utils.const import CONSTRAINT, GAUSSIAN, DYNAMIC
from utils.utils import read_config_file

CONFIG = read_config_file()
class GaussianConstraints:
  
  def __init__(self, solver):

    self.module_type = CONSTRAINT
    self.solver = solver
    self.name = "gaussian_constraints"
      
 
    logger.log(10, "Initializing Gaussian Constraints")
    logger.log(10, "Gaussian Constraints successfully initialized")

  def update(self, state, data, module_data):
    _dummy_x = state.get("x") + 100.
    _dummy_y = state.get("y") + 100.

  def set_parameters(self, data, module_data, k):

    set_solver_parameter_ego_disc_radius(k, self._solver._params, CONFIG["robot_radius"])
    for d in range(CONFIG["n_discs"]):
      set_solver_parameter_ego_disc_offset(k, self._solver._params, data.robot_area[d].offset, d)

    if k == 0: # Dummies
      for i in range(data.dynamic_obstacles.size()):
        set_solver_parameter_gaussian_obstacle_x(k, self._solver._params, _dummy_x, i)
        set_solver_parameter_gaussian_obstacle_y(k, self._solver._params, _dummy_y, i)
        set_solver_parameter_gaussian_obstacle_major(k, self._solver._params, 0.1, i)
        set_solver_parameter_gaussian_obstacle_minor(k, self._solver._params, 0.1, i)
        set_solver_parameter_gaussian_obstacle_risk(k, self._solver._params, 0.05, i)
        set_solver_parameter_gaussian_obstacle_r(k, self._solver._params, 0.1, i)
      return

    copied_obstacles = data.dynamic_obstacles

    for i in range(copied_obstacles.size()):

      obstacle = copied_obstacles[i]

      if obstacle.prediction.type == GAUSSIAN:
        set_solver_parameter_gaussian_obstacle_x(k, self._solver._params, obstacle.prediction.modes[0][k - 1].position(0), i)
        set_solver_parameter_gaussian_obstacle_y(k, self._solver._params, obstacle.prediction.modes[0][k - 1].position(1), i)

        if obstacle.type == DYNAMIC:
          set_solver_parameter_gaussian_obstacle_major(k, self._solver._params, obstacle.prediction.modes[0][k - 1].major_radius, i)
          set_solver_parameter_gaussian_obstacle_minor(k, self._solver._params, obstacle.prediction.modes[0][k - 1].minor_radius, i)
        else: # Static obstacles have no uncertainty
          set_solver_parameter_gaussian_obstacle_major(k, self._solver._params, 0.001, i)
          set_solver_parameter_gaussian_obstacle_minor(k, self._solver._params, 0.001, i)
        set_solver_parameter_gaussian_obstacle_risk(k, self._solver._params, CONFIG["probabilistic"]["risk"], i)
        set_solver_parameter_gaussian_obstacle_r(k, self._solver._params, CONFIG["obstacle_radius"], i)

  def is_data_ready(self, data, missing_data):
    if data.dynamic_obstacles.size() != CONFIG["max_obstacles"]:
      missing_data += "Obstacles "
      return False

    for i in range(data.dynamic_obstacles.size()):
      if (data.dynamic_obstacles[i].prediction.modes.empty()):
        missing_data += "Obstacle Prediction "
        return False

      if data.dynamic_obstacles[i].prediction.type != GAUSSIAN:
        missing_data += "Obstacle Prediction (Type is not Gaussian) "
        return False

    return True

  def visualize(self, data,module_data):

    PROFILE_SCOPE("GuidanceConstraints::Visualize")
    logger.log(10, "GaussianConstraints.visualize")
    publisher = VISUALS.missing_data(_name)

    ellipsoid = publisher.get_new_point_marker("CYLINDER")

    for obstacle in data.dynamic_obstacles:

      k = 1
      while k < _solver.N:

        ellipsoid.set_color_int(k, _solver.N, 0.5)

        if obstacle.type == DYNAMIC:
          chi = RosTools.exponential_quantile(0.5, 1.0 - CONFIG["probabilistic"]["risk"])
        else:
          chi = 0.
        ellipsoid.set_scale(2 * (obstacle.prediction.modes[0][k - 1].major_radius * sqrt(chi) + obstacle.radius),
                           2 * (obstacle.prediction.modes[0][k - 1].major_radius * sqrt(chi) + obstacle.radius), 0.005)

        ellipsoid.add_point_marker(obstacle.prediction.modes[0][k - 1].position)

        k += CONFIG["visualization"]["draw_every"]
    publisher.publish()

