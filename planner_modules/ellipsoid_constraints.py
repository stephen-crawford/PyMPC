from venv import logger

from utils.const import OBJECTIVE, CONSTRAINT, DETERMINISTIC, GAUSSIAN
from utils.utils import read_config_file

CONFIG = read_config_file()


class EllipsoidConstraints:

 def __init__(self, solver):
  self.solver = solver
  self.module_type = CONSTRAINT
  self.name = "ellipsoid_constraints"
  LOG_DEBUG( "Initializing Ellipsoid Constraints")
  self._get_num_segments = CONFIG["contouring"]["get_num_segments"]
  self._n_discs = CONFIG["n_discs"]
  self._robot_radius = CONFIG["robot_radius"]
  self._risk = CONFIG["probabilistic"]["risk"]
  LOG_DEBUG( "Ellipsoid Constraints successfully initialized")

 def update(self, state, data, module_data):

  _dummy_x = state.get("x") + 50
  _dummy_y = state.get("y") + 50

 def set_parameters(self, data, module_data, k):

  set_solver_parameter_ego_disc_radius(k, self.solver._params, _robot_radius)
  for d in range(_n_discs):
   set_solver_parameter_ego_disc_self.offset(k, self.solver._params, data.robot_area[d].self.offset, d)

  if k == 0: # Dummies

   # logger.log("Setting parameters for k = 0")
   for i in range(data.dynamic_obstacles.size()):
    set_solver_parameter_ellipsoid_obst_x(0, self.solver._params, _dummy_x, i)
    set_solver_parameter_ellipsoid_obst_y(0, self.solver._params, _dummy_y, i)
    set_solver_parameter_ellipsoid_obst_psi(0, self.solver._params, 0., i)
    set_solver_parameter_ellipsoid_obst_r(0, self.solver._params, 0.1, i)
    set_solver_parameter_ellipsoid_obst_major(0, self.solver._params, 0., i)
    set_solver_parameter_ellipsoid_obst_minor(0, self.solver._params, 0., i)
    set_solver_parameter_ellipsoid_obst_chi(0, self.solver._params, 1., i)
   
   return

  if k == 1:
   LOG_DEBUG( "EllipsoidConstraints::set_parameters")

  for i in range(data.dynamic_obstacles.size()):
  
   obstacle = data.dynamic_obstacles[i]
   mode = obstacle.prediction.modes[0]

   # The first prediction step is index 1 of the optimization problem, i.e., k-1 maps to the predictions for this stage 
   set_solver_parameter_ellipsoid_obst_x(k, self.solver._params, mode[k - 1].position(0), i)
   set_solver_parameter_ellipsoid_obst_y(k, self.solver._params, mode[k - 1].position(1), i)
   set_solver_parameter_ellipsoid_obst_psi(k, self.solver._params, mode[k - 1].angle, i)
   set_solver_parameter_ellipsoid_obst_r(k, self.solver._params, obstacle.radius, i)

   if obstacle.prediction.type == DETERMINISTIC:
   
    set_solver_parameter_ellipsoid_obst_major(k, self.solver._params, 0., i)
    set_solver_parameter_ellipsoid_obst_minor(k, self.solver._params, 0., i)
    set_solver_parameter_ellipsoid_obst_chi(k, self.solver._params, 1., i)
   
   elif obstacle.prediction.type == GAUSSIAN:
     chi = RosTools.exponential_quantile(0.5, 1.0 - _risk)

    set_solver_parameter_ellipsoid_obst_major(k, self.solver._params, mode[k - 1].major_radius, i)
    set_solver_parameter_ellipsoid_obst_minor(k, self.solver._params, mode[k - 1].minor_radius, i)
    set_solver_parameter_ellipsoid_obst_chi(k, self.solver._params, chi, i)


  if k == 1:
   LOG_DEBUG( "EllipsoidConstraints::set_parameters Done")

 def is_data_ready(self, data, missing_data):
  if data.robot_area.size() == 0:
   missing_data += "Robot area "
   return False


  if data.dynamic_obstacles.size() != CONFIG["max_obstacles"]:
   missing_data += "Obstacles "
   return False


  for i in range(data.dynamic_obstacles.size()):
   if data.dynamic_obstacles[i].prediction.empty():

    missing_data += "Obstacle Prediction "
    return False

   if data.dynamic_obstacles[i].prediction.type != GAUSSIAN and data.dynamic_obstacles[i].prediction.type != DETERMINISTIC:

    missing_data += "Obstacle Prediction (Type is incorrect) "
    return False

  return True

