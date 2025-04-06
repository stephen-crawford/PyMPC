from logging import Logger

from planner.src.data_prep import logger
from utils.const import CONSTRAINT, DETERMINISTIC, GAUSSIAN
from utils.utils import read_config_file

CONFIG = read_config_file()

class LinearizedConstraints:

 def __init__(self, solver):
  self.module_type = CONSTRAINT
  self.solver = solver
  self.name = "linearized_constraints"
  self.controller = (self.module_type, solver, self.name)
  Logger.log(10, "Initializing Linearized Constraints")
  self._n_discs = CONFIG["n_discs"] # Is overwritten to 1 for topology constraints  
  self._n_other_halfspaces = CONFIG["linearized_constraints"]["add_halfspaces"]
  self._max_obstacles = CONFIG["max_obstacles"]
  self.n_constraints = self._max_obstacles + self._n_other_halfspaces
  self._a1.resize(CONFIG["n_discs"])
  self._a2.resize(CONFIG["n_discs"])
  self._b.resize(CONFIG["n_discs"])
  for d in range(CONFIG["n_discs"]):
   self._a1[d].resize(CONFIG["N"])
   self._a2[d].resize(CONFIG["N"])
   self._b[d].resize(CONFIG["N"])
   for k in range(CONFIG["N"]):
    self._a1[d][k] = Eigen.ArrayXd(n_constraints)
    self._a2[d][k] = Eigen.ArrayXd(n_constraints)
    self._b[d][k] = Eigen.ArrayXd(n_constraints)
  self._num_obstacles = 0
  Logger.log(10, "Linearized Constraints successfully initialized")


 def setTopologyConstraints(self):
  self._n_discs = 1 # Only one disc is used for the topology constraints
  self._use_guidance = True

 def update(self, state, data, module_data):
  
  logger.log(10, "LinearizedConstraints::update")

  _dummy_b = state.get("x") + 100.

  # Thread safe
  copied_obstacles = data.dynamic_obstacles
  self._num_obstacles = copied_obstacles.size()

  # For all stages
  k = 1
  while k < self.solver.N:
   for d in range(self._n_discs):
    pos(self.solver.get_ego_prediction(k, "x"), self.solver.get_ego_prediction(k, "y")) # k = 0 is initial state

    if not self._use_guidance: # Use discs and their positions

     disc = data.robot_area[d]

     disc_pos = disc.get_position(pos, self.solver.get_ego_prediction(k, "psi"))
     project_to_safety(copied_obstacles, k, disc_pos) # Ensure that the vehicle position is collision-free

     # TODO: Set projected disc position

     pos = disc_pos
    else: # Use the robot position
     project_to_safety(copied_obstacles, k, pos) # Ensure that the vehicle position is collision-free
     # todo Set projected disc position

    # For all obstacles
    for obs_id in range(copied_obstacles.size()):

     copied_obstacle = copied_obstacles[obs_id]
     obstacle_pos = copied_obstacle.prediction.modes[0][k - 1].position

     diff_x = obstacle_pos(0) - pos(0)
     diff_y = obstacle_pos(1) - pos(1)

     dist = (obstacle_pos - pos).norm()

     # Compute the components of A for this obstacle (normalized normal vector)
     self._a1[d][k](obs_id) = diff_x / dist
     self._a2[d][k](obs_id) = diff_y / dist

     # Compute b (evaluate point on the collision circle)
     if self._use_guidance:
      radius = 1e-3
     else:
      radius = copied_obstacle.radius

     self._b[d][k](obs_id) = self._a1[d][k](obs_id) * obstacle_pos(0) + self._a2[d][k](obs_id) * obstacle_pos(1) - (radius + CONFIG["robot_radius"])

    if not module_data.static_obstacles.empty() and module_data.static_obstacles[k].size() < self._n_other_halfspaces:
     logger.log(10, self._n_other_halfspaces + " halfspaces expected, but " + module_data.static_obstacles[k].size() + " are present")


    if not module_data.static_obstacles.empty():

     num_halfspaces = min(module_data.static_obstacles[k].size(), self._n_other_halfspaces)
     for h in range(num_halfspaces):
     
      obs_id = copied_obstacles.size() + h
      self._a1[d][k](obs_id) = module_data.static_obstacles[k][h].A(0)
      self._a2[d][k](obs_id) = module_data.static_obstacles[k][h].A(1)
      self._b[d][k](obs_id) = module_data.static_obstacles[k][h].b
     
  logger.log(10, "LinearizedConstraints.update done")

 def project_to_safety(self, copied_obstacles, k, pos):
  if copied_obstacles.empty() # There is no anchor
   return

  # Project to a collision free position if necessary, considering all the obstacles
  iterate = 0
  while iterate < 3: # At most 3 iterations
   for obstacle in copied_obstacles:
    
    if self._use_guidance:
     radius = 1e-3 
    else: 
     radius = obstacle.radius

    dr_projection_.douglas_rachford_projection(pos, obstacle.prediction.modes[0][k - 1].position, copied_obstacles[0].prediction.modes[0][k - 1].position, radius + CONFIG["robot_radius"], pos)
   iterate += 1

 def set_parameters(self, data, module_data, k):
  
  constraint_counter = 0 # Necessary for now to map the disc and obstacle index to a single index

  if k == 0:
   i = 0
   while i < self._max_obstacles + self._n_other_halfspaces:
    set_solver_parameter_lin_constraint_a1(0, solver._params, _dummy_a1, constraint_counter)
    set_solver_parameter_lin_constraint_a2(0, solver._params, _dummy_a2, constraint_counter)
    set_solver_parameter_lin_constraint_b(0, solver._params, _dummy_b, constraint_counter)
    constraint_counter+=1
    i += 1
   return

  for d in range(self._n_discs):
   if not self._use_guidance:
    set_solver_parameter_ego_disc_self.offset(k, self.solver._params, data.robot_area[d].self.offset, d)

   i = 0
   while i < data.dynamic_obstacles.size() + self._n_other_halfspaces:
    set_solver_parameter_lin_constraint_a1(k, solver._params, _a1[d][k](i), constraint_counter)
    set_solver_parameter_lin_constraint_a2(k, solver._params, _a2[d][k](i), constraint_counter)
    set_solver_parameter_lin_constraint_b(k, solver._params, _b[d][k](i), constraint_counter)
    constraint_counter+=1
    i += 1

   i = data.dynamic_obstacles.size() + self._n_other_halfspaces
   while i < self._max_obstacles + self._n_other_halfspaces:
    set_solver_parameter_lin_constraint_a1(k, self.solver._params, _dummy_a1, constraint_counter)
    set_solver_parameter_lin_constraint_a2(k, self.solver._params, _dummy_a2, constraint_counter)
    set_solver_parameter_lin_constraint_b(k, self.solver._params, _dummy_b, constraint_counter)
    constraint_counter+=1
    i += 1
 
 def is_data_ready(self, data, missing_data):
  if data.dynamic_obstacles.size() != self._max_obstacles:
   missing_data += "Obstacles "
   return False

  for i in range(data.dynamic_obstacles.size()):
   if data.dynamic_obstacles[i].prediction.empty():
    missing_data += "Obstacle Prediction "
    return False

   if data.dynamic_obstacles[i].prediction.type != DETERMINISTIC and data.dynamic_obstacles[i].prediction.type != GAUSSIAN:
    missing_data += "Obstacle Prediction (type must be deterministic, or gaussian) "
    return False

  return True

 def visualize(self, data, module_data):
  if self._use_guidance and not CONFIG["debug_visuals"]:
   return

  k = 1
  while k < self.solver.N:
   for i in range(data.dynamic_obstacles.size()):
    visualize_linear_constraint(self._a1[0][k](i), self._a2[0][k](i), self._b[0][k](i), k, self.solver.N, _name, k == solver.N - 1 and i == data.dynamic_obstacles.size() - 1) # Publish at the end
   k +=1
