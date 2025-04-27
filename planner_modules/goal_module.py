from planner_modules.base_constraint import BaseConstraint
from utils.const import OBJECTIVE

from utils.utils import read_config_file, LOG_DEBUG, distance
from utils.visualizer import VISUALS



class GoalModule(BaseConstraint):

  def __init__(self, solver):
    super().__init__(solver)
    self.module_type = OBJECTIVE
    self.name = "goal_module"
    LOG_DEBUG( "Initializing Goal Module")
    LOG_DEBUG( "Goal Module successfully initialized")

  def update(self, state, data, module_data):
    return

  def set_parameters(self, data, module_data, k):
  
    
    if k == 0:
      LOG_DEBUG( "Goal Module.set_parameters()")

    set_solver_parameter_goal_x(k, self.solver.params, data.goal(0))
    set_solver_parameter_goal_y(k, self.solver.params, data.goal(1))
    set_solver_parameter_goal_weight(k, self.solver.params, CONFIG["weights"]["goal"])

  def is_data_ready(self, data, missing_data):
    if not data.goal_received:
      missing_data += "Goal "

    return data.goal_received

  def is_objective_reached(self, state, data):

    if not data.goal_received:
      return False

    # Check if we reached the goal
    return distance(self, state.get_pos(), data.goal) < 1.0


  def visualize(self, data, module_data):

    if not data.goal_received:
      return

    publisher = VISUALS.get_publisher(self.name)
    sphere = publisher.get_new_point_marker("SPHERE")

    sphere.set_color_int(5)
    sphere.set_scale(0.4, 0.4, 0.4)
    sphere.add_point_marker(data.goal, 0.0)

    publisher.publish()