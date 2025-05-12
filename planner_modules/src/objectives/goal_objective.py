from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.objectives.base_objective import BaseObjective
from utils.const import OBJECTIVE
from utils.math_utils import distance
from utils.utils import LOG_DEBUG


class GoalObjective(BaseObjective):

  def __init__(self, solver):
    super().__init__(solver)
    self.module_type = OBJECTIVE
    self.name = "goal_objective"
    LOG_DEBUG( "Initializing Goal Module")
    LOG_DEBUG( "Goal Module successfully initialized")
    self.goal_weight = self.get_config_value("goal_weight")

  def update(self, state, data, module_data):
    return

  def define_parameters(self, params):
    params.add("goal_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["goal"]')
    params.add("goal_x")
    params.add("goal_y")

  def get_value(self, model, params, settings, stage_idx):
    cost = 0

    # if stage_idx == settings["N"] - 1:
    pos_x = model.get("x")
    pos_y = model.get("y")

    goal_weight = params.get("goal_weight")

    goal_x = params.get("goal_x")
    goal_y = params.get("goal_y")

    cost += goal_weight * ((pos_x - goal_x) ** 2 + (pos_y - goal_y) ** 2) / (goal_x ** 2 + goal_y ** 2 + 0.01)

    return cost

  def set_parameters(self, parameter_manager, data, module_data, k):

    if k == 0:
      LOG_DEBUG( "Goal Module.set_parameters()")

    parameter_manager.set_parameter("goal_x", data.goal(0))
    parameter_manager.set_parameter("goal_y", data.goal(1))
    parameter_manager.set_parameter("goal_weight", self.goal_weight)

  def is_data_ready(self, data):
    missing_data = ""
    if not data.goal_received:
      missing_data += "Goal "

    return len(missing_data) < 1

  def is_objective_reached(self, state, data):

    if not data.goal_received:
      return False

    # Check if we reached the goal
    return distance(state.get_pos(), data.goal) < 1.0