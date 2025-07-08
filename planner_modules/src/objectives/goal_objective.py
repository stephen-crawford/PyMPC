import numpy as np
import casadi as cs
from planner_modules.src.constraints.base_constraint import BaseConstraint
from planner_modules.src.objectives.base_objective import BaseObjective
from planning.src.types import State
from utils.const import OBJECTIVE
from utils.math_utils import distance
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO


class GoalObjective(BaseObjective):

  def __init__(self, solver):
    super().__init__(solver)
    self.module_type = OBJECTIVE
    self.name = "goal_objective"
    LOG_INFO( "Initializing Goal Module")
    LOG_INFO( "Goal Module successfully initialized")
    self.goal_weight = self.get_config_value("weights.goal_weight")

  def update(self, state, data):
    return

  def define_parameters(self, params):
    print("Defining parameters for Goal Objective")
    params.add("goal_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["goal"]')
    params.add("control_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["control"]')

    params.add("angle_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["angle"]')
    params.add("goal_x")
    params.add("goal_y")


  def get_value(self, params, stage_idx):

    pos_x = self.solver.get("x", stage_idx)
    pos_y = self.solver.get("y", stage_idx)
    psi = self.solver.get("psi", stage_idx)

    goal_weight = params.get("goal_weight")
    angle_weight = params.get("angle_weight")
    goal_x = params.get("goal_x")
    goal_y = params.get("goal_y")

    # Positional error
    pos_cost = goal_weight * ((pos_x - goal_x) ** 2 + (pos_y - goal_y) ** 2) / (goal_x ** 2 + goal_y ** 2 + 0.01)

    # Angular error to face the goal
    theta_goal = cs.atan2(goal_y - pos_y, goal_x - pos_x)
    angle_error = cs.fmod(psi - theta_goal + cs.pi, 2 * cs.pi) - cs.pi

    angle_cost = angle_weight * angle_error ** 2

    cost = pos_cost + angle_cost
    return {"goal_pos_cost": pos_cost, "goal_angle_cost": angle_cost}


  def set_parameters(self, parameter_manager, data, k):
    LOG_DEBUG("Setting parameters for Goal Objective")
    if k == 0:
      LOG_DEBUG( "Goal Module.set_parameters()")

    parameter_manager.set_parameter("goal_x", data.goal[0])
    parameter_manager.set_parameter("goal_y", data.goal[1])
    parameter_manager.set_parameter("goal_weight", self.goal_weight)

  def is_data_ready(self, data):
    missing_data = ""
    if not data.has("goal_received") or data.goal_received is None or data.goal_received == False:
      missing_data += "Goal "

    return len(missing_data) < 1

  def is_objective_reached(self, state: State, data):
    if not data.goal_received:
      return False

    # Check if we reached the goal
    reached = distance(state.get_position(), data.goal) < 1.0
    LOG_DEBUG(f"Goal Objective.is_objective_reached(): {reached}")
    return reached