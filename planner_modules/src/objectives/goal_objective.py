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
    params.add("goal_control_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["control"]')
    params.add("goal_terminal_angle_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["terminal_angle"]')
    params.add("goal_angle_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["angle"]')
    params.add("goal_terminal_weight", add_to_rqt_reconfigure=True, rqt_config_name=lambda p: f'["weights"]["terminal"]')
    params.add("goal_x")
    params.add("goal_y")

  def get_value(self, state, params, stage_idx):

    pos_x = state.get("x")
    pos_y = state.get("y")
    psi = state.get("psi")

    goal_x = params.get("goal_x")
    goal_y = params.get("goal_y")

    goal_weight = params.get("goal_weight")
    angle_weight = params.get("goal_angle_weight")
    terminal_goal_weight = params.get("goal_terminal_weight")
    terminal_angle_weight = params.get("goal_terminal_angle_weight")

    LOG_DEBUG("pos x : " + str(pos_x))
    LOG_DEBUG("pos y : " + str(pos_y))
    LOG_DEBUG("psi : " + str(psi))
    LOG_DEBUG("goal_x : " + str(goal_x))
    LOG_DEBUG("goal_y : " + str(goal_y))

    # Compute positional and angle error
    dx = pos_x - goal_x
    dy = pos_y - goal_y

    pos_error_sq = dx ** 2 + dy ** 2
    theta_goal = cs.atan2(goal_y - pos_y, goal_x - pos_x)
    angle_error = cs.fmod(psi - theta_goal + cs.pi, 2 * cs.pi) - cs.pi

    # If at terminal stage, apply stronger penalty
    if stage_idx == self.solver.horizon - 1:
      pos_cost = terminal_goal_weight * pos_error_sq
      angle_cost = terminal_angle_weight * angle_error ** 2
    else:
      pos_cost = goal_weight * pos_error_sq
      angle_cost = angle_weight * angle_error ** 2

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