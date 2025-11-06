import numpy as np
import casadi as cs
from modules.objectives.base_objective import BaseObjective
from planning.types import State
from utils.const import OBJECTIVE
from utils.math_tools import distance
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO

class GoalObjective(BaseObjective):

  def __init__(self):
    super().__init__()
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

  def get_value(self, state, data, stage_idx):

    pos_x = state.get("x")
    pos_y = state.get("y")
    psi = state.get("psi")

    # Get parameters from data.parameters
    goal_x = None
    goal_y = None
    if data and hasattr(data, 'parameters') and data.parameters is not None:
        goal_x = data.parameters.get("goal_x")
        goal_y = data.parameters.get("goal_y")

    # If goal parameters not set, try to get from data.goal directly
    if goal_x is None or goal_y is None:
        if data and hasattr(data, 'goal') and data.goal is not None:
            goal_x = float(data.goal[0]) if len(data.goal) > 0 else None
            goal_y = float(data.goal[1]) if len(data.goal) > 1 else None

    # If still None, return zero cost (can't compute goal error without goal)
    if goal_x is None or goal_y is None:
        LOG_WARN(f"GoalObjective: goal_x={goal_x}, goal_y={goal_y} - returning zero cost")
        return {"goal_pos_cost": 0.0, "goal_angle_cost": 0.0}

    # Get weights with safe defaults
    if data and hasattr(data, 'parameters') and data.parameters is not None:
        goal_weight = data.parameters.get("goal_weight", 1.0)
        angle_weight = data.parameters.get("goal_angle_weight", 10.0)
        terminal_goal_weight = data.parameters.get("goal_terminal_weight", 10.0)
        terminal_angle_weight = data.parameters.get("goal_terminal_angle_weight", 1.0)
    else:
        goal_weight = 1.0
        angle_weight = 10.0
        terminal_goal_weight = 10.0
        terminal_angle_weight = 1.0

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
    # Get horizon from data or use default
    horizon_val = 10
    if hasattr(self, 'solver') and hasattr(self.solver, 'horizon') and self.solver.horizon is not None:
      horizon_val = self.solver.horizon
    elif data and hasattr(data, 'horizon') and data.horizon is not None:
      horizon_val = data.horizon
    if stage_idx == horizon_val - 1:
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
    # Set all weight parameters from config for proper convergence
    parameter_manager.set_parameter("goal_angle_weight", 
                                     self.get_config_value("weights.goal_angle_weight", 10.0))
    parameter_manager.set_parameter("goal_terminal_weight", 
                                     self.get_config_value("weights.goal_terminal_weight", 10.0))
    parameter_manager.set_parameter("goal_terminal_angle_weight", 
                                     self.get_config_value("weights.goal_terminal_angle", 1.0))
    parameter_manager.set_parameter("goal_control_weight", 
                                     self.get_config_value("weights.goal_control_weight", 1.0))

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