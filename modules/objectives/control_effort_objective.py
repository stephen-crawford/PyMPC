import casadi as cd

from modules.objectives.base_objective import BaseObjective
from utils.utils import LOG_INFO, LOG_DEBUG


class ControlEffortObjective(BaseObjective):
	def __init__(self):
		super().__init__()
		self.name = "control_effort_objective"
		self._weight_accel = float(self.get_config_value("weights.acceleration_weight", 0.0))
		self._weight_steer = float(self.get_config_value("weights.angular_velocity", 0.0))
		self._weight_slack = float(self.get_config_value("weights.slack_weight", 0.0))
		LOG_INFO(
			f"ControlEffortObjective initialized with weights: "
			f"a={self._weight_accel}, w={self._weight_steer}, slack={self._weight_slack}"
		)

	def define_parameters(self, parameter_manager):
		if hasattr(parameter_manager, "add"):
			parameter_manager.add("control_weight_acceleration")
			parameter_manager.add("control_weight_angular_velocity")
			parameter_manager.add("control_weight_slack")

	def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
		LOG_DEBUG(f"ControlEffortObjective.get_stage_cost_symbolic(stage={stage_idx})")
		cost_terms = {}

		if self._weight_accel > 0.0 and symbolic_state.has("a"):
			a = symbolic_state.get("a")
			cost_terms["control_acceleration_cost"] = self._weight_accel * cd.sqr(a)

		if self._weight_steer > 0.0 and symbolic_state.has("w"):
			w = symbolic_state.get("w")
			cost_terms["control_angular_velocity_cost"] = self._weight_steer * cd.sqr(w)

		if self._weight_slack > 0.0 and symbolic_state.has("slack"):
			slack = symbolic_state.get("slack")
			cost_terms["control_slack_cost"] = self._weight_slack * cd.sqr(slack)

		return cost_terms

