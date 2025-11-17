import casadi as cd

from modules.objectives.base_objective import BaseObjective
from utils.utils import LOG_INFO, LOG_DEBUG


class ControlJerkObjective(BaseObjective):
	def __init__(self):
		super().__init__()
		self.name = "control_jerk_objective"
		self.weight_accel_jerk = float(self.get_config_value("weights.acceleration_jerk_weight", 0.0))
		self.weight_steer_jerk = float(self.get_config_value("weights.angular_jerk_weight", 0.0))
		LOG_INFO(
			"ControlJerkObjective initialized with weights: "
			f"a_dot={self.weight_accel_jerk}, w_dot={self.weight_steer_jerk}"
		)

	def define_parameters(self, parameter_manager):
		if hasattr(parameter_manager, "add"):
			parameter_manager.add("acceleration_jerk_weight")
			parameter_manager.add("angular_jerk_weight")

	def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
		# Need solver var_dict to access neighboring control variables
		if not hasattr(self, "solver") or self.solver is None:
			return {}
		if not hasattr(self.solver, "var_dict") or not self.solver.var_dict:
			return {}

		var_dict = self.solver.var_dict
		timestep = self.get_timestep(
			data=self.solver.data if hasattr(self.solver, "data") else None,
			default=0.1,
		)
		inv_dt = 1.0 / max(timestep, 1e-6)
		cost_terms = {}

		if self.weight_accel_jerk > 0.0 and "a" in var_dict:
			a_curr = var_dict["a"][stage_idx]
			a_prev = var_dict["a"][stage_idx - 1] if stage_idx > 0 else var_dict["a"][0]
			a_diff = (a_curr - a_prev) * inv_dt
			cost_terms["acceleration_jerk_cost"] = self.weight_accel_jerk * cd.sqr(a_diff)

		if self.weight_steer_jerk > 0.0 and "w" in var_dict:
			w_curr = var_dict["w"][stage_idx]
			w_prev = var_dict["w"][stage_idx - 1] if stage_idx > 0 else var_dict["w"][0]
			w_diff = (w_curr - w_prev) * inv_dt
			cost_terms["angular_jerk_cost"] = self.weight_steer_jerk * cd.sqr(w_diff)

		LOG_DEBUG(
			f"ControlJerkObjective stage {stage_idx}: terms={list(cost_terms.keys())}"
		)
		return cost_terms

