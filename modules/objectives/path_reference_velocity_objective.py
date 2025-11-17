import casadi as cd
import numpy as np

from modules.objectives.base_objective import BaseObjective
from utils.utils import LOG_DEBUG, LOG_INFO


class PathReferenceVelocityObjective(BaseObjective):

	def __init__(self):
		super().__init__()
		self.name = "path_reference_velocity_objective"
		self.velocity_weight = float(self.get_config_value("weights.velocity_tracking_weight", 0.0))
		self.default_reference_velocity = float(self.get_config_value("weights.reference_velocity", 1.0))
		self.reference_path = None

	def define_parameters(self, parameter_manager):
		if hasattr(parameter_manager, "add"):
			parameter_manager.add("reference_velocity")
			parameter_manager.add("velocity_tracking_weight")

	def update(self, state, data):
		if data is not None and hasattr(data, "reference_path") and data.reference_path is not None:
			if self.reference_path is None:
				LOG_INFO("PathReferenceVelocityObjective: reference path received")
			self.reference_path = data.reference_path

	def set_parameters(self, parameter_manager, data, k):
		if self.velocity_weight <= 0.0:
			return
		v_ref = self._compute_reference_velocity(data, k)
		parameter_manager.set_parameter("reference_velocity", v_ref, stage_index=k)
		parameter_manager.set_parameter("velocity_tracking_weight", self.velocity_weight, stage_index=k)
		if k == 0:
			LOG_DEBUG(
				f"PathReferenceVelocityObjective: stage {k} reference velocity set to {v_ref:.3f} m/s (weight={self.velocity_weight})"
			)

	def get_stage_cost_symbolic(self, symbolic_state, stage_idx):
		if self.velocity_weight <= 0.0:
			return {}
		if not symbolic_state.has("v"):
			return {}

		v = symbolic_state.get("v")
		weight = self._get_stage_weight(stage_idx)
		if weight <= 0.0:
			return {}

		v_ref = self._get_stage_reference_velocity(stage_idx)
		cost = weight * cd.sqr(v - v_ref)
		return {"path_reference_velocity_cost": cost}

	def _get_stage_weight(self, stage_idx):
		if self.solver and hasattr(self.solver, "parameter_manager"):
			params = self.solver.parameter_manager.get_all(stage_idx)
			if "velocity_tracking_weight" in params:
				try:
					return float(params["velocity_tracking_weight"])
				except Exception:
					return self.velocity_weight
		return self.velocity_weight

	def _get_stage_reference_velocity(self, stage_idx):
		if self.solver and hasattr(self.solver, "parameter_manager"):
			params = self.solver.parameter_manager.get_all(stage_idx)
			ref_val = params.get("reference_velocity")
			if ref_val is not None:
				try:
					return float(ref_val)
				except Exception:
					pass
		return self.default_reference_velocity

	def _compute_reference_velocity(self, data, stage_idx):
		# Prefer explicit velocity samples along the reference path if available
		ref_path = getattr(data, "reference_path", None)
		if ref_path is None:
			ref_path = self.reference_path

		if ref_path is not None:
			if hasattr(ref_path, "v") and ref_path.v:
				idx = min(stage_idx, len(ref_path.v) - 1)
				try:
					return float(max(0.0, ref_path.v[idx]))
				except Exception:
					pass

			# Fallback: derive average speed from arc length over horizon duration
			if hasattr(ref_path, "s") and ref_path.s is not None and len(ref_path.s) > 1:
				try:
					s_arr = np.asarray(ref_path.s, dtype=float)
					total_length = float(s_arr[-1] - s_arr[0])
					horizon = self.get_horizon(data=data, default=10)
					dt = self.get_timestep(data=data, default=0.1)
					duration = horizon * dt
					if total_length > 1e-6 and duration > 1e-6:
						return max(0.2, total_length / duration)
				except Exception:
					pass

		return self.default_reference_velocity
