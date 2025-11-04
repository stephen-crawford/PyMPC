import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN


class ContouringConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = "contouring_constraints"
		# use robot discs if provided in Data
		self.num_discs = int(self.get_config_value("num_discs", 1))
		LOG_DEBUG("ContouringConstraints initialized")

	def update(self, state, data):
		# REQUIREMENT CHECK: Ensure reference path starts at current vehicle position
		# This is critical for feasibility with contouring constraints
		# Note: Adjustment should only happen at initialization, this is just validation
		if hasattr(data, 'reference_path') and data.reference_path is not None and state is not None:
			try:
				vehicle_pos = state.get_position()
				if vehicle_pos is not None and len(vehicle_pos) >= 2:
					ref_path_start = (float(data.reference_path.x[0]), float(data.reference_path.y[0]))
					vehicle_pos_tuple = (float(vehicle_pos[0]), float(vehicle_pos[1]))
					
					dist = np.sqrt((ref_path_start[0] - vehicle_pos_tuple[0])**2 + 
								 (ref_path_start[1] - vehicle_pos_tuple[1])**2)
					
					# Only warn if significant difference (likely means path wasn't adjusted at initialization)
					if dist > 0.5:  # More than 50cm difference indicates a problem
						LOG_WARN(f"ContouringConstraints: Reference path start ({ref_path_start[0]:.2f}, {ref_path_start[1]:.2f}) "
								f"is far from vehicle position ({vehicle_pos_tuple[0]:.2f}, {vehicle_pos_tuple[1]:.2f}), "
								f"distance: {dist:.3f}m. This may cause infeasibility.")
			except Exception as e:
				LOG_DEBUG(f"Could not verify reference path start position: {e}")
		
		# Nothing to precompute here; road halfspaces are built by contouring objective into data.static_obstacles
		return

	def _iter_halfspaces(self, data, stage_idx):
		"""Yield (A, b) pairs for halfspaces at given stage from Data."""
		if not hasattr(data, "static_obstacles") or data.static_obstacles is None:
			return
		if stage_idx >= len(data.static_obstacles):
			return
		obstacle = data.static_obstacles[stage_idx]
		if obstacle is None or not hasattr(obstacle, "halfspaces"):
			return
		for hs in obstacle.halfspaces:
			# Expect hs.A as [a1, a2] and hs.b as scalar defining A·p <= b
			yield np.array(hs.A).flatten(), float(hs.b)

	def calculate_constraints(self, state, data, stage_idx):
		"""Return linear halfspace constraints A·(p_disc) <= b as dicts {a1,a2,b,disc_offset}.
		Solver reconstructs expr = a1*x_disc + a2*y_disc - (b - disc_offset) and applies ub=0.
		"""
		LOG_DEBUG(f"ContouringConstraints.calculate_constraints: stage_idx={stage_idx}")
		constraints = []
		has_static_obs = hasattr(data, "static_obstacles") and data.static_obstacles is not None
		obs_len = len(data.static_obstacles) if has_static_obs else 0
		LOG_DEBUG(f"  has_static_obstacles={has_static_obs}, len={obs_len}")
		
		# Log constraint details for first few stages
		if stage_idx <= 2:
			LOG_INFO(f"ContouringConstraints stage {stage_idx}: checking static_obstacles")
		try:
			import logging as _logging
			_logger = _logging.getLogger("integration_test")
			_logger.debug(f"ContouringConstraints.calculate_constraints stage={stage_idx} has_static_obstacles={has_static_obs} len={obs_len}")
		except Exception:
			pass
		halfspace_count = 0
		for A, b in self._iter_halfspaces(data, stage_idx) or []:
			halfspace_count += 1
			for disc_id in range(self.num_discs):
				disc_offset = 0.0
				disc_radius = 0.5  # Default
				if hasattr(data, "robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
					disc_offset = float(data.robot_area[disc_id].offset)
					disc_radius = float(data.robot_area[disc_id].radius)
				elif stage_idx <= 2:
					LOG_WARN(f"  Stage {stage_idx}, disc_id {disc_id}: robot_area not available or insufficient discs (need {self.num_discs}, have {len(data.robot_area) if hasattr(data, 'robot_area') and data.robot_area is not None else 0})")
				constraints.append({
					"type": "linear",
					"a1": float(A[0]),
					"a2": float(A[1]),
					"b": float(b),
					"disc_offset": disc_offset,
				})
				if stage_idx <= 2 and disc_id == 0:
					LOG_DEBUG(f"  Stage {stage_idx}, disc_id {disc_id}: disc_offset={disc_offset:.3f}, disc_radius={disc_radius:.3f}")
		LOG_DEBUG(f"ContouringConstraints.calculate_constraints: Returning {len(constraints)} constraint(s) ({halfspace_count} halfspace(s) × {self.num_discs} disc(s))")
		
		# Log constraint details for first few stages
		if stage_idx <= 2 and constraints:
			first_const = constraints[0]
			LOG_INFO(f"  First constraint: a1={first_const.get('a1', 'N/A')}, a2={first_const.get('a2', 'N/A')}, b={first_const.get('b', 'N/A')}")
		
		return constraints

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		# For A·p <= b, expr = A·p - b ≤ 0 → lb = -inf per constraint
		count = 0
		if data is not None and stage_idx is not None:
			for _ in self._iter_halfspaces(data, stage_idx) or []:
				count += self.num_discs
		return [-np.inf] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		# Upper bound 0 per constraint for A·p <= b
		count = 0
		if data is not None and stage_idx is not None:
			for _ in self._iter_halfspaces(data, stage_idx) or []:
				count += self.num_discs
		return [0.0] * count

	def get_visualizer(self):
		class RoadBoundsVisualizer:
			def __init__(self, module):
				self.module = module
			def visualize(self, state, data, stage_idx=0):
				try:
					import matplotlib.pyplot as plt
				except Exception:
					return
				# Draw road bounds from reference_path and configured width
				if not hasattr(data, 'reference_path') or data.reference_path is None:
					return
				ref = data.reference_path
				if not hasattr(ref, 'x_spline') or not hasattr(ref, 'y_spline'):
					return
				# Determine width
				width = float(self.module.get_config_value("planner.road.width", 7.0))
				half_w = 0.5 * width
				# Sample along arc length
				s_vals = ref.s if hasattr(ref, 's') else np.linspace(0.0, 1.0, 100)
				xc = []
				yc = []
				xl = []
				yl = []
				xr = []
				yr = []
				# If s is normalized, attempt to map to arc length
				if len(s_vals) > 0 and (np.max(s_vals) <= 1.001):
					s_min = float(s_vals[0])
					s_max = float(s_vals[-1])
					s_sample = np.linspace(s_min, s_max, 100)
				else:
					s_sample = np.linspace(float(s_vals[0]), float(s_vals[-1]), 100)
				for s in s_sample:
					x = ref.x_spline(s)
					y = ref.y_spline(s)
					dx = ref.x_spline.derivative()(s)
					dy = ref.y_spline.derivative()(s)
					norm = np.hypot(float(dx), float(dy))
					if norm < 1e-6:
						continue
					nx = -dy / norm
					ny = dx / norm
					xc.append(float(x))
					yc.append(float(y))
					xl.append(float(x) + half_w * float(nx))
					yl.append(float(y) + half_w * float(ny))
					xr.append(float(x) - half_w * float(nx))
					yr.append(float(y) - half_w * float(ny))
				ax = plt.gca()
				ax.plot(xl, yl, 'k--', linewidth=1.0, alpha=0.7)
				ax.plot(xr, yr, 'k--', linewidth=1.0, alpha=0.7)
		return RoadBoundsVisualizer(self)