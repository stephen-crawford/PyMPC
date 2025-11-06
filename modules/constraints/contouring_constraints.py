import numpy as np
import casadi as cd
from scipy.interpolate import CubicSpline

from modules.constraints.base_constraint import BaseConstraint
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN
from utils.math_tools import safe_norm, haar_difference_without_abs


class ContouringConstraints(BaseConstraint):
	def __init__(self):
		super().__init__()
		self.name = "contouring_constraints"
		# use robot discs if provided in Data
		self.num_discs = int(self.get_config_value("num_discs", 1))
		# Get slack parameter (adaptive slack increases with horizon)
		# Increased default slack to ensure feasibility - can be tuned down later
		self.slack = float(self.get_config_value("contouring.slack", 1.0))
		LOG_DEBUG("ContouringConstraints initialized")

	def update(self, state, data):
		# REQUIREMENT CHECK: Ensure reference path starts at current vehicle position
		if hasattr(data, 'reference_path') and data.reference_path is not None and state is not None:
			try:
				vehicle_pos = state.get_position()
				if vehicle_pos is not None and len(vehicle_pos) >= 2:
					ref_path_start = (float(data.reference_path.x[0]), float(data.reference_path.y[0]))
					vehicle_pos_tuple = (float(vehicle_pos[0]), float(vehicle_pos[1]))
					
					dist = np.sqrt((ref_path_start[0] - vehicle_pos_tuple[0])**2 + 
								 (ref_path_start[1] - vehicle_pos_tuple[1])**2)
					
					if dist > 0.5:
						LOG_WARN(f"ContouringConstraints: Reference path start ({ref_path_start[0]:.2f}, {ref_path_start[1]:.2f}) "
								f"is far from vehicle position ({vehicle_pos_tuple[0]:.2f}, {vehicle_pos_tuple[1]:.2f}), "
								f"distance: {dist:.3f}m. This may cause infeasibility.")
			except Exception as e:
				LOG_DEBUG(f"Could not verify reference path start position: {e}")
		
		# Create road constraints if they don't exist
		# ContouringObjective won't create them if ContouringConstraints exists, so we must create them here
		if hasattr(data, 'reference_path') and data.reference_path is not None:
			needs_constraints = False
			if not hasattr(data, 'static_obstacles') or data.static_obstacles is None:
				needs_constraints = True
			else:
				has_halfspaces = False
				for obs in data.static_obstacles:
					if obs is not None and hasattr(obs, 'halfspaces') and len(obs.halfspaces) > 0:
						has_halfspaces = True
						break
				if not has_halfspaces:
					needs_constraints = True
			
			if needs_constraints:
				LOG_INFO("ContouringConstraints: Creating road constraints from reference path")
				self._create_road_constraints_from_centerline(data, state)
		
		return
	
	def _create_road_constraints_from_centerline(self, data, state):
		"""Create road constraints based on reference path centerline and width.
		
		Uses contour error formulation similar to reference implementation:
		- contour_error = path_dy_norm * (x - path_x) - path_dx_norm * (y - path_y)
		- Right constraint: contour_error <= width_right - w_cur + slack
		- Left constraint: -contour_error <= width_left - w_cur + slack
		
		Converted to halfspace form: A·p <= b where:
		- A = [path_dy_norm, -path_dx_norm] (normal pointing left)
		- b = A·path_point + width_right - w_cur + slack (for right boundary)
		"""
		from planning.types import StaticObstacle
		
		# Get horizon using base class method
		horizon_val = self.get_horizon(data, default=10)
		
		LOG_DEBUG(f"ContouringConstraints: Creating {horizon_val + 1} road constraint obstacles")
		data.set("static_obstacles", [None] * (horizon_val + 1))
		
		# Get road width
		road_width_half = float(self.get_config_value("road.width", 7.0)) / 2.0
		
		# Get reference path
		ref_path = data.reference_path
		s_arr = np.asarray(ref_path.s, dtype=float)
		if s_arr.size < 2:
			LOG_WARN("ContouringConstraints: Reference path has insufficient points")
			return
		
		# Get robot parameters
		robot_radius = 0.5
		vehicle_width = 2.0  # Default vehicle width
		lr = 1.0  # Default rear axle distance
		if hasattr(data, 'robot_area') and data.robot_area and len(data.robot_area) > 0:
			robot_radius = float(data.robot_area[0].radius)
		# Try to get vehicle width from data if available
		if hasattr(data, 'vehicle_width') and data.vehicle_width is not None:
			try:
				vehicle_width = float(data.vehicle_width)
			except (TypeError, ValueError):
				pass
		if hasattr(data, 'lr') and data.lr is not None:
			try:
				lr = float(data.lr)
			except (TypeError, ValueError):
				pass
		
		# Determine starting arc length
		s0 = float(s_arr[0])
		s_end = float(s_arr[-1])
		step_s = (s_end - s0) / float(max(1, horizon_val))
		
		# Create constraints for each stage
		for k in range(horizon_val + 1):
			data.static_obstacles[k] = StaticObstacle()
			cur_s = min(s_end, s0 + k * step_s)
			
			# Get path point and derivatives
			path_point_x = ref_path.x_spline(cur_s)
			path_point_y = ref_path.y_spline(cur_s)
			path_point = np.array([float(path_point_x), float(path_point_y)])
			
			# Get path tangent and normalize
			path_dx = float(ref_path.x_spline.derivative()(cur_s))
			path_dy = float(ref_path.y_spline.derivative()(cur_s))
			
			norm = safe_norm(path_dx, path_dy)
			if norm < 1e-6:
				LOG_DEBUG(f"  Stage {k}: Skipping (norm too small: {norm})")
				continue
			path_dx_norm = path_dx / norm
			path_dy_norm = path_dy / norm
			
			# Normal vector pointing left: A = [-path_dy_norm, path_dx_norm]
			# Contour error = path_dy_norm * (x - path_x) - path_dx_norm * (y - path_y)
			# = [path_dy_norm, -path_dx_norm] · [x - path_x, y - path_y]
			# So A = [path_dy_norm, -path_dx_norm] points LEFT (positive contour_error = left of path)
			A = np.array([path_dy_norm, -path_dx_norm])  # Normal pointing left
			
			# Adaptive slack increases with horizon (like reference implementation)
			# Use larger slack to ensure feasibility - the reference uses this to handle uncertainty
			horizon_factor = 1.0 + (k * 0.2)  # Looser constraints further ahead (increased from 0.1)
			adaptive_slack = self.slack * horizon_factor
			
			# For stage 0, ensure constraints accommodate current vehicle position
			stage_road_width_half = road_width_half
			if k == 0 and state is not None:
				try:
					current_pos = state.get_position()
					if current_pos is not None and len(current_pos) >= 2:
						vehicle_pos = np.array([float(current_pos[0]), float(current_pos[1])])
						dist_to_path = np.linalg.norm(vehicle_pos - path_point)
						
						# Calculate effective vehicle width (worst case: perpendicular to path)
						# w_cur_max = vehicle_width/2 + lr (when delta_psi = 90 degrees)
						w_cur_max = vehicle_width / 2.0 + lr
						road_edge_dist = stage_road_width_half - w_cur_max
						
						if dist_to_path > road_edge_dist:
							required_edge_dist = dist_to_path + w_cur_max + 1.0  # 1.0m margin
							required_width_half = required_edge_dist
							LOG_INFO(f"  Stage 0: Vehicle at ({current_pos[0]:.2f}, {current_pos[1]:.2f}) is {dist_to_path:.2f}m from path")
							LOG_INFO(f"  Stage 0: Expanding road width from {stage_road_width_half * 2:.2f}m to {required_width_half * 2:.2f}m")
							stage_road_width_half = required_width_half
				except Exception as e:
					LOG_DEBUG(f"  Could not adjust road width for stage 0: {e}")
			
			# Calculate effective vehicle width for constraint
			# In the reference: w_cur = vehicle_width/2 * cos(delta_psi) + lr * sin(|delta_psi|)
			# For constraint creation, use a conservative estimate that accounts for vehicle orientation
			# Use robot_radius as a simpler approximation (accounts for disc size)
			# The w_cur term in reference accounts for vehicle width when oriented, but we use disc_offset in solver
			# So we just need to account for the disc radius here
			w_cur_estimate = robot_radius  # Use robot radius as conservative estimate
			
			# Reference implementation constraints (from GitHub):
			# c1 = contour_error + w_cur - width_right - slack <= 0  (right boundary - limits leftward deviation)
			# c2 = -contour_error + w_cur - width_left - slack <= 0  (left boundary - limits rightward deviation)
			# Where contour_error = A·(p - path_point) with A = [path_dy_norm, -path_dx_norm] pointing LEFT
			# 
			# Converting to halfspace form A·p <= b:
			# Right boundary (c1): contour_error <= width_right - w_cur + slack
			#                     A·(p - path_point) <= width_right - w_cur + slack
			#                     A·p <= A·path_point + width_right - w_cur + slack
			width_right = stage_road_width_half
			b_right = np.dot(A, path_point) + width_right - w_cur_estimate + adaptive_slack
			data.static_obstacles[k].add_halfspace(A, b_right)
			
			# Left boundary (c2): -contour_error <= width_left - w_cur + slack
			#                    -A·(p - path_point) <= width_left - w_cur + slack
			#                    A·(p - path_point) >= -width_left + w_cur - slack
			#                    A·p >= A·path_point - width_left + w_cur - slack
			#                    -A·p <= -A·path_point + width_left - w_cur + slack
			width_left = stage_road_width_half
			b_left = np.dot(-A, path_point) + width_left - w_cur_estimate + adaptive_slack
			data.static_obstacles[k].add_halfspace(-A, b_left)
			
			if k <= 2:
				LOG_DEBUG(f"  Stage {k}: Created 2 halfspaces, path_point=({path_point[0]:.2f}, {path_point[1]:.2f}), "
				         f"width_half={stage_road_width_half:.2f}, w_cur={w_cur_estimate:.2f}, slack={adaptive_slack:.3f}")
		
		LOG_INFO(f"ContouringConstraints: Created road constraints for {horizon_val + 1} stages")

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
		
		if stage_idx <= 2:
			LOG_INFO(f"ContouringConstraints stage {stage_idx}: checking static_obstacles")
		
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
					LOG_WARN(f"  Stage {stage_idx}, disc_id {disc_id}: robot_area not available")
				
				constraints.append({
					"type": "linear",
					"a1": float(A[0]),
					"a2": float(A[1]),
					"b": float(b),
					"disc_offset": disc_offset,
				})
				
				if stage_idx <= 2 and disc_id == 0:
					LOG_DEBUG(f"  Stage {stage_idx}, disc_id {disc_id}: disc_offset={disc_offset:.3f}, disc_radius={disc_radius:.3f}")
					# Diagnose feasibility at current state
					try:
						if state is not None and hasattr(state, 'get'):
							x = float(state.get('x')) if state.has('x') else None
							y = float(state.get('y')) if state.has('y') else None
							psi = float(state.get('psi')) if state.has('psi') else 0.0
							if x is not None and y is not None:
								xd = x + disc_offset * np.cos(psi)
								yd = y + disc_offset * np.sin(psi)
								lhs = float(A[0] * xd + A[1] * yd)
								violation = lhs - float(b)
								LOG_INFO(f"    Halfspace[{halfspace_count}] A=({A[0]:.6f},{A[1]:.6f}) b={float(b):.6f} at p_disc=({xd:.3f},{yd:.3f}) ⇒ A·p−b={violation:.6f}")
					except Exception as _e:
						LOG_DEBUG(f"    Feasibility diagnostic failed: {_e}")
		
		LOG_DEBUG(f"ContouringConstraints.calculate_constraints: Returning {len(constraints)} constraint(s) ({halfspace_count} halfspace(s) × {self.num_discs} disc(s))")
		
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
				width = float(self.module.get_config_value("road.width", 7.0))
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
