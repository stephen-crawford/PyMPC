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
		self.slack = float(self.get_config_value("contouring.slack", 1.0))
		# Store reference path data for dynamic constraint computation
		self._reference_path = None
		self._road_width_half = None
		self._width_left_spline = None  # Spline for left width (distance from centerline to left boundary)
		self._width_right_spline = None  # Spline for right width (distance from centerline to right boundary)
		LOG_DEBUG("ContouringConstraints initialized")

	def update(self, state, data):
		"""Prepare reference path data for dynamic constraint computation.
		
		Analogous to C++ onDataReceived: stores reference path and width information.
		Constraints are computed dynamically in calculate_constraints based on predicted spline values.
		"""
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
		
		# Store reference path data for dynamic constraint computation (analogous to C++ onDataReceived)
		if hasattr(data, 'reference_path') and data.reference_path is not None:
			self._reference_path = data.reference_path
			
			# Compute width splines from actual road boundaries (analogous to C++ onDataReceived)
			# C++ computes: widths_left[i] = distance(center, left), widths_right[i] = distance(center, right)
			# Then creates splines: _width_left->set_points(s, widths_left), _width_right->set_points(s, widths_right)
			self._width_left_spline = None
			self._width_right_spline = None
			
			# Check if we have left and right boundaries
			has_boundaries = False
			left_bound = None
			right_bound = None
			
			# Try to get boundaries from data object
			if hasattr(data, 'left_bound') and data.left_bound is not None:
				left_bound = data.left_bound
				has_boundaries = True
			if hasattr(data, 'right_bound') and data.right_bound is not None:
				right_bound = data.right_bound
				has_boundaries = True
			
			if has_boundaries and left_bound is not None and right_bound is not None:
				# Compute widths as distances from centerline to boundaries
				ref_path = data.reference_path
				s_arr = np.asarray(ref_path.s, dtype=float)
				
				widths_left = []
				widths_right = []
				
				# Get boundary points (as arrays or from Bound objects)
				left_x = np.asarray(left_bound.x, dtype=float) if hasattr(left_bound, 'x') else None
				left_y = np.asarray(left_bound.y, dtype=float) if hasattr(left_bound, 'y') else None
				right_x = np.asarray(right_bound.x, dtype=float) if hasattr(right_bound, 'x') else None
				right_y = np.asarray(right_bound.y, dtype=float) if hasattr(right_bound, 'y') else None
				
				# Also check for array-based boundaries
				if left_x is None and hasattr(data, 'left_boundary_x') and hasattr(data, 'left_boundary_y'):
					left_x = np.asarray(data.left_boundary_x, dtype=float)
					left_y = np.asarray(data.left_boundary_y, dtype=float)
				if right_x is None and hasattr(data, 'right_boundary_x') and hasattr(data, 'right_boundary_y'):
					right_x = np.asarray(data.right_boundary_x, dtype=float)
					right_y = np.asarray(data.right_boundary_y, dtype=float)
				
				if (left_x is not None and left_y is not None and 
					right_x is not None and right_y is not None and
					len(left_x) == len(ref_path.x) and len(right_x) == len(ref_path.x)):
					# Compute distances from centerline to boundaries
					for i in range(len(ref_path.x)):
						center = np.array([float(ref_path.x[i]), float(ref_path.y[i])])
						left_point = np.array([float(left_x[i]), float(left_y[i])])
						right_point = np.array([float(right_x[i]), float(right_y[i])])
						
						width_left = np.linalg.norm(center - left_point)
						width_right = np.linalg.norm(center - right_point)
						
						widths_left.append(width_left)
						widths_right.append(width_right)
					
					# Create splines for widths (analogous to C++ _width_left->set_points(s, widths_left))
					if len(widths_left) > 0 and len(widths_right) > 0:
						self._width_left_spline = CubicSpline(s_arr, np.array(widths_left))
						self._width_right_spline = CubicSpline(s_arr, np.array(widths_right))
						LOG_DEBUG(f"ContouringConstraints: Created width splines from road boundaries ({len(widths_left)} points)")
			
			# Fallback to fixed width if boundaries not available
			if self._width_left_spline is None or self._width_right_spline is None:
				self._road_width_half = float(self.get_config_value("road.width", 7.0)) / 2.0
				LOG_DEBUG(f"ContouringConstraints: Using fixed road width half={self._road_width_half:.3f}m (boundaries not available)")
			else:
				LOG_DEBUG("ContouringConstraints: Stored reference path data with width splines for dynamic constraint computation")
		
		return
	
	def _compute_constraint_for_s(self, cur_s, stage_idx, state=None):
		"""Compute road boundary constraints for a given arc length s.
		
		Analogous to C++ setParameters: computes constraints based on the predicted spline value.
		This is called dynamically in calculate_constraints for each stage.
		"""
		if self._reference_path is None:
			return []
		
		ref_path = self._reference_path
		s_arr = np.asarray(ref_path.s, dtype=float)
		if s_arr.size < 2:
			return []
		
		s0 = float(s_arr[0])
		s_end = float(s_arr[-1])
		
		# Clamp s to valid range
		cur_s = max(s0, min(s_end, float(cur_s)))
		
		# Get path point and derivatives (analogous to C++ module_data.path->getPoint(cur_s) and getOrthogonal(cur_s))
		try:
			path_point_x = ref_path.x_spline(cur_s)
			path_point_y = ref_path.y_spline(cur_s)
			path_point = np.array([float(path_point_x), float(path_point_y)])
			
			# Get path tangent and normalize
			path_dx = float(ref_path.x_spline.derivative()(cur_s))
			path_dy = float(ref_path.y_spline.derivative()(cur_s))
			
			norm = safe_norm(path_dx, path_dy)
			if norm < 1e-6:
				return []
			path_dx_norm = path_dx / norm
			path_dy_norm = path_dy / norm
		except Exception as e:
			LOG_DEBUG(f"  Stage {stage_idx}: Failed to evaluate path at s={cur_s:.3f}: {e}")
			return []
		
		# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
		# Contour error = path_dy_norm * (x - path_x) - path_dx_norm * (y - path_y)
		# = [path_dy_norm, -path_dx_norm] · [x - path_x, y - path_y]
		# So A = [path_dy_norm, -path_dx_norm] points LEFT (positive contour_error = left of path)
		A = np.array([path_dy_norm, -path_dx_norm])  # Normal pointing left
		
		# Get robot parameters
		robot_radius = 0.5
		vehicle_width = 2.0
		lr = 1.0
		if hasattr(self, 'solver') and self.solver is not None:
			if hasattr(self.solver, 'data') and self.solver.data is not None:
				data = self.solver.data
				if hasattr(data, 'robot_area') and data.robot_area and len(data.robot_area) > 0:
					robot_radius = float(data.robot_area[0].radius)
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
		
		# Adaptive slack increases with horizon (like reference implementation)
		horizon_factor = 1.0 + (stage_idx * 0.2)
		adaptive_slack = self.slack * horizon_factor
		
		# Calculate effective vehicle width for constraint (w_cur in C++ code)
		w_cur_estimate = robot_radius
		
		# Compute width_right and width_left (analogous to C++ _width_right->operator()(cur_s) and _width_left->operator()(cur_s))
		if self._width_right_spline is not None and self._width_left_spline is not None:
			# Use width splines computed from actual road boundaries
			try:
				width_right = float(self._width_right_spline(cur_s))
				width_left = float(self._width_left_spline(cur_s))
			except Exception as e:
				LOG_DEBUG(f"  Stage {stage_idx}: Failed to evaluate width splines at s={cur_s:.3f}: {e}, using fallback")
				width_right = self._road_width_half if self._road_width_half is not None else 3.5
				width_left = self._road_width_half if self._road_width_half is not None else 3.5
		else:
			# Fallback to fixed width if splines not available
			width_right = self._road_width_half if self._road_width_half is not None else 3.5
			width_left = self._road_width_half if self._road_width_half is not None else 3.5
		
		# Right boundary constraint: contour_error <= width_right - w_cur + slack
		# A·p <= A·path_point + width_right - w_cur + slack
		b_right = np.dot(A, path_point) + width_right - w_cur_estimate + adaptive_slack
		
		# Left boundary constraint: -contour_error <= width_left - w_cur + slack
		# -A·p <= -A·path_point + width_left - w_cur + slack
		b_left = np.dot(-A, path_point) + width_left - w_cur_estimate + adaptive_slack
		
		# Return constraints as (A, b, is_left) tuples
		constraints = [
			(A, b_right, False),  # Right boundary (is_left=False)
			(-A, b_left, True),   # Left boundary (is_left=True)
		]
		
		return constraints

	def calculate_constraints(self, state, data, stage_idx):
		"""Return linear halfspace constraints computed dynamically based on predicted spline value.
		
		Analogous to C++ setParameters + solver constraint evaluation:
		- Gets predicted spline value from state (may be symbolic)
		- Computes constraints for that spline value
		- Returns constraints for all discs
		"""
		LOG_DEBUG(f"ContouringConstraints.calculate_constraints: stage_idx={stage_idx}")
		
		if self._reference_path is None:
			LOG_DEBUG("  No reference path stored, skipping constraints")
			return []
		
		# Get predicted spline value for this stage (analogous to C++ _solver->getOutput(k, "spline"))
		spline_val = None
		if state is not None and hasattr(state, 'get') and state.has('spline'):
			try:
				spline_val = state.get('spline')
			except Exception as e:
				LOG_DEBUG(f"  Could not get spline from state: {e}")
		
		# If spline is symbolic, we need to handle it differently
		# For now, try to get a numeric estimate from warmstart or current state
		cur_s = None
		is_symbolic = False
		
		if spline_val is not None:
			try:
				import casadi as cd
				if isinstance(spline_val, (cd.MX, cd.SX)):
					# Symbolic spline - try to get numeric estimate from warmstart
					is_symbolic = True
					if hasattr(self, 'solver') and self.solver is not None:
						if hasattr(self.solver, 'warmstart') and self.solver.warmstart is not None:
							try:
								# Try to get warmstart value for spline at this stage
								warmstart = self.solver.warmstart
								if hasattr(warmstart, 'get') and warmstart.has('spline'):
									cur_s = float(warmstart.get('spline'))
								elif isinstance(warmstart, dict) and 'spline' in warmstart:
									if isinstance(warmstart['spline'], (list, np.ndarray)):
										if stage_idx < len(warmstart['spline']):
											cur_s = float(warmstart['spline'][stage_idx])
									else:
										cur_s = float(warmstart['spline'])
							except Exception as e:
								LOG_DEBUG(f"  Could not get warmstart spline: {e}")
					
					# Fallback: estimate from current state if available
					if cur_s is None and stage_idx == 0:
						if hasattr(self, 'solver') and self.solver is not None:
							if hasattr(self.solver, 'data') and self.solver.data is not None:
								data = self.solver.data
								if hasattr(data, 'state') and data.state is not None:
									try:
										if data.state.has('spline'):
											cur_s = float(data.state.get('spline'))
									except Exception:
										pass
					
					if cur_s is None:
						# Last resort: use path start
						s_arr = np.asarray(self._reference_path.s, dtype=float)
						if s_arr.size > 0:
							cur_s = float(s_arr[0])
							LOG_DEBUG(f"  Using path start s0={cur_s:.3f} as fallback for symbolic spline")
				else:
					# Numeric spline value
					cur_s = float(spline_val)
			except Exception as e:
				LOG_DEBUG(f"  Error processing spline value: {e}")
		
		# Fallback if we still don't have cur_s
		if cur_s is None:
			s_arr = np.asarray(self._reference_path.s, dtype=float)
			if s_arr.size > 0:
				cur_s = float(s_arr[0])
				LOG_DEBUG(f"  Using path start s0={cur_s:.3f} as final fallback")
			else:
				LOG_WARN(f"  Stage {stage_idx}: Cannot determine spline value, skipping constraints")
				return []
		
		# Compute constraints for this spline value (analogous to C++ evaluating width_left/width_right at cur_s)
		constraint_tuples = self._compute_constraint_for_s(cur_s, stage_idx, state)
		
		if not constraint_tuples:
			return []
		
		# Convert to structured constraint format for solver
		constraints = []
		halfspace_count = 0
		
		for A, b, is_left in constraint_tuples:
			halfspace_count += 1
			# Apply constraint for each disc (solver handles disc_offset)
			for disc_id in range(self.num_discs):
				disc_offset = 0.0
				disc_radius = 0.5
				if hasattr(data, "robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
					disc_offset = float(data.robot_area[disc_id].offset)
					disc_radius = float(data.robot_area[disc_id].radius)
				
				constraints.append({
					"type": "linear",
					"a1": float(A[0]),
					"a2": float(A[1]),
					"b": float(b),
					"disc_offset": disc_offset,
					"is_left": is_left,  # Store is_left flag for visualization
				})
		
		LOG_DEBUG(f"ContouringConstraints.calculate_constraints: stage_idx={stage_idx}, cur_s={cur_s:.3f}, "
		         f"returning {len(constraints)} constraint(s) ({halfspace_count} halfspace(s) × {self.num_discs} disc(s))")
		
		if stage_idx <= 2 and constraints:
			first_const = constraints[0]
			LOG_INFO(f"  Stage {stage_idx}: cur_s={cur_s:.3f}, first constraint: a1={first_const.get('a1', 'N/A'):.6f}, "
			        f"a2={first_const.get('a2', 'N/A'):.6f}, b={first_const.get('b', 'N/A'):.6f}")
		
		return constraints

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		"""Return lower bounds for constraints (all -inf for halfspace constraints A·p <= b)."""
		# For A·p <= b, expr = A·p - b ≤ 0 → lb = -inf per constraint
		count = 0
		if data is not None and stage_idx is not None:
			# Estimate constraint count (2 halfspaces × num_discs)
			count = 2 * self.num_discs
		return [-np.inf] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		"""Return upper bounds for constraints (all 0 for halfspace constraints A·p <= b)."""
		# Upper bound 0 per constraint for A·p <= b
		count = 0
		if data is not None and stage_idx is not None:
			# Estimate constraint count (2 halfspaces × num_discs)
			count = 2 * self.num_discs
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
