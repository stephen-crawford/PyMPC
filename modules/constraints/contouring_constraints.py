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
		# Solver will be set by framework later
		self.solver = None
		# use robot discs if provided in Data
		self.num_discs = int(self.get_config_value("num_discs", 1))
		# Get slack parameter (adaptive slack increases with horizon)
		self.slack = float(self.get_config_value("contouring.slack", 1.0))
		# Number of spline segments to use per boundary at each step (default 5)
		# NOTE: For symbolic spline evaluation, we compute constraints along the entire path
		# to ensure coverage regardless of predicted position
		self.num_segments_per_boundary = int(self.get_config_value("contouring.num_segments_per_boundary", 5))
		# Store reference path data for dynamic constraint computation
		self._reference_path = None
		self._road_width_half = None
		self._width_left_spline = None  # Spline for left width (distance from centerline to left boundary)
		self._width_right_spline = None  # Spline for right width (distance from centerline to right boundary)
		LOG_DEBUG(f"ContouringConstraints initialized with {self.num_segments_per_boundary} segments per boundary")

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
		# CRITICAL: Always use the path from data, don't cache it, so it stays in sync with updates
		if hasattr(data, 'reference_path') and data.reference_path is not None:
			self._reference_path = data.reference_path  # Store reference, but always check data.reference_path for latest
			
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
					# Compute signed distances from centerline to boundaries along the normal vector
					# This is the correct way to compute widths for contouring constraints
					for i in range(len(ref_path.x)):
						center = np.array([float(ref_path.x[i]), float(ref_path.y[i])])
						left_point = np.array([float(left_x[i]), float(left_y[i])])
						right_point = np.array([float(right_x[i]), float(right_y[i])])
						
						# Get path tangent at this point (for computing normal)
						# Use spline derivative if available, otherwise estimate from neighbors
						try:
							s_val = float(s_arr[i])
							path_dx = float(ref_path.x_spline.derivative()(s_val))
							path_dy = float(ref_path.y_spline.derivative()(s_val))
						except Exception:
							# Fallback: estimate tangent from neighbors
							if i > 0 and i < len(ref_path.x) - 1:
								path_dx = float(ref_path.x[i+1] - ref_path.x[i-1])
								path_dy = float(ref_path.y[i+1] - ref_path.y[i-1])
							elif i > 0:
								path_dx = float(ref_path.x[i] - ref_path.x[i-1])
								path_dy = float(ref_path.y[i] - ref_path.y[i-1])
							else:
								path_dx = float(ref_path.x[i+1] - ref_path.x[i])
								path_dy = float(ref_path.y[i+1] - ref_path.y[i])
						
						# Normalize tangent
						norm = safe_norm(path_dx, path_dy)
						if norm < 1e-6:
							# Fallback to Euclidean distance if tangent is invalid
							width_left = np.linalg.norm(center - left_point)
							width_right = np.linalg.norm(center - right_point)
						else:
							path_dx_norm = path_dx / norm
							path_dy_norm = path_dy / norm
							
							# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
							normal = np.array([path_dy_norm, -path_dx_norm])
							
							# Compute signed distances along normal
							# Positive = left of path, negative = right of path
							left_vec = left_point - center
							right_vec = right_point - center
							
							# Signed distance = dot product with normal
							width_left = np.dot(normal, left_vec)  # Should be positive (left is in +normal direction)
							width_right = -np.dot(normal, right_vec)  # Should be positive (right is in -normal direction)
							
							# Ensure widths are positive (take absolute value if needed)
							width_left = abs(width_left)
							width_right = abs(width_right)
						
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
		
		# Strict boundary constraints: vehicle must stay within road bounds
		# Right boundary constraint: contour_error <= width_right - w_cur
		# A·p <= A·path_point + width_right - w_cur
		# This enforces: A·(p - path_point) <= width_right - w_cur
		# Since A points left, this limits how far left the vehicle can go (right boundary)
		b_right = np.dot(A, path_point) + width_right - w_cur_estimate
		
		# Left boundary constraint: -contour_error <= width_left - w_cur
		# -A·p <= -A·path_point + width_left - w_cur
		# This enforces: -A·(p - path_point) <= width_left - w_cur
		# Since -A points right, this limits how far right the vehicle can go (left boundary)
		b_left = np.dot(-A, path_point) + width_left - w_cur_estimate
		
		# Note: Slack is removed to strictly enforce road boundaries
		# If slack is needed for feasibility, it should be handled separately
		
		# Return constraints as (A, b, is_left) tuples
		constraints = [
			(A, b_right, False),  # Right boundary (is_left=False)
			(-A, b_left, True),   # Left boundary (is_left=True)
		]
		
		return constraints

	def calculate_constraints(self, state, data, stage_idx):
		"""Return linear halfspace constraints computed dynamically based on predicted spline value.
		
		CRITICAL FIX: Constraints must be computed using the SYMBOLIC spline value, not a numeric estimate.
		In the reference codebase (mpc_planner), constraints are computed symbolically based on the
		predicted spline value at each stage. Using a numeric estimate causes constraints to be computed
		for the wrong path location, allowing the vehicle to exit road boundaries.
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
		"""
		LOG_INFO(f"ContouringConstraints.calculate_constraints: stage_idx={stage_idx}")
		
		if self._reference_path is None:
			LOG_INFO(f"  ContouringConstraints stage {stage_idx}: No reference path stored, skipping constraints")
			return []
		
		# Get predicted spline value for this stage (analogous to C++ _solver->getOutput(k, "spline"))
		spline_val = None
		if state is not None:
			if hasattr(state, 'has') and state.has('spline'):
				try:
					spline_val = state.get('spline')
					LOG_DEBUG(f"  Stage {stage_idx}: Got spline from state: type={type(spline_val).__name__}, value={spline_val if not hasattr(spline_val, '__class__') or 'MX' not in str(type(spline_val)) else 'symbolic'}")
				except Exception as e:
					LOG_DEBUG(f"  Could not get spline from state: {e}")
			else:
				LOG_DEBUG(f"  Stage {stage_idx}: State does not have 'spline' variable (has method: {hasattr(state, 'has')})")
		else:
			LOG_DEBUG(f"  Stage {stage_idx}: State is None")
		
		# CRITICAL: Check if spline is symbolic
		import casadi as cd
		is_symbolic = isinstance(spline_val, (cd.MX, cd.SX))
		if is_symbolic:
			LOG_INFO(f"  Stage {stage_idx}: Spline is SYMBOLIC (CasADi {type(spline_val).__name__})")
		elif spline_val is not None:
			LOG_INFO(f"  Stage {stage_idx}: Spline is NUMERIC: {float(spline_val):.3f}")
		else:
			LOG_INFO(f"  Stage {stage_idx}: Spline is None - will try to get from warmstart")
		
		if is_symbolic:
			# SYMBOLIC spline: Compute constraints symbolically using the symbolic spline value
			# This is the correct approach - constraints will be evaluated at the actual predicted spline value
			LOG_INFO(f"  Stage {stage_idx}: Computing constraints SYMBOLICALLY using symbolic spline value")
			return self._compute_symbolic_constraints(spline_val, state, data, stage_idx)
		else:
			# LEGACY: Numeric spline should not happen in normal operation
			# Try to get symbolic spline from solver's var_dict
			LOG_WARN(f"  Stage {stage_idx}: Spline is not symbolic - attempting to get from solver var_dict")
			if hasattr(self, 'solver') and self.solver is not None:
				try:
					dynamics_model = self.solver._get_dynamics_model() if hasattr(self.solver, '_get_dynamics_model') else None
					if dynamics_model is not None and hasattr(self.solver, 'var_dict'):
						if 'spline' in self.solver.var_dict:
							if stage_idx < self.solver.var_dict['spline'].shape[0]:
								spline_sym = self.solver.var_dict['spline'][stage_idx]
								# Also update state with symbolic spline for consistency
								if state is not None:
									state.set('spline', spline_sym)
								LOG_INFO(f"  Stage {stage_idx}: Retrieved symbolic spline from solver var_dict, computing symbolically")
								return self._compute_symbolic_constraints(spline_sym, state, data, stage_idx)
							else:
								LOG_WARN(f"  Stage {stage_idx}: stage_idx {stage_idx} out of bounds for spline var_dict")
								return []
						else:
							LOG_WARN(f"  Stage {stage_idx}: 'spline' not in var_dict")
							return []
					else:
						LOG_WARN(f"  Stage {stage_idx}: Cannot get dynamics_model or var_dict")
						return []
				except Exception as e:
					LOG_WARN(f"  Stage {stage_idx}: Error retrieving symbolic spline: {e}")
					return []
			else:
				# Last resort: numeric fallback (should not happen in normal operation)
				LOG_WARN(f"  Stage {stage_idx}: No solver available, falling back to NUMERIC computation (legacy mode)")
				# Get numeric value for fallback
				cur_s = None
				if spline_val is not None:
					try:
						cur_s = float(spline_val)
					except Exception as e:
						LOG_DEBUG(f"  Error converting spline to float: {e}")
				
				# Fallback: get from warmstart or current state
				if cur_s is None:
					if hasattr(self, 'solver') and self.solver is not None:
						if hasattr(self.solver, 'warmstart_values') and 'spline' in self.solver.warmstart_values:
							if stage_idx < len(self.solver.warmstart_values['spline']):
								cur_s = float(self.solver.warmstart_values['spline'][stage_idx])
					
					if cur_s is None:
						s_arr = np.asarray(self._reference_path.s, dtype=float)
						if s_arr.size > 0:
							cur_s = float(s_arr[0])
							LOG_DEBUG(f"  Using path start s0={cur_s:.3f} as fallback")
						else:
							LOG_WARN(f"  Stage {stage_idx}: Cannot determine spline value, skipping constraints")
							return []
				
				LOG_WARN(f"  Stage {stage_idx}: Computing constraints NUMERICALLY for cur_s={cur_s:.3f} (legacy fallback)")
				return self._compute_numeric_constraints(cur_s, state, data, stage_idx)
	
	def _compute_symbolic_constraints(self, spline_sym, state, data, stage_idx):
		"""Compute constraints symbolically using the symbolic spline value.
		
		CRITICAL: This uses CasADi interpolants to evaluate path points and tangents symbolically,
		matching the approach used in ContouringObjective. This ensures constraints are evaluated
		at the actual predicted spline value, not a numeric estimate.
		
		Reference: https://github.com/tud-amr/mpc_planner - constraints are evaluated symbolically.
		"""
		import casadi as cd
		
		# Get path bounds
		s_arr = np.asarray(self._reference_path.s, dtype=float)
		s_min = float(s_arr[0])
		s_max = float(s_arr[-1])
		
		# Clamp symbolic spline to valid range
		spline_clamped = cd.fmax(s_min, cd.fmin(s_max, spline_sym))
		
		# CRITICAL: Create CasADi interpolants from scipy splines (same approach as ContouringObjective)
		# Sample path at discrete points for interpolation
		s_vals = s_arr if s_arr.size > 0 else np.linspace(s_min, s_max, 100)
		x_vals = np.array([float(self._reference_path.x_spline(si)) for si in s_vals])
		y_vals = np.array([float(self._reference_path.y_spline(si)) for si in s_vals])
		
		# Create CasADi interpolants for path position
		x_interp = cd.interpolant('x_interp', 'linear', [s_vals], x_vals)
		y_interp = cd.interpolant('y_interp', 'linear', [s_vals], y_vals)
		
		# Evaluate path point symbolically
		path_x_sym = x_interp(spline_clamped)
		path_y_sym = y_interp(spline_clamped)
		
		# Compute path tangent using finite differences (same as ContouringObjective)
		eps = 1e-3
		dx_sym = (x_interp(spline_clamped + eps) - x_interp(spline_clamped - eps)) / (2 * eps)
		dy_sym = (y_interp(spline_clamped + eps) - y_interp(spline_clamped - eps)) / (2 * eps)
		norm_sym = cd.sqrt(dx_sym*dx_sym + dy_sym*dy_sym)
		norm_sym = cd.fmax(norm_sym, 1e-6)
		path_dx_norm_sym = dx_sym / norm_sym
		path_dy_norm_sym = dy_sym / norm_sym
		
		# Normal vector pointing left: A = [path_dy_norm, -path_dx_norm]
		A_sym = cd.vertcat(path_dy_norm_sym, -path_dx_norm_sym)
		
		# Get robot parameters
		robot_radius = 0.5
		if hasattr(self, 'solver') and self.solver is not None:
			if hasattr(self.solver, 'data') and self.solver.data is not None:
				data = self.solver.data
				if hasattr(data, 'robot_area') and data.robot_area and len(data.robot_area) > 0:
					robot_radius = float(data.robot_area[0].radius)
		
		# Get road width (for now, use fixed width - can be made symbolic if width splines exist)
		width_right = self._road_width_half if self._road_width_half is not None else 3.5
		width_left = self._road_width_half if self._road_width_half is not None else 3.5
		w_cur_estimate = robot_radius
		
		# Get vehicle position symbolically
		pos_x_sym = state.get('x')
		pos_y_sym = state.get('y')
		psi_sym = state.get('psi')
		
		if pos_x_sym is None or pos_y_sym is None:
			# Try to get from solver's var_dict if available
			LOG_WARN(f"  Stage {stage_idx}: Cannot get symbolic position, attempting to get from solver var_dict")
			if hasattr(self, 'solver') and self.solver is not None:
				try:
					dynamics_model = self.solver._get_dynamics_model() if hasattr(self.solver, '_get_dynamics_model') else None
					if dynamics_model is not None and hasattr(self.solver, 'var_dict'):
						if 'x' in self.solver.var_dict and 'y' in self.solver.var_dict:
							if stage_idx < self.solver.var_dict['x'].shape[0] and stage_idx < self.solver.var_dict['y'].shape[0]:
								pos_x_sym = self.solver.var_dict['x'][stage_idx]
								pos_y_sym = self.solver.var_dict['y'][stage_idx]
								# Update state with symbolic values
								if state is not None:
									state.set('x', pos_x_sym)
									state.set('y', pos_y_sym)
								LOG_INFO(f"  Stage {stage_idx}: Retrieved symbolic position from solver var_dict")
								# Retry with updated state
								return self._compute_symbolic_constraints(spline_sym, state, data, stage_idx)
							else:
								LOG_WARN(f"  Stage {stage_idx}: stage_idx {stage_idx} out of bounds for var_dict")
								return []
						else:
							LOG_WARN(f"  Stage {stage_idx}: var_dict missing x or y")
							return []
					else:
						LOG_WARN(f"  Stage {stage_idx}: Cannot get dynamics_model or var_dict")
						return []
				except Exception as e:
					LOG_WARN(f"  Stage {stage_idx}: Error retrieving symbolic position: {e}")
					return []
			else:
				LOG_WARN(f"  Stage {stage_idx}: No solver available, cannot compute constraints")
				return []
		
		# Compute constraint b values symbolically
		# Right boundary: A·p <= A·path_point + width_right - w_cur
		# Left boundary: -A·p <= -A·path_point + width_left - w_cur
		# Where A = [path_dy_norm, -path_dx_norm] points LEFT
		# A·path_point = path_dy_norm * path_x - path_dx_norm * path_y
		path_point_dot_A = path_dy_norm_sym * path_x_sym - path_dx_norm_sym * path_y_sym
		b_right_sym = path_point_dot_A + width_right - w_cur_estimate
		# For left boundary: -A·p <= -A·path_point + width_left - w_cur
		# -A = [-path_dy_norm, path_dx_norm] points RIGHT
		# -A·path_point = -path_dy_norm * path_x + path_dx_norm * path_y
		b_left_sym = -path_point_dot_A + width_left - w_cur_estimate
		
		# Create constraints symbolically
		# Right boundary: A·[x, y] <= b_right
		# Left boundary: -A·[x, y] <= b_left
		constraint_right_expr = path_dy_norm_sym * pos_x_sym - path_dx_norm_sym * pos_y_sym - b_right_sym
		constraint_left_expr = -path_dy_norm_sym * pos_x_sym + path_dx_norm_sym * pos_y_sym - b_left_sym
		
		# Return constraints as CasADi expressions (solver will handle them)
		# For now, we need to return them in the dict format the solver expects
		# Extract numeric values for a1, a2, b from the symbolic expressions
		# Actually, we can't extract numeric values from symbolic - we need to return symbolic expressions
		# But the solver expects dict format with a1, a2, b...
		
		# CRITICAL: Return symbolic constraint expressions directly
		# The solver's _translate_constraint can handle CasADi expressions directly
		# This matches the reference codebase where constraints are symbolic
		
		# Get vehicle position symbolically (already checked above)
		pos_x_sym = state.get('x')
		pos_y_sym = state.get('y')
		
		# Get orientation for disc offset if needed
		psi_sym = state.get('psi')
		
		# Create symbolic constraint expressions
		constraints = []
		
		# Right boundary constraint: A·[x, y] <= b_right
		# Constraint expression: path_dy_norm_sym * pos_x_sym - path_dx_norm_sym * pos_y_sym - b_right_sym <= 0
		constraint_right_expr = path_dy_norm_sym * pos_x_sym - path_dx_norm_sym * pos_y_sym - b_right_sym
		
		# Left boundary constraint: -A·[x, y] <= b_left
		# Constraint expression: -path_dy_norm_sym * pos_x_sym + path_dx_norm_sym * pos_y_sym - b_left_sym <= 0
		constraint_left_expr = -path_dy_norm_sym * pos_x_sym + path_dx_norm_sym * pos_y_sym - b_left_sym
		
		# Apply disc offset if needed (for each disc)
		for disc_id in range(self.num_discs):
			disc_offset = 0.0
			if hasattr(data, "robot_area") and data.robot_area is not None and disc_id < len(data.robot_area):
				disc_offset = float(data.robot_area[disc_id].offset)
			
			if abs(disc_offset) > 1e-9 and psi_sym is not None:
				# Adjust constraints for disc offset: p_disc = p_robot + offset * [cos(psi), sin(psi)]
				# For right boundary: A·p_disc <= b_right
				#   path_dy_norm_sym * (x + offset*cos(psi)) - path_dx_norm_sym * (y + offset*sin(psi)) <= b_right
				#   = path_dy_norm_sym * x - path_dx_norm_sym * y + offset*(path_dy_norm_sym*cos(psi) - path_dx_norm_sym*sin(psi)) <= b_right
				offset_adjustment_right = disc_offset * (path_dy_norm_sym * cd.cos(psi_sym) - path_dx_norm_sym * cd.sin(psi_sym))
				constraint_right_expr_disc = constraint_right_expr - offset_adjustment_right
				
				# For left boundary: -A·p_disc <= b_left
				#   -path_dy_norm_sym * (x + offset*cos(psi)) + path_dx_norm_sym * (y + offset*sin(psi)) <= b_left
				#   = -path_dy_norm_sym * x + path_dx_norm_sym * y + offset*(-path_dy_norm_sym*cos(psi) + path_dx_norm_sym*sin(psi)) <= b_left
				offset_adjustment_left = disc_offset * (-path_dy_norm_sym * cd.cos(psi_sym) + path_dx_norm_sym * cd.sin(psi_sym))
				constraint_left_expr_disc = constraint_left_expr - offset_adjustment_left
			else:
				constraint_right_expr_disc = constraint_right_expr
				constraint_left_expr_disc = constraint_left_expr
			
			# Return as CasADi expressions directly (solver will handle them)
			# The solver's get_constraints will pair them with bounds
			# For now, return as dicts with a special marker to indicate they're symbolic expressions
			# The solver's _translate_constraint will detect CasADi expressions and handle them
			constraints.append({
				"type": "symbolic_expression",
				"expression": constraint_right_expr_disc,
				"ub": 0.0,  # expr <= 0
			})
			constraints.append({
				"type": "symbolic_expression",
				"expression": constraint_left_expr_disc,
				"ub": 0.0,  # expr <= 0
			})
		
		LOG_INFO(f"  Symbolic constraints: {len(constraints)} constraint expressions computed symbolically")
		return constraints
	
	def _compute_numeric_constraints(self, cur_s, state, data, stage_idx):
		"""Compute constraints numerically for a given spline value (for warmstart/initialization)."""
		s_arr = np.asarray(self._reference_path.s, dtype=float)
		s_min = float(s_arr[0])
		s_max = float(s_arr[-1])
		
		# Clamp cur_s to valid range
		cur_s = max(s_min, min(s_max, float(cur_s)))
		
		# Determine segment spacing
		segment_spacing = 1.0  # meters
		try:
			dx = float(self._reference_path.x_spline.derivative()(cur_s))
			dy = float(self._reference_path.y_spline.derivative()(cur_s))
			ds_per_meter = 1.0 / (np.sqrt(dx**2 + dy**2) + 1e-6)
			ds_segment = segment_spacing * ds_per_meter
		except:
			ds_segment = (s_max - s_min) * 0.1
		
		# Generate segment s values centered around cur_s
		segment_s_values = []
		if self.num_segments_per_boundary == 1:
			segment_s_values = [cur_s]
		else:
			total_span = (self.num_segments_per_boundary - 1) * ds_segment
			s_start = max(s_min, cur_s - total_span / 2)
			s_end = min(s_max, cur_s + total_span / 2)
			
			if s_start == s_min and s_end < s_max:
				s_end = min(s_max, s_start + total_span)
			elif s_end == s_max and s_start > s_min:
				s_start = max(s_min, s_end - total_span)
			
			if self.num_segments_per_boundary > 1:
				segment_s_values = np.linspace(s_start, s_end, self.num_segments_per_boundary)
			else:
				segment_s_values = [cur_s]
		
		segment_s_values = [max(s_min, min(s_max, float(s))) for s in segment_s_values]
		
		constraints = []
		halfspace_count = 0
		
		for segment_s in segment_s_values:
			constraint_tuples = self._compute_constraint_for_s(segment_s, stage_idx, state)
			
			if not constraint_tuples:
				continue
			
			for A, b, is_left in constraint_tuples:
				halfspace_count += 1
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
						"is_left": is_left,
						"spline_s": float(segment_s),
					})
		
		LOG_INFO(f"ContouringConstraints.calculate_constraints: stage_idx={stage_idx}, cur_s={cur_s:.3f}, "
		         f"segments={len(segment_s_values)}, returning {len(constraints)} constraint(s) "
		         f"({halfspace_count} halfspace(s) × {self.num_discs} disc(s))")
		
		if constraints:
			for i, const in enumerate(constraints[:3]):
				LOG_INFO(f"  Contouring constraint stage {stage_idx}, constraint {i}: a1={const.get('a1', 'N/A'):.6f}, "
				        f"a2={const.get('a2', 'N/A'):.6f}, b={const.get('b', 'N/A'):.6f}, "
				        f"disc_offset={const.get('disc_offset', 0.0):.4f}, is_left={const.get('is_left', 'N/A')}")
			if len(constraints) > 3:
				LOG_INFO(f"  ... and {len(constraints) - 3} more contouring constraints")
		else:
			LOG_INFO(f"  ContouringConstraints stage {stage_idx}: No constraints returned")
		
		return constraints

	def lower_bounds(self, state=None, data=None, stage_idx=None):
		"""Return lower bounds for constraints (all -inf for halfspace constraints A·p <= b)."""
		# For A·p <= b, expr = A·p - b ≤ 0 → lb = -inf per constraint
		count = 0
		if data is not None and stage_idx is not None:
			# Estimate constraint count: (2 boundaries × num_segments_per_boundary) × num_discs
			count = 2 * self.num_segments_per_boundary * self.num_discs
		return [-np.inf] * count

	def upper_bounds(self, state=None, data=None, stage_idx=None):
		"""Return upper bounds for constraints (all 0 for halfspace constraints A·p <= b)."""
		# Upper bound 0 per constraint for A·p <= b
		count = 0
		if data is not None and stage_idx is not None:
			# Estimate constraint count: (2 boundaries × num_segments_per_boundary) × num_discs
			count = 2 * self.num_segments_per_boundary * self.num_discs
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
