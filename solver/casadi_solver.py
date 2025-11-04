import copy

import casadi as cs
import numpy as np

from planning.dynamic_models import DynamicsModel
from planning.types import State, Trajectory
from solver.base_solver import BaseSolver
from utils.const import OBJECTIVE, CONSTRAINT, DEFAULT_BRAKING
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO

'''
Casadi solver used for trajectory optimization.
'''

class CasADiSolver(BaseSolver):
	def __init__(self, config):
		super().__init__(config)
		# Extract horizon and timestep from config
		planner_config = config.get("planner", {})
		self.horizon = planner_config.get("horizon", 10)
		self.timestep = planner_config.get("timestep", 0.1)
		# Initialize solver components
		self.opti = None
		self.var_dict = {}
		self.warmstart_values = {}
		self.dynamics_model = None
		self.solution = None
		self.exit_flag = None
		self.info = {}
		self.forecast = []
		LOG_INFO(f"{(id(self))} CasADiSolver: Initializing solver")
		

	def intialize_solver(self, data):
		"""Initialize solver variables based on dynamics model and horizon."""
		LOG_INFO("=== CasADiSolver.intialize_solver() ===")
		if data is None:
			LOG_WARN("Cannot initialize solver: data is None")
			return
		
		# Use self.horizon if data.horizon is not set
		horizon = getattr(data, 'horizon', None) or self.horizon
		dynamics = getattr(data, 'dynamics_model', None) or self.dynamics_model
		LOG_DEBUG(f"Initializing solver: horizon={horizon}, dynamics_model={dynamics.__class__.__name__ if dynamics else 'None'}")
		
		if dynamics is None:
			LOG_WARN("Cannot initialize solver: dynamics_model is None")
			return
		
		if self.opti is None:
			self.opti = cs.Opti()
			LOG_DEBUG("Created new CasADi Opti instance")
		
		LOG_DEBUG("Creating optimization variables...")
		dependent_vars = dynamics.get_dependent_vars()
		input_vars = dynamics.get_inputs()
		LOG_DEBUG(f"  Dependent variables ({len(dependent_vars)}): {dependent_vars}")
		LOG_DEBUG(f"  Input variables ({len(input_vars)}): {input_vars}")
		
		# Create state variables with bounds
		for var_name in dependent_vars:
			self.var_dict[var_name] = self.opti.variable(horizon + 1)
			self.warmstart_values[var_name] = np.zeros(horizon + 1)
			# Set bounds from dynamics model
			try:
				lb, ub, _ = dynamics.get_bounds(var_name)
				self.opti.subject_to(self.opti.bounded(lb, self.var_dict[var_name], ub))
				LOG_DEBUG(f"  Created variable '{var_name}': shape={horizon + 1}, bounds=[{lb}, {ub}]")
			except Exception as e:
				LOG_WARN(f"  Created variable '{var_name}': shape={horizon + 1}, no bounds (error: {e})")
		
		# Create input variables with bounds
		for var_name in input_vars:
			self.var_dict[var_name] = self.opti.variable(horizon)
			self.warmstart_values[var_name] = np.zeros(horizon)
			# Set bounds from dynamics model
			try:
				lb, ub, _ = dynamics.get_bounds(var_name)
				self.opti.subject_to(self.opti.bounded(lb, self.var_dict[var_name], ub))
				LOG_DEBUG(f"  Created input variable '{var_name}': shape={horizon}, bounds=[{lb}, {ub}]")
			except Exception as e:
				LOG_WARN(f"  Created input variable '{var_name}': shape={horizon}, no bounds (error: {e})")
		opts = {
			'ipopt.print_level': 0,
			'print_time': 0,
			'ipopt.sb': 'yes',
			'ipopt.max_iter': 2000,  # Increased for difficult problems
			'ipopt.tol': 1e-3,  # Slightly relaxed for faster convergence
			'ipopt.acceptable_tol': 1e-1,  # More relaxed to accept suboptimal solutions
			'ipopt.acceptable_iter': 10,  # Accept sooner if acceptable_tol met
			'ipopt.constr_viol_tol': 1e-3,  # More relaxed constraint violation tolerance
			'ipopt.mu_strategy': 'adaptive',
			'ipopt.hessian_approximation': 'limited-memory',
			'ipopt.warm_start_init_point': 'yes',
			'ipopt.nlp_scaling_method': 'gradient-based',  # Better scaling for constraints
			'ipopt.obj_scaling_factor': 1.0,  # No objective scaling
		}
		# Only set fast_step_computation if supported (may not be available in all IPOPT versions)
		try:
			opts['ipopt.fast_step_computation'] = 'yes'
		except:
			pass

		LOG_DEBUG("Configuring IPOPT solver...")
		self.opti.solver('ipopt', opts)
		LOG_DEBUG("IPOPT solver configured")
		
		horizon = getattr(data, 'horizon', None) or self.horizon
		dynamics = getattr(data, 'dynamics_model', None) or self.dynamics_model
		timestep = getattr(data, 'timestep', None) or self.timestep
		LOG_DEBUG(f"Adding dynamics constraints for {horizon} stages (timestep={timestep})...")
		
		for k in range(horizon):
			x_k_list = [self.var_dict[var][k] for var in dynamics.get_dependent_vars()]
			u_k_list = [self.var_dict[var][k] for var in dynamics.get_inputs()]

			x_k = cs.vertcat(*x_k_list)
			u_k = cs.vertcat(*u_k_list)

			dynamics = getattr(data, 'dynamics_model', None) or self.dynamics_model
			x_next_list = [self.var_dict[var][k + 1] for var in dynamics.get_dependent_vars()]
			x_next = cs.vertcat(*x_next_list)

			# Use the model's symbolic dynamics function with a callable parameter getter
			def _param_getter(key):
				defaults = {
					"wheel_base": 2.79,
					"wheel_tread": 1.64,
					"front_overhang": 1.0,
					"rear_overhang": 1.1,
					"left_overhang": 0.128,
					"right_overhang": 0.128,
				}
				try:
					return data.parameters.get(key)
				except Exception:
					return defaults.get(key, 0.0)
			
			x_next_pred = dynamics.symbolic_dynamics(x_k, u_k, _param_getter, timestep)

			self.opti.subject_to(x_next == x_next_pred)
			if k == 0 or k == horizon - 1:
				LOG_DEBUG(f"  Added dynamics constraint for stage {k}")
		
		LOG_INFO(f"CasADiSolver initialized: {horizon} stages, {len(dependent_vars)} state vars, {len(input_vars)} input vars")

	def initialize_rollout(self, state: State, data=None, shift_forward=True):
		"""Initialize rollout for solving. Sets dynamics_model if provided via data."""
		if data is not None and hasattr(data, 'dynamics_model') and data.dynamics_model:
			self.dynamics_model = data.dynamics_model
			self.data = data
		if self.dynamics_model is None:
			LOG_WARN("initialize_rollout called but dynamics_model is not set")
		self._set_initial_state(state)
		self._initialize_warmstart(state, shift_forward)

	def get_objective_cost(self, state, stage_idx):
		"""Fetch per-stage objective cost terms via BaseSolver (ModuleManager-backed)."""
		from solver.base_solver import BaseSolver as _Base
		# Delegate to BaseSolver implementation to keep a single source of truth
		return _Base.get_objective_cost(self, state, stage_idx)

	def _initialize_warmstart(self, state: State, shift_forward=True):
		if self.solution is None or not shift_forward:
			self._initialize_base_warmstart(state)
		else:
			self._shift_warmstart_forward()

		self._set_opti_initial_values()

		initial_forecast = self._create_trajectory_from_warmstart()
		self.forecast.append(copy.deepcopy(initial_forecast))
		self.warmstart_intiailized = True

	def _initialize_base_warmstart(self, state: State):
		# Initialize states with a more realistic trajectory
		current_vel = state.get('v') if state.get('v') is not None else 0.0
		current_psi = state.get('psi') if state.get('psi') is not None else 0.0
		current_x = state.get('x') if state.get('x') is not None else 0.0
		current_y = state.get('y') if state.get('y') is not None else 0.0

		# Ensure horizon and timestep are set
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1

		# CRITICAL FIX: Initialize warmstart to follow reference path if available
		# This ensures the warmstart trajectory is feasible with road constraints
		has_reference_path = (self.data is not None and 
		                      hasattr(self.data, 'reference_path') and 
		                      self.data.reference_path is not None and
		                      hasattr(self.data.reference_path, 'x_spline') and
		                      hasattr(self.data.reference_path, 'y_spline') and
		                      self.data.reference_path.x_spline is not None)
		
		if has_reference_path and 'spline' in self.dynamics_model.get_dependent_vars():
			# Initialize warmstart to follow reference path
			LOG_DEBUG("Initializing warmstart to follow reference path")
			ref_path = self.data.reference_path
			
			# Get arc length bounds
			s_arr = np.asarray(ref_path.s, dtype=float) if hasattr(ref_path, 's') and ref_path.s is not None else None
			if s_arr is None or len(s_arr) == 0:
				# Fallback: estimate arc length from path points
				x_arr = np.asarray(ref_path.x, dtype=float)
				y_arr = np.asarray(ref_path.y, dtype=float)
				dx = np.diff(x_arr)
				dy = np.diff(y_arr)
				arc_lengths = np.sqrt(dx**2 + dy**2)
				s_arr = np.concatenate([[0], np.cumsum(arc_lengths)])
			
			s_min = float(s_arr[0])
			s_max = float(s_arr[-1])
			
			# Initialize spline variable: start at closest point on path
			current_spline = state.get('spline')
			if current_spline is None:
				# Find closest point on path to current position
				try:
					# Sample path at discrete points
					s_sample = np.linspace(s_min, s_max, min(100, len(s_arr)))
					x_sample = np.array([float(ref_path.x_spline(s)) for s in s_sample])
					y_sample = np.array([float(ref_path.y_spline(s)) for s in s_sample])
					# Find closest point
					distances = np.sqrt((x_sample - current_x)**2 + (y_sample - current_y)**2)
					closest_idx = np.argmin(distances)
					current_spline = float(s_sample[closest_idx])
					LOG_DEBUG(f"  Initialized spline to {current_spline:.4f} (closest point on path)")
				except Exception as e:
					LOG_DEBUG(f"  Could not find closest point: {e}, using s_min={s_min}")
					current_spline = s_min
			else:
				current_spline = float(current_spline)
			
			# Initialize spline profile: advance along path at current velocity
			spline_profile = np.zeros(horizon_val + 1)
			spline_profile[0] = current_spline
			
			# Use constant velocity profile
			if current_vel > 5.0:
				v_profile = np.maximum(0.0, current_vel + DEFAULT_BRAKING * np.arange(horizon_val + 1) * timestep_val * 0.5)
			else:
				v_profile = np.full(horizon_val + 1, current_vel)
			
			# Initialize position and heading by following reference path
			x_pos = np.zeros(horizon_val + 1)
			y_pos = np.zeros(horizon_val + 1)
			psi_profile = np.zeros(horizon_val + 1)
			
			x_pos[0] = current_x
			y_pos[0] = current_y
			psi_profile[0] = current_psi
			
			for k in range(1, horizon_val + 1):
				# Advance spline along path based on velocity
				ds = v_profile[k - 1] * timestep_val
				spline_profile[k] = min(s_max, spline_profile[k - 1] + ds)
				
				# Get position and heading from path at this spline value
				try:
					s_k = spline_profile[k]
					x_pos[k] = float(ref_path.x_spline(s_k))
					y_pos[k] = float(ref_path.y_spline(s_k))
					# Get heading from path tangent
					dx_path = float(ref_path.x_spline.derivative()(s_k))
					dy_path = float(ref_path.y_spline.derivative()(s_k))
					norm = np.sqrt(dx_path**2 + dy_path**2)
					if norm > 1e-6:
						psi_profile[k] = np.arctan2(dy_path, dx_path)
					else:
						psi_profile[k] = psi_profile[k - 1]
				except Exception as e:
					LOG_DEBUG(f"  Could not evaluate path at s={spline_profile[k]:.4f}: {e}, using previous values")
					x_pos[k] = x_pos[k - 1]
					y_pos[k] = y_pos[k - 1]
					psi_profile[k] = psi_profile[k - 1]
			
			self.warmstart_values['x'] = x_pos
			self.warmstart_values['y'] = y_pos
			self.warmstart_values['v'] = v_profile
			self.warmstart_values['psi'] = psi_profile
			self.warmstart_values['spline'] = spline_profile
			
			LOG_DEBUG(f"  Warmstart initialized: spline range [{spline_profile[0]:.4f}, {spline_profile[-1]:.4f}], positions follow path")
		else:
			# Fallback: straight-line motion (original behavior)
			LOG_DEBUG("Initializing warmstart with straight-line motion (no reference path available)")
			# Use constant velocity profile
			if current_vel > 5.0:
				v_profile = np.maximum(0.0, current_vel + DEFAULT_BRAKING * np.arange(horizon_val + 1) * timestep_val * 0.5)
			else:
				v_profile = np.full(horizon_val + 1, current_vel)
			
			psi_profile = np.full(horizon_val + 1, current_psi)

			self.warmstart_values['v'] = v_profile
			self.warmstart_values['psi'] = psi_profile

			# Integrate positions
			x_pos = np.zeros(horizon_val + 1)
			y_pos = np.zeros(horizon_val + 1)
			x_pos[0] = current_x
			y_pos[0] = current_y

			for k in range(1, horizon_val + 1):
				x_pos[k] = x_pos[k - 1] + v_profile[k - 1] * np.cos(psi_profile[k - 1]) * timestep_val
				y_pos[k] = y_pos[k - 1] + v_profile[k - 1] * np.sin(psi_profile[k - 1]) * timestep_val

			self.warmstart_values['x'] = x_pos
			self.warmstart_values['y'] = y_pos

		# Initialize other state variables if they exist
		for var_name in self.dynamics_model.get_dependent_vars():
			if var_name not in ['x', 'y', 'v', 'psi', 'spline']:
				current_val = state.get(var_name)
				if current_val is not None:
					self.warmstart_values[var_name][:] = current_val
				else:
					self.warmstart_values[var_name][:] = 0.0

		# Initialize inputs with small values (zero can cause issues)
		for var_name in self.dynamics_model.get_inputs():
			current_val = state.get(var_name)
			if current_val is not None:
				self.warmstart_values[var_name][:] = current_val
			else:
				# Use small non-zero values for better convergence
				self.warmstart_values[var_name][:] = 0.01

	def _shift_warmstart_forward(self):
		for var_name in self.dynamics_model.get_all_vars():
			if var_name in self.warmstart_values:
				self.warmstart_values[var_name][:-1] = self.warmstart_values[var_name][1:]
				if len(self.warmstart_values[var_name]) > 1:
					self.warmstart_values[var_name][-1] = self.warmstart_values[var_name][-2]

	def _update_warmstart_from_solution(self):
		for var_name in self.var_dict:
			try:
				self.warmstart_values[var_name] = np.array(self.solution.value(self.var_dict[var_name]))
			except Exception as e:
				LOG_WARN(f"Could not update warmstart for {var_name}: {e}")

	def _create_trajectory_from_warmstart(self):
		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		traj = Trajectory(timestep=timestep_val, length=horizon_val + 1)
		for k in range(horizon_val + 1):
			state_k = State(model_type=self.dynamics_model)
			for var_name in self.dynamics_model.get_dependent_vars():
				state_k.set(var_name, self.warmstart_values[var_name][k])
			if k < horizon_val:
				for var_name in self.dynamics_model.get_inputs():
					state_k.set(var_name, self.warmstart_values[var_name][k])
			traj.add_state(state_k)
		return traj

	def _set_opti_initial_values(self):
		for var_name, values in self.warmstart_values.items():
			if var_name in self.var_dict:
				self.opti.set_initial(self.var_dict[var_name], values)

	def _set_initial_state(self, state: State):
		self.initial_state = state
		if self.dynamics_model is None:
			LOG_WARN("_set_initial_state called but dynamics_model is not set")
			return
		for var_name in self.dynamics_model.get_dependent_vars():
			value = state.get(var_name)
			if value is not None:
				self.opti.subject_to(self.var_dict[var_name][0] == value)
				self.warmstart_values[var_name][0] = value

	def solve(self, state=None, data=None):
		"""Solve the optimization problem.
		
		Args:
			state: Current vehicle state (optional, can use self.state if set)
			data: Data object with constraints/objectives (optional, uses self.data if set)
		"""
		LOG_INFO("=== CasADiSolver.solve() ===")
		LOG_DEBUG("Attempting to solve in Casadi solver")
		
		# Use provided data or fallback to stored data
		if data is not None:
			self.data = data
			# Ensure data has required attributes
			if not hasattr(data, 'horizon') or data.horizon is None:
				data.horizon = self.horizon
			if not hasattr(data, 'timestep') or data.timestep is None:
				data.timestep = self.timestep
			if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
				data.dynamics_model = self.dynamics_model
		
		# Initialize solver if not already initialized
		if not hasattr(self, 'opti') or self.opti is None:
			self.opti = cs.Opti()
			if not self.var_dict:
				self.var_dict = {}
			if not self.warmstart_values:
				self.warmstart_values = {}
			
			# Ensure we have dynamics_model set
			if data and hasattr(data, 'dynamics_model') and data.dynamics_model:
				self.dynamics_model = data.dynamics_model
			elif self.data and hasattr(self.data, 'dynamics_model') and self.data.dynamics_model:
				self.dynamics_model = self.data.dynamics_model
			
			if self.dynamics_model and self.data:
				self.intialize_solver(self.data)
		
		total_objective = 0
		total_constraints_added = 0
		
		# Ensure horizon is set
		horizon_val = self.horizon if self.horizon is not None else 10
		
		# Log robot_area and disc information
		if self.data is not None:
			if hasattr(self.data, 'robot_area') and self.data.robot_area is not None:
				disc_info = []
				for i, disc in enumerate(self.data.robot_area):
					if hasattr(disc, 'offset') and hasattr(disc, 'radius'):
						disc_info.append(f"disc[{i}]: offset={disc.offset:.3f}, radius={disc.radius:.3f}")
				LOG_INFO(f"  robot_area in data: {len(self.data.robot_area)} disc(s) - {', '.join(disc_info)}")
			else:
				LOG_WARN("  robot_area not found in data or is None")
		
		# Log module manager state
		try:
			import logging as _logging
			_integ_logger = _logging.getLogger("integration_test")
			modules_list = [m.name for m in (self.module_manager.get_modules() or [])]
			msg_mod = f"Solver solve: module_manager has {len(modules_list)} modules: {modules_list}, data={self.data is not None}"
			LOG_INFO(msg_mod)
			try:
				_integ_logger.info(msg_mod)
			except Exception:
				pass
		except Exception:
			pass

		# CRITICAL: Loop over all stages (0 to horizon) to collect objectives and constraints
		# State variables: k in [0, horizon] (horizon + 1 values)
		# Input variables: k in [0, horizon-1] (horizon values)
		LOG_INFO(f"Collecting objectives and constraints for {horizon_val + 1} stages (0 to {horizon_val})")
		for stage_idx in range(horizon_val + 1):
			# Create symbolic state for this stage
			symbolic_state = State(self.dynamics_model)
			for var_name in self.dynamics_model.get_all_vars():
				if var_name in self.var_dict and stage_idx < self.var_dict[var_name].shape[0]:
					symbolic_state.set(var_name, self.var_dict[var_name][stage_idx])

			objective_costs = self.get_objective_cost(symbolic_state, stage_idx)
			# Log objectives BEFORE processing
			try:
				import logging as _logging
				_integ_logger = _logging.getLogger("integration_test")
				obj_keys = []
				for item in (objective_costs or []):
					if isinstance(item, dict):
						obj_keys.extend(list(item.keys()))
				msg_obj = f"Stage {stage_idx}: objectives count={len(objective_costs or [])} keys={obj_keys}"
				LOG_INFO(msg_obj)
				try:
					_integ_logger.info(msg_obj)
				except Exception:
					pass
			except Exception:
				pass
			for cost_dict in objective_costs:
				for cost_val in cost_dict.values():
					total_objective += cost_val

			constraints = self.get_constraints(stage_idx)
			# Log constraints BEFORE processing
			try:
				import logging as _logging
				_integ_logger = _logging.getLogger("integration_test")
				cons_summ = []
				for (c, lb, ub) in (constraints or []):
					ctype = None
					if isinstance(c, dict):
						ctype = c.get('type') or ('linear' if ('a1' in c and 'a2' in c) else 'dict')
					else:
						ctype = type(c).__name__
					cons_summ.append({
						"type": ctype,
						"lb": None if lb is None else 'set',
						"ub": None if ub is None else 'set',
					})
				msg_cons = f"Stage {stage_idx}: constraints count={len(constraints or [])} summary={cons_summ}"
				LOG_INFO(msg_cons)
				try:
					_integ_logger.info(msg_cons)
				except Exception:
					pass
			except Exception:
				pass
			if constraints:
				for (c, lb, ub) in constraints:
					# Robust bound handling with translation of structured constraints to CasADi
					if c is None:
						continue

					def _to_mx(val):
						if val is None:
							return None
						if isinstance(val, (int, float, np.floating)):
							return cs.DM(val)
						if isinstance(val, np.ndarray) and val.size == 1:
							return cs.DM(float(val))
						return val

					def _translate_constraint(expr_or_dict):
						# Accept already-built CasADi expressions
						if not isinstance(expr_or_dict, dict):
							return expr_or_dict
						cdef = expr_or_dict
						# Support common linearized halfspace: a1*x + a2*y <= b
						if ('a1' in cdef and 'a2' in cdef) or cdef.get('type') == 'linear':
							a1 = _to_mx(cdef.get('a1', 0.0))
							a2 = _to_mx(cdef.get('a2', 0.0))
							b = _to_mx(cdef.get('b', 0.0))
							xk = self.var_dict.get('x')[stage_idx] if 'x' in self.var_dict else None
							yk = self.var_dict.get('y')[stage_idx] if 'y' in self.var_dict else None
							if xk is None or yk is None:
								return None
							
							# CRITICAL FIX: Apply disc_offset to constraint
							# Constraint should be applied to disc position: p_disc = p_robot + offset * [cos(psi), sin(psi)]
							# For constraint a·p_disc <= b, we have:
							#   a1*(x + offset*cos(psi)) + a2*(y + offset*sin(psi)) <= b
							# Which expands to: a1*x + a2*y + offset*(a1*cos(psi) + a2*sin(psi)) <= b
							# So: a1*x + a2*y <= b - offset*(a1*cos(psi) + a2*sin(psi))
							disc_offset = cdef.get('disc_offset', 0.0)
							if abs(float(disc_offset)) > 1e-9:
								# Get orientation angle (psi or theta depending on model)
								psi_k = None
								for angle_var in ['psi', 'theta', 'heading']:
									if angle_var in self.var_dict:
										psi_k = self.var_dict[angle_var][stage_idx]
										break
								
								if psi_k is not None:
									# Apply disc offset: adjust b by offset*(a1*cos(psi) + a2*sin(psi))
									offset_adjustment = disc_offset * (a1 * cs.cos(psi_k) + a2 * cs.sin(psi_k))
									b_adjusted = b - offset_adjustment
									LOG_DEBUG(f"  Stage {stage_idx}: Applied disc_offset={disc_offset:.3f}, b adjustment={float(offset_adjustment) if hasattr(offset_adjustment, '__float__') else 'symbolic'}")
								else:
									LOG_WARN(f"  Stage {stage_idx}: disc_offset={disc_offset:.3f} specified but no orientation angle variable found (psi/theta/heading)")
									b_adjusted = b
							else:
								b_adjusted = b
							
							# Halfspace: a·p_disc <= b → expr := a·p_robot - b_adjusted <= 0
							return a1 * xk + a2 * yk - b_adjusted
						# Unknown structured type; skip
						return None

					c_expr = _translate_constraint(c)
					if c_expr is None:
						continue

					lb_mx = _to_mx(lb)
					ub_mx = _to_mx(ub)

					try:
						# Log constraint details for stage 0 to diagnose issues
						if stage_idx == 0 and total_constraints_added < 4:
							if isinstance(c, dict):
								LOG_DEBUG(f"  Stage 0 constraint {total_constraints_added}: a1={c.get('a1', 'N/A')}, a2={c.get('a2', 'N/A')}, b={c.get('b', 'N/A')}, disc_offset={c.get('disc_offset', 'N/A')}, lb={lb}, ub={ub}")
						
						if lb_mx is not None and ub_mx is not None:
							self.opti.subject_to(self.opti.bounded(lb_mx, c_expr, ub_mx))
						elif lb_mx is not None:
							self.opti.subject_to(lb_mx <= c_expr)
						elif ub_mx is not None:
							self.opti.subject_to(c_expr <= ub_mx)
						else:
							self.opti.subject_to(c_expr)
						total_constraints_added += 1
					except Exception as e:
						LOG_WARN(f"Failed to add constraint at stage {stage_idx}: {e}")

				# All costs should come from objective modules via BaseSolver.get_objective_cost

		self.opti.minimize(total_objective)
		LOG_INFO(f"Optimization problem setup complete: {total_objective.shape[0] if hasattr(total_objective, 'shape') else 'scalar'} objective, {total_constraints_added} constraints")
		LOG_DEBUG(f"  Solving over horizon: {horizon_val} steps (states: 0 to {horizon_val}, controls: 0 to {horizon_val-1})")

		try:
			LOG_INFO("Attempting to solve optimization problem over entire horizon...")
			self.solution = self.opti.solve()
			self.exit_flag = 1
			self._update_warmstart_from_solution()
			LOG_INFO("=== SOLUTION FOUND ===")
			LOG_INFO(f"  Optimal trajectory computed for {horizon_val + 1} states and {horizon_val} control inputs")
			
			# Log solution details
			if self.solution:
				LOG_DEBUG("Extracting solution values...")
				# Log first state values
				try:
					for var_name in self.dynamics_model.get_dependent_vars():
						if var_name in self.var_dict:
							val0 = self.solution.value(self.var_dict[var_name][0])
							LOG_DEBUG(f"  Solution[{var_name}][0] = {val0}")
					# Log first control values
					for var_name in self.dynamics_model.get_inputs():
						if var_name in self.var_dict:
							val0 = self.solution.value(self.var_dict[var_name][0])
							LOG_INFO(f"  Solution u[{var_name}][0] = {val0} (first control input to apply)")
				except Exception as e:
					LOG_DEBUG(f"  Could not extract solution values: {e}")
			
			return 1
		except RuntimeError as e:
			LOG_WARN(f"[ERROR] CasADi solver failed: {e}")
			self.exit_flag = -1
			self.info["status"] = "failed"
			self.info["error"] = str(e)
			
			# Try to diagnose infeasibility
			LOG_WARN("Attempting to diagnose infeasibility...")
			try:
				# Check if we can get debug values
				if hasattr(self.opti, 'debug'):
					LOG_DEBUG("Checking opti.debug values...")
					for var_name in list(self.var_dict.keys())[:5]:  # Check first few variables
						try:
							debug_val = self.opti.debug.value(self.var_dict[var_name])
							if isinstance(debug_val, np.ndarray) and len(debug_val) > 0:
								LOG_DEBUG(f"  Debug[{var_name}][0] = {debug_val[0]}")
							else:
								LOG_DEBUG(f"  Debug[{var_name}] = {debug_val}")
						except Exception:
							pass
					
					# Check constraint violations
					try:
						# Get constraint values
						LOG_DEBUG("Checking constraint violations...")
						# Try to get constraint info if available
						if hasattr(self.opti, 'debug') and hasattr(self.opti.debug, 'g'):
							constraint_violations = self.opti.debug.value(self.opti.debug.g)
							if constraint_violations is not None:
								# For constraints expr <= 0, violation means expr > 0
								# For constraints lb <= expr <= ub, violation means expr < lb or expr > ub
								# The debug.g values are the constraint expressions themselves
								# We need to check against bounds to find actual violations
								violations = []
								constraints_with_bounds = []
								for stage_idx_check in range(horizon_val + 1):
									cons_list = self.get_constraints(stage_idx_check)
									constraints_with_bounds.extend(cons_list)
								
								# Check each constraint against its bounds
								if len(constraint_violations) == len(constraints_with_bounds):
									for i, (c, lb, ub) in enumerate(constraints_with_bounds[:min(10, len(constraints_with_bounds))]):
										expr_val = float(constraint_violations[i])
										if lb is not None and expr_val < lb:
											violations.append(lb - expr_val)
										elif ub is not None and expr_val > ub:
											violations.append(expr_val - ub)
								
								if violations:
									max_violation = max(violations)
									LOG_WARN(f"  Maximum constraint violation: {max_violation:.6f}")
								else:
									# Fallback: use absolute values (may not be accurate for all constraint types)
									violations_abs = np.abs(constraint_violations)
									max_violation = np.max(violations_abs)
									LOG_WARN(f"  Maximum constraint value (abs): {max_violation:.6f}")
								if max_violation > 1e-3:
									LOG_WARN(f"  Problem likely infeasible - constraint violation too large")
					except Exception as const_err:
						LOG_DEBUG(f"Could not check constraint violations: {const_err}")
			except Exception as debug_err:
				LOG_DEBUG(f"Could not get debug values: {debug_err}")
			
			# Log current state and data
			if state and self.dynamics_model:
				try:
					state_vars = {}
					for v in self.dynamics_model.get_all_vars():
						if state.has(v):
							state_vars[v] = state.get(v)
					LOG_WARN(f"State at failure: {state_vars}")
					# Check if initial state might violate constraints
					if 'x' in state_vars and 'y' in state_vars:
						LOG_DEBUG(f"  Initial position: ({state_vars['x']:.2f}, {state_vars['y']:.2f})")
				except Exception as state_err:
					LOG_DEBUG(f"Could not log state: {state_err}")
			if self.data:
				has_ref = hasattr(self.data, 'reference_path') and self.data.reference_path is not None
				has_static = hasattr(self.data, 'static_obstacles') and self.data.static_obstacles is not None
				LOG_WARN(f"Data at failure: reference_path={'present' if has_ref else 'missing'}, static_obstacles={'present' if has_static else 'missing'}")
				# Check if initial state is within road bounds
				if has_static and state and hasattr(state, 'get'):
					try:
						x0 = state.get('x')
						y0 = state.get('y')
						if x0 is not None and y0 is not None and hasattr(self.data, 'static_obstacles'):
							# Check first stage obstacle
							if self.data.static_obstacles and len(self.data.static_obstacles) > 0:
								obs0 = self.data.static_obstacles[0]
								if obs0 is not None and hasattr(obs0, 'halfspaces'):
									# Check if point violates any halfspace
									for i, halfspace in enumerate(obs0.halfspaces):
										if hasattr(halfspace, 'A') and hasattr(halfspace, 'b'):
											A = np.array(halfspace.A).flatten()
											b = float(halfspace.b)
											violation = np.dot(A, [x0, y0]) - b
											LOG_WARN(f"  Halfspace {i}: A={A}, b={b:.6f}, A·p={np.dot(A, [x0, y0]):.6f}, violation={violation:.6f}")
											if violation > 1e-6:
												LOG_WARN(f"  Initial position ({x0:.2f}, {y0:.2f}) violates road constraint {i}: A·p - b = {violation:.6f}")
					except Exception as check_err:
						LOG_DEBUG(f"Could not check initial position against constraints: {check_err}")
			
			return -1

	def get_output(self, k, var_name):
		"""Get output value for variable at stage k.
		
		Args:
			k: Stage index (0-based). For state variables: k in [0, horizon]. For input variables: k in [0, horizon-1]
			var_name: Name of variable (state or input)
		
		Returns:
			Value at stage k, or None if not available
		"""
		if not self.solution:
			LOG_DEBUG(f"get_output({k}, {var_name}): No solution available")
			return None
		
		if var_name not in self.var_dict:
			LOG_DEBUG(f"get_output({k}, {var_name}): Variable not in var_dict")
			return None
		
		var = self.var_dict[var_name]
		if k >= var.shape[0]:
			LOG_DEBUG(f"get_output({k}, {var_name}): k={k} >= var.shape[0]={var.shape[0]}")
			return None
		
		try:
			val = self.solution.value(var[k])
			LOG_DEBUG(f"get_output({k}, {var_name}): {val}")
			return val
		except Exception as e:
			LOG_WARN(f"get_output({k}, {var_name}): Error extracting value: {e}")
			return None

	def get_reference_trajectory(self):
		if not self.solution:
			LOG_WARN("No solution available. Returning trajectory from warmstart.")
			return self._create_trajectory_from_warmstart()

		horizon_val = self.horizon if self.horizon is not None else 10
		timestep_val = self.timestep if self.timestep is not None else 0.1
		traj = Trajectory(timestep=timestep_val, length=horizon_val + 1)
		for k in range(horizon_val + 1):
			state_k = State(model_type=self.dynamics_model)
			for var_name in self.dynamics_model.get_dependent_vars():
				state_k.set(var_name, self.solution.value(self.var_dict[var_name][k]))
			if k < horizon_val:
				for var_name in self.dynamics_model.get_inputs():
					state_k.set(var_name, self.solution.value(self.var_dict[var_name][k]))
			traj.add_state(state_k)
		return traj

	def print_if_bound_limited(self):
		if not self.solution: return
		LOG_DEBUG("Checking variable bounds:")
		for var_name, var in self.var_dict.items():
			values = self.solution.value(var)
			lb, ub, _ = self.dynamics_model.get_bounds(var_name)
			if np.any(np.isclose(values, lb, atol=1e-4)):
				LOG_WARN(f"Variable '{var_name}' is at its lower bound.")
			if np.any(np.isclose(values, ub, atol=1e-4)):
				LOG_WARN(f"Variable '{var_name}' is at its upper bound.")

	def explain_exit_flag(self, code=None):
		code_to_check = code if code is not None else self.exit_flag
		explanations = {
			1: "Optimization successful.",
			-1: f"Optimization failed: {self.info.get('error', 'Unknown error')}"
		}
		return explanations.get(code_to_check, f"Unknown exit code: {code_to_check}")

	def set_dynamics_model(self, model):
		"""Set the dynamics model for the solver."""
		self.dynamics_model = model
		# If solver is already initialized, reinitialize with new model
		if self.opti is not None and self.data:
			self.intialize_solver(self.data)
	
	def reset(self):
		LOG_DEBUG("Resetting CasADi solver.")
		self.opti = cs.Opti()
		self.solution = None
		self.exit_flag = None
		self.info.clear()
		self.var_dict.clear()
		self.warmstart_values.clear()
		self.forecast.clear()

		if self.dynamics_model:
			self.set_dynamics_model(self.dynamics_model)

		for module in self.module_manager.get_modules():
			module.reset()