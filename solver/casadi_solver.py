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
		# DO NOT store dynamics_model - always get it from data to avoid discrepancies
		self.solution = None
		self.exit_flag = None
		self.info = {}
		self.forecast = []
		LOG_INFO(f"{(id(self))} CasADiSolver: Initializing solver")
	
	def _get_dynamics_model(self):
		"""Get dynamics model from data - always use data as source of truth."""
		if self.data is None:
			return None
		if not hasattr(self.data, 'dynamics_model'):
			return None
		return self.data.dynamics_model

	def intialize_solver(self, data):
		"""Initialize solver variables based on dynamics model and horizon."""
		LOG_INFO("=== CasADiSolver.intialize_solver() ===")
		if data is None:
			LOG_WARN("Cannot initialize solver: data is None")
			return
		
		# Always get dynamics model from data - do not use stored version
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("Cannot initialize solver: data.dynamics_model is None")
			return
		
		dynamics = data.dynamics_model
		horizon = getattr(data, 'horizon', None) or self.horizon
		LOG_DEBUG(f"Initializing solver: horizon={horizon}, dynamics_model={dynamics.__class__.__name__}")
		
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
		
		# Always get dynamics model from data
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("Cannot add dynamics constraints: data.dynamics_model is None")
			return
		
		dynamics = data.dynamics_model
		horizon = getattr(data, 'horizon', None) or self.horizon
		timestep = getattr(data, 'timestep', None) or self.timestep
		LOG_DEBUG(f"Adding dynamics constraints for {horizon} stages (timestep={timestep})...")
		
		for k in range(horizon):
			x_k_list = [self.var_dict[var][k] for var in dynamics.get_dependent_vars()]
			u_k_list = [self.var_dict[var][k] for var in dynamics.get_inputs()]

			x_k = cs.vertcat(*x_k_list)
			u_k = cs.vertcat(*u_k_list)
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
		"""Initialize rollout for solving. Gets state and dynamics_model from data."""
		if data is None:
			LOG_WARN("initialize_rollout called but data is None")
			return
		
		# Store data reference - this is the single source of truth
		self.data = data
		
		# Get state from data if available, otherwise use provided state
		# Priority: data.state > provided state parameter
		if hasattr(data, 'state') and data.state is not None:
			state = data.state
			LOG_DEBUG("Using state from data.state")
		elif state is None:
			LOG_WARN("initialize_rollout: no state available (neither data.state nor state parameter)")
			return
		
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("initialize_rollout called but data.dynamics_model is not set")
			return
		
		# Ensure solver is initialized before setting initial state
		if self.opti is None or not self.var_dict:
			LOG_DEBUG("Solver not initialized in initialize_rollout - initializing now")
			self.intialize_solver(data)
		
		# DO NOT store state separately - always get from data
		self._set_initial_state(state)
		# Enable warmstart initialization to provide feasible initial guess
		self._initialize_warmstart(state, shift_forward)

	def get_objective_cost(self, state, stage_idx):
		"""Fetch per-stage objective cost terms via BaseSolver (ModuleManager-backed)."""
		from solver.base_solver import BaseSolver as _Base
		# Delegate to BaseSolver implementation to keep a single source of truth
		return _Base.get_objective_cost(self, state, stage_idx)

	def _initialize_warmstart(self, state: State, shift_forward=True):
		# Initialize warmstart to provide feasible initial guess for IPOPT
		LOG_DEBUG("Initializing warmstart trajectory")
		
		if not hasattr(self, 'forecast'):
			self.forecast = []
		
		# Initialize base warmstart which creates path-following trajectory
		self._initialize_base_warmstart(state)
		
		# Set initial values in CasADi opti problem
		self._set_opti_initial_values()
		
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
		
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("_initialize_base_warmstart: no dynamics_model available")
			return
		
		if has_reference_path and 'spline' in dynamics_model.get_dependent_vars():
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
			
			# Use constant velocity profile, but ensure it's within bounds
			try:
				v_lb, v_ub, _ = dynamics_model.get_bounds('v')
			except Exception:
				v_lb, v_ub = -0.01, 3.0  # Default bounds
			
			if current_vel > 5.0:
				v_profile = np.maximum(v_lb, np.minimum(v_ub, current_vel + DEFAULT_BRAKING * np.arange(horizon_val + 1) * timestep_val * 0.5))
			else:
				v_profile = np.full(horizon_val + 1, np.clip(current_vel, v_lb, v_ub))
			
			# Initialize position and heading by following reference path
			x_pos = np.zeros(horizon_val + 1)
			y_pos = np.zeros(horizon_val + 1)
			psi_profile = np.zeros(horizon_val + 1)
			
			x_pos[0] = current_x
			y_pos[0] = current_y
			psi_profile[0] = current_psi
			
			# Initialize control inputs to guide along path
			a_profile = np.zeros(horizon_val)
			w_profile = np.zeros(horizon_val)
			
			for k in range(1, horizon_val + 1):
				# Advance spline along path based on velocity
				ds = v_profile[k - 1] * timestep_val
				spline_profile[k] = min(s_max, spline_profile[k - 1] + ds)
				
				# Get desired position and heading from path at this spline value
				# CRITICAL: Stay close to path centerline to ensure road constraint feasibility
				try:
					s_k = spline_profile[k]
					x_desired = float(ref_path.x_spline(s_k))
					y_desired = float(ref_path.y_spline(s_k))
					# Get desired heading from path tangent
					dx_path = float(ref_path.x_spline.derivative()(s_k))
					dy_path = float(ref_path.y_spline.derivative()(s_k))
					norm = np.sqrt(dx_path**2 + dy_path**2)
					if norm > 1e-6:
						psi_desired = np.arctan2(dy_path, dx_path)
					else:
						psi_desired = psi_profile[k - 1]
					
					# CRITICAL FIX: Ensure warmstart stays well within road bounds
					# The warmstart should follow the path centerline, which is the safest position
					# This ensures the trajectory satisfies road constraints at all stages
				
					# CRITICAL: Compute dynamics-consistent next state from previous state and control
					# The dynamics constraint for transition k->k+1 uses: x[k+1] = x[k] + dt * v[k] * cos(psi[k])
					# So we need to ensure that when we compute x[k], we use the correct values from k-1
					# Then when the solver checks x[k+1] = x[k] + dt * v[k] * cos(psi[k]), it will be satisfied
					
					# Get state at k-1 (already computed in previous iteration)
					x_km1 = x_pos[k - 1]
					y_km1 = y_pos[k - 1]
					psi_km1 = psi_profile[k - 1]
					v_km1 = v_profile[k - 1]
					
					# Calculate steering input to turn toward desired heading
					psi_error = psi_desired - psi_km1
					# Normalize angle error to [-pi, pi]
					while psi_error > np.pi:
						psi_error -= 2 * np.pi
					while psi_error < -np.pi:
						psi_error += 2 * np.pi
					
					# Proportional control for steering: w[k-1] should produce psi[k] ≈ psi_desired
					# For Euler: psi[k] = psi[k-1] + dt * w[k-1], so w[k-1] = (psi_desired - psi[k-1]) / dt
					w_max = 0.8  # From bounds
					w_desired = psi_error / timestep_val  # Direct calculation to achieve desired heading
					w_km1 = np.clip(w_desired, -w_max, w_max)
					w_profile[k - 1] = w_km1
					
					# Calculate acceleration to maintain/reach desired velocity
					# For now, use zero acceleration (can be improved)
					a_km1 = 0.0
					a_profile[k - 1] = a_km1
					
					# Propagate dynamics: Euler integration
					# Compute state at k using state at k-1 and control u[k-1]
					# This matches the dynamics constraint: x[k] = x[k-1] + dt * v[k-1] * cos(psi[k-1])
					psi_k = psi_km1 + timestep_val * w_km1
					v_k = v_km1 + timestep_val * a_km1
					x_k = x_km1 + timestep_val * v_km1 * np.cos(psi_km1)
					y_k = y_km1 + timestep_val * v_km1 * np.sin(psi_km1)
					
					# Store computed state at k
					x_pos[k] = x_k
					y_pos[k] = y_k
					psi_profile[k] = psi_k
					v_profile[k] = v_k
					
				except Exception as e:
					LOG_DEBUG(f"  Could not evaluate path at s={spline_profile[k]:.4f}: {e}, using dynamics propagation")
					# Fallback: propagate using zero control
					x_km1 = x_pos[k - 1]
					y_km1 = y_pos[k - 1]
					psi_km1 = psi_profile[k - 1]
					v_km1 = v_profile[k - 1]
					
					# Propagate with zero control: u[k-1] = [0, 0]
					psi_k = psi_km1
					v_k = v_km1
					x_k = x_km1 + timestep_val * v_km1 * np.cos(psi_km1)
					y_k = y_km1 + timestep_val * v_km1 * np.sin(psi_km1)
					
					x_pos[k] = x_k
					y_pos[k] = y_k
					psi_profile[k] = psi_k
					v_profile[k] = v_k
					a_profile[k - 1] = 0.0
					w_profile[k - 1] = 0.0
			
			# CRITICAL: After clipping velocity, we need to recompute positions to maintain dynamics consistency
			# Store original values before clipping
			v_profile_original = v_profile.copy()
			
			# Ensure velocity is within bounds
			try:
				v_lb, v_ub, _ = dynamics_model.get_bounds('v')
				v_profile = np.clip(v_profile, v_lb, v_ub)
				# If velocity was clipped, positions might need adjustment, but we'll keep them as-is
				# since they were computed from the original velocity profile
			except Exception:
				pass
			
			# CRITICAL: Recompute positions using the clipped velocity to ensure dynamics consistency
			# The dynamics constraint for transition k->k+1 uses: x[k+1] = x[k] + dt * v[k] * cos(psi[k])
			# So we need to ensure: x_pos[k+1] = x_pos[k] + dt * v_profile[k] * cos(psi_profile[k])
			# We already computed x_pos[k] using v_profile[k-1], so we need to recompute using v_profile[k]
			# But wait - we computed x_pos[k] using v_profile[k-1], which is correct for the transition (k-1)->k
			# For the transition k->(k+1), we need x_pos[k+1] = x_pos[k] + dt * v_profile[k] * cos(psi_profile[k])
			# So we should NOT recompute - the positions are already correct!
			# The issue is that we're clipping v_profile AFTER computing positions, which breaks consistency
			# Solution: Don't clip velocity after computing positions - clip it during the computation loop
			# Actually, we already clip it, so the positions should be consistent. Let's verify instead of recomputing.
			
			self.warmstart_values['x'] = x_pos
			self.warmstart_values['y'] = y_pos
			self.warmstart_values['v'] = v_profile
			self.warmstart_values['psi'] = psi_profile
			# Ensure spline is within bounds
			try:
				s_lb, s_ub, _ = dynamics_model.get_bounds('spline')
				spline_profile = np.clip(spline_profile, s_lb, s_ub)
			except Exception:
				# Fallback: clip to path bounds
				spline_profile = np.clip(spline_profile, s_min, s_max)
			self.warmstart_values['spline'] = spline_profile
			
			# Set control inputs in warmstart
			self.warmstart_values['a'] = a_profile
			self.warmstart_values['w'] = w_profile
			
			LOG_DEBUG(f"  Warmstart initialized: dynamics-consistent trajectory with steering inputs")
			
			# Verify dynamics consistency for first few transitions
			for k_check in range(min(5, horizon_val)):
				x_k = x_pos[k_check]
				x_kp1 = x_pos[k_check + 1]
				v_k = v_profile[k_check]
				psi_k = psi_profile[k_check]
				x_pred = x_k + timestep_val * v_k * np.cos(psi_k)
				violation = x_kp1 - x_pred
				if abs(violation) > 1e-3:
					LOG_WARN(f"  WARMSTART WARNING: Transition {k_check}->{k_check+1}: x[{k_check+1}]={x_kp1:.6f}, predicted={x_pred:.6f}, violation={violation:.6f}")
				else:
					LOG_DEBUG(f"  Warmstart transition {k_check}->{k_check+1}: dynamics consistent (violation={violation:.6f})")
			
			LOG_DEBUG(f"  Warmstart initialized: spline range [{spline_profile[0]:.4f}, {spline_profile[-1]:.4f}], positions follow path")
		else:
			# Fallback: straight-line motion (original behavior)
			LOG_DEBUG("Initializing warmstart with straight-line motion (no reference path available)")
			# Use constant velocity profile, but ensure it's within bounds
			dynamics_model = self._get_dynamics_model()
			if dynamics_model:
				try:
					v_lb, v_ub, _ = dynamics_model.get_bounds('v')
				except Exception:
					v_lb, v_ub = -0.01, 3.0  # Default bounds
			else:
				v_lb, v_ub = -0.01, 3.0  # Default bounds
			
			if current_vel > 5.0:
				v_profile = np.maximum(v_lb, np.minimum(v_ub, current_vel + DEFAULT_BRAKING * np.arange(horizon_val + 1) * timestep_val * 0.5))
			else:
				v_profile = np.full(horizon_val + 1, np.clip(current_vel, v_lb, v_ub))
			
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
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return
		for var_name in dynamics_model.get_dependent_vars():
			if var_name not in ['x', 'y', 'v', 'psi', 'spline']:
				current_val = state.get(var_name)
				if current_val is not None:
					self.warmstart_values[var_name][:] = current_val
				else:
					self.warmstart_values[var_name][:] = 0.0

		# Initialize inputs with small values (zero can cause issues)
		for var_name in dynamics_model.get_inputs():
			current_val = state.get(var_name)
			if current_val is not None:
				self.warmstart_values[var_name][:] = current_val
			else:
				# Use small non-zero values for better convergence
				self.warmstart_values[var_name][:] = 0.01

	def _shift_warmstart_forward(self):
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return
		for var_name in dynamics_model.get_all_vars():
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
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return Trajectory(timestep=timestep_val, length=0)
		traj = Trajectory(timestep=timestep_val, length=horizon_val + 1)
		for k in range(horizon_val + 1):
			state_k = State(model_type=dynamics_model)
			for var_name in dynamics_model.get_dependent_vars():
				state_k.set(var_name, self.warmstart_values[var_name][k])
			if k < horizon_val:
				for var_name in dynamics_model.get_inputs():
					state_k.set(var_name, self.warmstart_values[var_name][k])
			traj.add_state(state_k)
		return traj

	def _set_opti_initial_values(self):
		"""Set initial values for optimization variables from warmstart values."""
		if not hasattr(self, 'warmstart_values') or not self.warmstart_values:
			LOG_DEBUG("_set_opti_initial_values: No warmstart values available")
			return
		
		LOG_DEBUG("Setting initial values in CasADi opti problem from warmstart")
		for var_name in self.var_dict:
			if var_name in self.warmstart_values:
				try:
					ws_vals = self.warmstart_values[var_name]
					if isinstance(ws_vals, np.ndarray) and len(ws_vals) > 0:
						# Ensure values are within bounds
						dynamics_model = self._get_dynamics_model()
						if dynamics_model:
							try:
								lb, ub, _ = dynamics_model.get_bounds(var_name)
								ws_vals = np.clip(ws_vals, lb, ub)
							except Exception:
								pass
						# Set initial value in CasADi
						self.opti.set_initial(self.var_dict[var_name], ws_vals)
						LOG_DEBUG(f"  Set initial value for {var_name}: shape={ws_vals.shape}, range=[{np.min(ws_vals):.3f}, {np.max(ws_vals):.3f}]")
				except Exception as e:
					LOG_WARN(f"  Could not set initial value for {var_name}: {e}")

	def _set_initial_state(self, state: State):
		"""Set initial state constraints. State comes from data, not stored separately."""
		# DO NOT store state - always get from data when needed
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("_set_initial_state called but dynamics_model is not set")
			return
		if not self.var_dict:
			LOG_WARN("_set_initial_state called but var_dict is empty - solver not initialized")
			return
		for var_name in dynamics_model.get_dependent_vars():
			if var_name not in self.var_dict:
				LOG_WARN(f"_set_initial_state: variable '{var_name}' not in var_dict. Available: {list(self.var_dict.keys())}")
				continue
			value = state.get(var_name)
			if value is not None:
				self.opti.subject_to(self.var_dict[var_name][0] == value)
				if var_name in self.warmstart_values:
					self.warmstart_values[var_name][0] = value

	def solve(self, state=None, data=None):
		"""Solve the optimization problem.
		
		Args:
			state: Current vehicle state (DEPRECATED - use data.state instead)
			data: Data object with state, constraints, objectives, and dynamics_model
		"""
		LOG_INFO("=== CasADiSolver.solve() ===")
		LOG_DEBUG("Attempting to solve in Casadi solver")
		
		# Data is the single source of truth - store reference but don't duplicate
		if data is None:
			if self.data is None:
				LOG_WARN("solve: data is None and no stored data available - cannot solve")
				return -1
			data = self.data
			LOG_DEBUG("Using stored data reference")
		else:
			# Update data reference - this is the source of truth
			self.data = data
		
		# Get state from data - this is the authoritative source
		if hasattr(data, 'state') and data.state is not None:
			state = data.state
			LOG_DEBUG("Using state from data.state")
		elif state is None:
			LOG_WARN("solve: no state available (neither data.state nor state parameter)")
			return -1
		else:
			LOG_WARN("solve: state provided as parameter but data.state should be used instead")
		
		# Ensure data has required attributes
		if not hasattr(data, 'horizon') or data.horizon is None:
			data.horizon = self.horizon
		if not hasattr(data, 'timestep') or data.timestep is None:
			data.timestep = self.timestep
		# Ensure data has dynamics_model - it should be set by planner
		if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
			LOG_WARN("solve: data.dynamics_model is not set - cannot solve")
			return -1
		
		# Initialize solver if not already initialized
		if not hasattr(self, 'opti') or self.opti is None:
			self.opti = cs.Opti()
			if not self.var_dict:
				self.var_dict = {}
			if not self.warmstart_values:
				self.warmstart_values = {}
			
			# Ensure data has dynamics_model - it should be set by planner
			if not hasattr(data, 'dynamics_model') or data.dynamics_model is None:
				LOG_WARN("solve: data.dynamics_model is not set - cannot initialize solver")
				return -1
			
			# Initialize solver with data
			self.intialize_solver(data)
		
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

		# Before building objectives/constraints, dump warmstart and bounds at k=0
		try:
			dynamics_model = self._get_dynamics_model()
			dv = dynamics_model.get_dependent_vars() if dynamics_model else []
			iv = dynamics_model.get_inputs() if dynamics_model else []
			keys_to_show = [k for k in ['x','y','psi','v','spline'] if k in dv] + [k for k in ['a','w','delta'] if k in iv]
			msg_ws = []
			msg_bd = []
			for name in keys_to_show:
				ws0 = None
				try:
					ws0 = float(self.warmstart_values.get(name, [None])[0])
				except Exception:
					ws0 = 'n/a'
				msg_ws.append(f"{name}0={ws0}")
				try:
					lb, ub, _ = dynamics_model.get_bounds(name)
					msg_bd.append(f"{name}∈[{lb},{ub}]")
				except Exception:
					pass
			if msg_ws:
				LOG_INFO(f"  Warmstart k=0: {' , '.join(msg_ws)}")
				try:
					import logging as _logging
					_integ_logger = _logging.getLogger("integration_test")
					_integ_logger.info(f"Warmstart k=0: {' , '.join(msg_ws)}")
				except Exception:
					pass
			if msg_bd:
				LOG_INFO(f"  Bounds: {' , '.join(msg_bd)}")
				try:
					import logging as _logging
					_integ_logger = _logging.getLogger("integration_test")
					_integ_logger.info(f"Bounds: {' , '.join(msg_bd)}")
				except Exception:
					pass
		except Exception:
			pass

		# Multi-stage dynamics residual diagnostics (k=0..2) using planner/base-solver path
		try:
			max_k = min(3, horizon_val)
			import numpy as _np
			dynamics_model = self._get_dynamics_model()
			dep_vars = dynamics_model.get_dependent_vars() if dynamics_model else []
			in_vars = dynamics_model.get_inputs() if dynamics_model else []
			for k in range(max_k):
				xk_num = []
				uk_num = []
				for name in dep_vars:
					seq = self.warmstart_values.get(name, [])
					xk_num.append(float(seq[k] if k < len(seq) else 0.0))
				for name in in_vars:
					seq = self.warmstart_values.get(name, [])
					uk_num.append(float(seq[k] if k < len(seq) else 0.0))
				# Compute predicted next state via planner/base-solver path
				x_sym = cs.SX.sym('x', len(xk_num))
				u_sym = cs.SX.sym('u', len(uk_num) if len(uk_num) > 0 else 1)
				x_next_sym = self._compute_next_state(dynamics_model, x_sym, u_sym, self.timestep, self.data, symbolic=True)
				# Debug: check shape of x_next_sym
				try:
					x_next_shape = x_next_sym.shape if hasattr(x_next_sym, 'shape') else 'unknown'
					x_next_size = x_next_sym.size1() if hasattr(x_next_sym, 'size1') else (x_next_sym.shape[0] if hasattr(x_next_sym, 'shape') and len(x_next_sym.shape) > 0 else 'unknown')
					LOG_INFO(f"  Dyn residual k={k}: x_next_sym shape={x_next_shape}, size={x_next_size}, xk_num len={len(xk_num)}")
				except Exception as _e:
					LOG_WARN(f"  Dyn residual k={k}: failed to get x_next_sym shape: {_e}")
				f_fun = cs.Function('f_next', [x_sym, u_sym], [x_next_sym])
				f_result = f_fun(cs.DM(xk_num), cs.DM(uk_num if len(uk_num) > 0 else [0.0]))
				# Debug: check what f_fun returns
				try:
					f_result_shape = f_result.shape if hasattr(f_result, 'shape') else 'unknown'
					f_result_size = f_result.size1() if hasattr(f_result, 'size1') else 'unknown'
					LOG_INFO(f"  Dyn residual k={k}: f_result type={type(f_result)}, shape={f_result_shape}, size={f_result_size}")
				except Exception as _e:
					LOG_WARN(f"  Dyn residual k={k}: failed to check f_result: {_e}")
				# Extract full vector from DM (f_result is already the output DM, not a tuple)
				try:
					x_next_pred = f_result.full().flatten().tolist()
					LOG_INFO(f"  Dyn residual k={k}: extracted pred len={len(x_next_pred)}, values={x_next_pred}")
				except Exception as _e:
					LOG_WARN(f"  Dyn residual k={k}: failed to extract pred: {_e}")
					x_next_pred = [0.0] * len(dep_vars)  # Fallback
				# Warmstart x_{k+1}
				xkp1_ws = []
				for name in dep_vars:
					seq = self.warmstart_values.get(name, [])
					xkp1_ws.append(float(seq[k+1] if k + 1 < len(seq) else (seq[-1] if seq else 0.0)))
				# Debug: check if pred has right size
				if len(x_next_pred) != len(xkp1_ws):
					LOG_WARN(f"  Dyn residual k={k}: pred has {len(x_next_pred)} elements, expected {len(xkp1_ws)}")
				# Residual
				res = [xp - xw for xp, xw in zip(x_next_pred, xkp1_ws)]
				res_norm = float(_np.linalg.norm(_np.array(res)))
				LOG_INFO(f"  Dyn residual k={k}: pred={x_next_pred}, ws={xkp1_ws}, res={res}, res_norm={res_norm:.3e}")
				try:
					import logging as _logging
					_logging.getLogger("integration_test").info(f"Dyn residual k={k}: pred={x_next_pred}, ws={xkp1_ws}, res={res}, res_norm={res_norm:.3e}")
				except Exception:
					pass
		except Exception as _e:
			LOG_WARN(f"Dyn residual diag failed: {_e}")
			try:
				import logging as _logging
				_logging.getLogger("integration_test").warning(f"Dyn residual diag failed: {_e}")
			except Exception:
				pass

		# CRITICAL: Loop over all stages (0 to horizon) to collect objectives and constraints
		# State variables: k in [0, horizon] (horizon + 1 values)
		# Input variables: k in [0, horizon-1] (horizon values)
		LOG_INFO(f"Collecting objectives and constraints for {horizon_val + 1} stages (0 to {horizon_val})")
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			LOG_WARN("solve: cannot create symbolic states without dynamics_model")
			return -1
		
		for stage_idx in range(horizon_val + 1):
			# Create symbolic state for this stage (converted from data)
			# This is a symbolic representation for CasADi - actual state comes from data
			symbolic_state = State(dynamics_model)
			for var_name in dynamics_model.get_all_vars():
				if var_name in self.var_dict and stage_idx < self.var_dict[var_name].shape[0]:
					symbolic_state.set(var_name, self.var_dict[var_name][stage_idx])

			# CRITICAL: Ensure data is up-to-date before getting objectives/constraints
			# The state in data should be the current state (k=0), not the predicted state
			# For objectives/constraints at stage k, we use symbolic_state for CasADi variables
			# but data.state should be the current actual state from the planner
			if self.data is not None:
				LOG_DEBUG(f"Stage {stage_idx}: Using data.state={self.data.state is not None}, symbolic_state for CasADi vars")
			
			# Get objectives from module manager - it uses data as source of truth
			# Note: get_objective_cost will override state with data.state if available
			objective_costs = self.get_objective_cost(symbolic_state, stage_idx)
			# Log objectives BEFORE processing
			obj_count = len(objective_costs or [])
			if obj_count > 0:
				try:
					import logging as _logging
					_integ_logger = _logging.getLogger("integration_test")
					obj_keys = []
					obj_values = []
					for item in (objective_costs or []):
						if isinstance(item, dict):
							obj_keys.extend(list(item.keys()))
							for k, v in item.items():
								try:
									# Try to get numeric value if symbolic
									if hasattr(v, '__float__'):
										obj_values.append((k, float(v)))
									else:
										obj_values.append((k, 'symbolic'))
								except:
									obj_values.append((k, 'unknown'))
					msg_obj = f"Stage {stage_idx}: {obj_count} objective(s) with keys: {obj_keys}"
					if stage_idx <= 2 and obj_values:
						LOG_DEBUG(f"  Objective values: {obj_values[:5]}")  # Show first 5
					LOG_INFO(msg_obj)
					try:
						_integ_logger.info(msg_obj)
					except Exception:
						pass
				except Exception:
					pass
			else:
				if stage_idx <= 2:
					LOG_DEBUG(f"Stage {stage_idx}: No objectives")
			for cost_dict in objective_costs:
				for cost_val in cost_dict.values():
					total_objective += cost_val

			# Get constraints from module manager - it uses data as source of truth
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
							
							# Apply disc_offset to constraint (matching reference implementation)
							# Constraint should be applied to disc position: p_disc = p_robot + offset * [cos(psi), sin(psi)]
							# For constraint a·p_disc <= b, we have:
							#   a1*(x + offset*cos(psi)) + a2*(y + offset*sin(psi)) <= b
							# Which expands to: a1*x + a2*y + offset*(a1*cos(psi) + a2*sin(psi)) <= b
							# So: a1*x + a2*y <= b - offset*(a1*cos(psi) + a2*sin(psi))
							# For the expression a1*x + a2*y - b, we need to adjust b:
							#   a1*x + a2*y - b_adjusted <= 0
							#   a1*(x + offset*cos(psi)) + a2*(y + offset*sin(psi)) - b_adjusted <= 0
							#   a1*x + a2*y + offset*(a1*cos(psi) + a2*sin(psi)) - b_adjusted <= 0
							#   b_adjusted = b - offset*(a1*cos(psi) + a2*sin(psi))
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
							
							# Halfspace constraint for obstacle avoidance (matching reference implementation):
							# The normal vector (a1, a2) points FROM vehicle TO obstacle
							# To keep vehicle AWAY from obstacle, we need: a·p_vehicle <= b
							# Reference uses: constraint_expr = a1*x + a2*y - (b + slack)
							# With bounds: lb=-inf, ub=0, this enforces: a1*x + a2*y - b <= 0
							# So the constraint expression is: (a1*x + a2*y) - b_adjusted
							return (a1 * xk + a2 * yk) - b_adjusted
						# Unknown structured type; skip
						return None

					c_expr = _translate_constraint(c)
					if c_expr is None:
						continue

					lb_mx = _to_mx(lb)
					ub_mx = _to_mx(ub)

					try:
						# Log constraint details for stage 0 to diagnose issues
						if stage_idx == 0 and total_constraints_added < 10:
							if isinstance(c, dict):
								# Log constraint expression value for debugging
								try:
									if 'a1' in c and 'a2' in c:
										# This is a linearized constraint
										a1_val = float(c.get('a1', 0.0))
										a2_val = float(c.get('a2', 0.0))
										b_val = float(c.get('b', 0.0))
										# Get current state values for constraint evaluation
										if hasattr(self, 'data') and self.data is not None:
											if hasattr(self.data, 'state') and self.data.state is not None:
												x_val = float(self.data.state.get('x', 0.0))
												y_val = float(self.data.state.get('y', 0.0))
												constraint_val = a1_val * x_val + a2_val * y_val
												constraint_satisfied = constraint_val <= b_val
												LOG_DEBUG(f"  Stage 0 linearized constraint {total_constraints_added}: a1={a1_val:.3f}, a2={a2_val:.3f}, b={b_val:.3f}, "
												         f"at vehicle ({x_val:.2f}, {y_val:.2f}): {constraint_val:.3f} <= {b_val:.3f}? {constraint_satisfied}")
								except Exception as e:
									LOG_DEBUG(f"  Stage 0 constraint {total_constraints_added}: a1={c.get('a1', 'N/A')}, a2={c.get('a2', 'N/A')}, b={c.get('b', 'N/A')}, disc_offset={c.get('disc_offset', 'N/A')}, lb={lb}, ub={ub}, error={e}")
						
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
		
		# Log problem statistics before solving
		LOG_INFO("=== OPTIMIZATION PROBLEM SUMMARY ===")
		LOG_INFO(f"  Horizon: {horizon_val} steps")
		LOG_INFO(f"  Total constraints: {total_constraints_added}")
		
		# Count variables
		num_state_vars = sum(1 for v in dynamics_model.get_dependent_vars() if v in self.var_dict)
		num_input_vars = sum(1 for v in dynamics_model.get_inputs() if v in self.var_dict)
		total_state_vars = num_state_vars * (horizon_val + 1)
		total_input_vars = num_input_vars * horizon_val
		LOG_INFO(f"  Variables: {num_state_vars} state types × {horizon_val + 1} = {total_state_vars} state vars, "
		         f"{num_input_vars} input types × {horizon_val} = {total_input_vars} input vars")
		
		# Log objective structure
		obj_type = "scalar" if not hasattr(total_objective, 'shape') else f"array shape {total_objective.shape}"
		LOG_INFO(f"  Objective: {obj_type}")
		
		# Check warmstart feasibility
		LOG_DEBUG("Checking warmstart values...")
		try:
			for var_name in list(self.var_dict.keys())[:5]:  # Check first few
				if var_name in self.warmstart_values:
					ws_vals = self.warmstart_values[var_name]
					if isinstance(ws_vals, np.ndarray) and len(ws_vals) > 0:
						lb, ub, _ = dynamics_model.get_bounds(var_name)
						ws_min, ws_max = float(np.min(ws_vals)), float(np.max(ws_vals))
						in_bounds = (ws_min >= lb) and (ws_max <= ub)
						LOG_DEBUG(f"    Warmstart[{var_name}]: range=[{ws_min:.3f}, {ws_max:.3f}], bounds=[{lb:.3f}, {ub:.3f}], feasible={in_bounds}")
		except Exception as ws_err:
			LOG_DEBUG(f"    Could not check warmstart: {ws_err}")
		
		LOG_DEBUG(f"  Solving over horizon: {horizon_val} steps (states: 0 to {horizon_val}, controls: 0 to {horizon_val-1})")

		try:
			LOG_INFO("=== ATTEMPTING TO SOLVE ===")
			LOG_INFO("Calling IPOPT solver...")
			self.solution = self.opti.solve()
			self.exit_flag = 1
			self._update_warmstart_from_solution()
			
			# Check IPOPT stats if available
			try:
				stats = self.opti.stats()
				if stats:
					LOG_INFO("=== IPOPT SOLVER STATISTICS ===")
					LOG_INFO(f"  Return status: {stats.get('return_status', 'N/A')}")
					LOG_INFO(f"  Iterations: {stats.get('iter_count', 'N/A')}")
					LOG_INFO(f"  Objective value: {stats.get('obj', 'N/A')}")
					LOG_INFO(f"  Constraint violation: {stats.get('constr_viol', 'N/A')}")
					LOG_INFO(f"  Dual infeasibility: {stats.get('du_inf', 'N/A')}")
					LOG_INFO(f"  Primal infeasibility: {stats.get('pr_inf', 'N/A')}")
			except Exception as stats_err:
				LOG_DEBUG(f"Could not get IPOPT stats: {stats_err}")
			
			LOG_INFO("=== SOLUTION FOUND ===")
			LOG_INFO(f"  Solver successfully computed optimal trajectory")
			LOG_INFO(f"  Trajectory: {horizon_val + 1} states and {horizon_val} control inputs")
			
			# Log solution details
			if self.solution:
				LOG_DEBUG("Extracting solution values...")
				# Log first state values
				try:
					for var_name in dynamics_model.get_dependent_vars():
						if var_name in self.var_dict:
							val0 = self.solution.value(self.var_dict[var_name][0])
							LOG_DEBUG(f"  Solution[{var_name}][0] = {val0}")
					# Log first control values
					for var_name in dynamics_model.get_inputs():
						if var_name in self.var_dict:
							val0 = self.solution.value(self.var_dict[var_name][0])
							LOG_INFO(f"  Solution u[{var_name}][0] = {val0} (first control input to apply)")
				except Exception as e:
					LOG_DEBUG(f"  Could not extract solution values: {e}")
			
			return 1
		except RuntimeError as e:
			LOG_WARN("=== SOLVER FAILED ===")
			LOG_WARN(f"Error message: {e}")
			self.exit_flag = -1
			self.info["status"] = "failed"
			self.info["error"] = str(e)
			
			# Check IPOPT return status if available
			try:
				stats = self.opti.stats()
				if stats:
					return_status = stats.get('return_status', 'N/A')
					LOG_WARN(f"IPOPT return status: {return_status}")
					# Map common IPOPT status codes
					status_map = {
						'Maximum_Iterations_Exceeded': 'Solver hit maximum iterations - problem may be infeasible or difficult',
						'Infeasible_Problem_Detected': 'Problem is infeasible - constraints cannot be satisfied',
						'Search_Direction_Becomes_Too_Small': 'Solver cannot make progress - problem may be ill-conditioned',
						'Diverging_Iterates': 'Solution diverging - problem may be unbounded or infeasible',
						'Restoration_Failed': 'Restoration phase failed - initial point likely infeasible',
						'Error_In_Step_Computation': 'Error computing step - problem may be numerically difficult',
					}
					if return_status in status_map:
						LOG_WARN(f"  Interpretation: {status_map[return_status]}")
					
					LOG_WARN(f"  Final iteration: {stats.get('iter_count', 'N/A')}")
					LOG_WARN(f"  Final objective: {stats.get('obj', 'N/A')}")
					LOG_WARN(f"  Constraint violation: {stats.get('constr_viol', 'N/A')}")
					LOG_WARN(f"  Dual infeasibility: {stats.get('du_inf', 'N/A')}")
					LOG_WARN(f"  Primal infeasibility: {stats.get('pr_inf', 'N/A')}")
			except Exception as stats_err:
				LOG_DEBUG(f"Could not get IPOPT stats: {stats_err}")
			
			# Try to diagnose infeasibility
			LOG_WARN("=== DIAGNOSING FAILURE ===")
			LOG_WARN("Attempting to identify root cause...")
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
								# CRITICAL: Constraint violations array includes ALL constraints in order:
								# 1. Variable bounds (from opti.bounded() when creating variables)
								# 2. Initial state constraints (from _set_initial_state)
								# 3. Dynamics constraints (from intialize_solver)
								# 4. Module constraints (from solve loop)
								# The debug.g values are at the CURRENT point (likely infeasible initial values if solve failed)
								LOG_DEBUG(f"  Constraint violations array length: {len(constraint_violations)}")
								LOG_DEBUG(f"  Note: These values are at the current point (may be infeasible initial values)")
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
									if max_violation > 1.0:
										LOG_WARN(f"  SEVERE: Constraint violation > 1.0 - problem is highly infeasible")
									elif max_violation > 1e-3:
										LOG_WARN(f"  Problem likely infeasible - constraint violation too large")
									
									# Find which constraints are violated
									violated_indices = [i for i, v in enumerate(violations) if v > 1e-6]
									if violated_indices:
										LOG_WARN(f"  Violated constraints (indices): {violated_indices[:10]}")  # Show first 10
								else:
									# Fallback: use absolute values (may not be accurate for all constraint types)
									# But also check for positive violations (constraints that are actually violated)
									constraint_violations_arr = np.array(constraint_violations)
									violations_abs = np.abs(constraint_violations_arr)
									max_violation_abs = np.max(violations_abs)
									max_idx_abs = np.argmax(violations_abs)
									
									# Also check for positive violations (actual constraint violations)
									positive_violations = constraint_violations_arr[constraint_violations_arr > 1e-6]
									if len(positive_violations) > 0:
										max_violation_pos = np.max(positive_violations)
										# Find the index of the maximum positive violation in the original array
										positive_indices = np.where(constraint_violations_arr > 1e-6)[0]
										max_idx_pos = positive_indices[np.argmax(positive_violations)]
										LOG_WARN(f"  Found {len(positive_violations)} constraint(s) with positive values (violations)")
										LOG_WARN(f"  Maximum positive violation: {max_violation_pos:.6f} at constraint index {max_idx_pos}")
										LOG_WARN(f"    This is an ACTUAL constraint violation (constraint value > 0)")
										# Try to identify this constraint
										try:
											actual_val_pos = float(constraint_violations[max_idx_pos])
											LOG_WARN(f"    Constraint {max_idx_pos} value: {actual_val_pos:.6f}")
											# Identify constraint type
											# CRITICAL: Constraint order in CasADi debug.g is:
											# 1. Variable bounds (from opti.bounded() - 2 per variable: lb and ub)
											# 2. Initial state constraints (from _set_initial_state - 5 constraints)
											# 3. Dynamics constraints (from intialize_solver - 5 states × horizon = 50)
											# 4. Module constraints (from solve loop - 2 per stage × (horizon+1) = 22)
											# So the actual indexing is:
											# - Bounds: indices 0 to (num_vars * 2 - 1)
											# - Initial state: indices after bounds
											# - Dynamics: indices after initial state
											# - Module: indices after dynamics
											
											# Count variable bounds first
											dynamics_model = self._get_dynamics_model()
											num_state_vars = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
											num_input_vars = len(dynamics_model.get_inputs()) if dynamics_model else 2
											num_bounds = (num_state_vars * (horizon_val + 1) + num_input_vars * horizon_val) * 2  # 2 per variable (lb, ub)
											num_initial = num_state_vars  # 5
											num_dynamics = num_state_vars * horizon_val  # 50
											num_module = 2 * (horizon_val + 1)  # 22
											
											LOG_WARN(f"    Constraint indexing: bounds={num_bounds}, initial={num_initial}, dynamics={num_dynamics}, module={num_module}")
											
											if max_idx_pos < num_bounds:
												bound_idx = max_idx_pos
												var_idx = bound_idx // 2
												bound_type = "lower" if (bound_idx % 2 == 0) else "upper"
												LOG_WARN(f"    This is a variable bound constraint (index {bound_idx}, {bound_type} bound for variable {var_idx})")
											elif max_idx_pos < num_bounds + num_initial:
												initial_idx = max_idx_pos - num_bounds
												var_names = dynamics_model.get_dependent_vars() if dynamics_model else ['x', 'y', 'psi', 'v', 'spline']
												var_name = var_names[initial_idx] if initial_idx < len(var_names) else f'state[{initial_idx}]'
												LOG_WARN(f"    This is an initial state constraint: {var_name}[0]")
											elif max_idx_pos < num_bounds + num_initial + num_dynamics:
												module_idx = max_idx_pos - num_bounds - num_initial
												# Calculate actual constraints per stage from actual constraint count
												# This accounts for variable number of constraints per stage
												# Find which stage this constraint belongs to by counting constraints up to that point
												constraints_counted = 0
												stage_idx = 0
												constraint_idx_in_stage = 0
												found_stage = False
												for stage_check in range(horizon_val + 1):
													cons_check = self.get_constraints(stage_check)
													if module_idx < constraints_counted + len(cons_check):
														stage_idx = stage_check
														constraint_idx_in_stage = module_idx - constraints_counted
														found_stage = True
														break
													constraints_counted += len(cons_check)
												if not found_stage:
													# Fallback: use last stage if out of bounds
													stage_idx = horizon_val
													constraint_idx_in_stage = module_idx - constraints_counted
												LOG_WARN(f"    This is a module constraint at stage {stage_idx}, constraint {constraint_idx_in_stage} in that stage")
												# Try to get more details about this constraint
												try:
													if stage_idx < horizon_val + 1:
														cons_list = self.get_constraints(stage_idx)
														if constraint_idx_in_stage < len(cons_list):
															c, lb, ub = cons_list[constraint_idx_in_stage]
															if isinstance(c, dict):
																LOG_WARN(f"      Module constraint details: a1={c.get('a1', 'N/A')}, a2={c.get('a2', 'N/A')}, b={c.get('b', 'N/A')}, disc_offset={c.get('disc_offset', 'N/A')}")
															LOG_WARN(f"      Constraint bounds: lb={lb}, ub={ub}")
												except Exception as e:
													LOG_DEBUG(f"      Could not get constraint details: {e}")
											elif max_idx_pos < num_bounds + num_initial + num_dynamics + num_module:
												# This is a dynamics constraint (after module constraints)
												dynamics_idx = max_idx_pos - num_bounds - num_initial - num_module
												num_states = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
												stage_idx = dynamics_idx // num_states
												state_idx = dynamics_idx % num_states
												state_names = dynamics_model.get_dependent_vars()
												state_name = state_names[state_idx] if state_idx < len(state_names) else f"state[{state_idx}]"
												LOG_WARN(f"    This is a dynamics constraint: {state_name} at transition {stage_idx} -> {stage_idx + 1}")
												# Check warmstart values for this transition
												try:
													dt_check = self.timestep if self.timestep is not None else 0.1
													LOG_WARN(f"      Checking warmstart for {state_name} at transition {stage_idx}->{stage_idx+1} (timestep={dt_check})")
													LOG_WARN(f"      Warmstart values available: {list(self.warmstart_values.keys())}")
													if state_name in self.warmstart_values:
														ws_len = len(self.warmstart_values[state_name])
														LOG_WARN(f"      Warmstart array length for {state_name}: {ws_len}, need stage_idx={stage_idx}")
														if stage_idx < ws_len:
															x_k_ws = float(self.warmstart_values[state_name][stage_idx])
															x_kp1_ws = float(self.warmstart_values[state_name][stage_idx + 1]) if stage_idx + 1 < ws_len else None
															x_kp1_str = f"{x_kp1_ws:.6f}" if x_kp1_ws is not None else 'N/A'
															LOG_WARN(f"      Warmstart: {state_name}[{stage_idx}]={x_k_ws:.6f}, {state_name}[{stage_idx+1}]={x_kp1_str}")
															# Compute predicted value from warmstart
															if state_name == 'x' and 'v' in self.warmstart_values and 'psi' in self.warmstart_values:
																v_k_ws = float(self.warmstart_values['v'][stage_idx]) if stage_idx < len(self.warmstart_values['v']) else 0.0
																psi_k_ws = float(self.warmstart_values['psi'][stage_idx]) if stage_idx < len(self.warmstart_values['psi']) else 0.0
																x_pred_ws = x_k_ws + dt_check * v_k_ws * np.cos(psi_k_ws)
																violation_ws = x_kp1_ws - x_pred_ws if x_kp1_ws is not None else None
																LOG_WARN(f"      Computed: x_pred = {x_k_ws:.6f} + {dt_check:.4f} * {v_k_ws:.6f} * cos({psi_k_ws:.6f}) = {x_pred_ws:.6f}")
																if violation_ws is not None:
																	LOG_WARN(f"      Warmstart dynamics violation: x[{stage_idx+1}]_ws - x_pred = {x_kp1_ws:.6f} - {x_pred_ws:.6f} = {violation_ws:.6f}")
																	if abs(violation_ws) > 1e-3:
																		LOG_WARN(f"      WARMSTART IS NOT DYNAMICS-CONSISTENT! Violation: {violation_ws:.6f}")
																# Check what CasADi actually has for these variables
																try:
																	if 'x' in self.var_dict and stage_idx < self.var_dict['x'].shape[0]:
																		x_k_casadi = float(self.opti.debug.value(self.var_dict['x'][stage_idx]))
																		x_kp1_casadi = float(self.opti.debug.value(self.var_dict['x'][stage_idx + 1])) if stage_idx + 1 < self.var_dict['x'].shape[0] else None
																		if 'v' in self.var_dict and 'psi' in self.var_dict:
																			v_k_casadi = float(self.opti.debug.value(self.var_dict['v'][stage_idx])) if stage_idx < self.var_dict['v'].shape[0] else None
																			psi_k_casadi = float(self.opti.debug.value(self.var_dict['psi'][stage_idx])) if stage_idx < self.var_dict['psi'].shape[0] else None
																			if v_k_casadi is not None and psi_k_casadi is not None and x_kp1_casadi is not None:
																				x_pred_casadi = x_k_casadi + dt_check * v_k_casadi * np.cos(psi_k_casadi)
																				constraint_val_casadi = x_kp1_casadi - x_pred_casadi
																				LOG_WARN(f"      CasADi values: x[{stage_idx}]={x_k_casadi:.6f}, x[{stage_idx+1}]={x_kp1_casadi:.6f}, v[{stage_idx}]={v_k_casadi:.6f}, psi[{stage_idx}]={psi_k_casadi:.6f}")
																				LOG_WARN(f"      CasADi constraint value: x[{stage_idx+1}] - (x[{stage_idx}] + dt*v[{stage_idx}]*cos(psi[{stage_idx}])) = {constraint_val_casadi:.6f}")
																				LOG_WARN(f"      Warmstart vs CasADi: x[{stage_idx}] ws={x_k_ws:.6f} casadi={x_k_casadi:.6f}, x[{stage_idx+1}] ws={x_kp1_ws:.6f} casadi={x_kp1_casadi:.6f}")
																except Exception as e_casadi:
																	LOG_DEBUG(f"      Could not get CasADi values: {e_casadi}")
															elif state_name == 'y' and 'v' in self.warmstart_values and 'psi' in self.warmstart_values:
																v_k_ws = float(self.warmstart_values['v'][stage_idx]) if stage_idx < len(self.warmstart_values['v']) else 0.0
																psi_k_ws = float(self.warmstart_values['psi'][stage_idx]) if stage_idx < len(self.warmstart_values['psi']) else 0.0
																y_pred_ws = x_k_ws + dt_check * v_k_ws * np.sin(psi_k_ws)
																violation_ws = x_kp1_ws - y_pred_ws if x_kp1_ws is not None else None
																if violation_ws is not None:
																	LOG_WARN(f"      Warmstart dynamics violation: y[{stage_idx+1}]_ws - y_pred = {x_kp1_ws:.6f} - {y_pred_ws:.6f} = {violation_ws:.6f}")
																	if abs(violation_ws) > 1e-3:
																		LOG_WARN(f"      WARMSTART IS NOT DYNAMICS-CONSISTENT! Violation: {violation_ws:.6f}")
														else:
															LOG_WARN(f"      Stage index {stage_idx} out of bounds for warmstart array (length={ws_len})")
													else:
														LOG_WARN(f"      State name '{state_name}' not in warmstart_values")
												except Exception as e3:
													LOG_WARN(f"      Could not check warmstart dynamics: {e3}")
													import traceback
													LOG_DEBUG(f"      Traceback: {traceback.format_exc()}")
											else:
												# This is beyond all expected constraints, likely a bound constraint
												LOG_WARN(f"    This constraint is beyond expected range (index {max_idx_pos})")
										except Exception:
											pass
									
									LOG_WARN(f"  Maximum constraint value (abs): {max_violation_abs:.6f} at constraint index {max_idx_abs}")
									# Log the actual constraint value
									try:
										actual_val = float(constraint_violations[max_idx_abs])
										LOG_WARN(f"    Constraint {max_idx_abs} value: {actual_val:.6f}")
										# Try to identify which variable this constraint corresponds to
										# CasADi adds bounds as constraints after explicit constraints
										# Each variable has a lower and upper bound constraint
										explicit_constraints = 77  # initial (5) + module (22) + dynamics (50)
										if max_idx_abs >= explicit_constraints:
											bound_idx = max_idx_abs - explicit_constraints
											# Try to map to variable
											all_vars = []
											for vname in dynamics_model.get_dependent_vars():
												all_vars.extend([(vname, k) for k in range(horizon_val + 1)])
											for vname in dynamics_model.get_inputs():
												all_vars.extend([(vname, k) for k in range(horizon_val)])
											# Each variable has 2 bounds (lb, ub)
											if bound_idx < len(all_vars) * 2:
												var_idx = bound_idx // 2
												bound_type = "lower" if (bound_idx % 2 == 0) else "upper"
												if var_idx < len(all_vars):
													var_name, stage_k = all_vars[var_idx]
													LOG_WARN(f"    This is {bound_type} bound constraint for '{var_name}' at stage {stage_k}")
													# Get the actual value and bound
													try:
														if var_name in self.var_dict and stage_k < self.var_dict[var_name].shape[0]:
															actual_var_val = float(self.opti.debug.value(self.var_dict[var_name][stage_k]))
															lb, ub, _ = dynamics_model.get_bounds(var_name)
															bound_val = lb if bound_type == "lower" else ub
															# For upper bound: constraint is x - ub <= 0, so value < 0 means x < ub (satisfied)
															# For lower bound: constraint is lb - x <= 0, so value < 0 means x > lb (satisfied)
															# If value > 0, constraint is violated
															LOG_WARN(f"      Variable value: {actual_var_val:.6f}, bound: {bound_val:.6f}, constraint value: {actual_val:.6f}")
															if bound_type == "lower":
																# Lower bound constraint: lb - x <= 0, so x >= lb
																if actual_var_val < lb:
																	LOG_WARN(f"      VIOLATION: {var_name}[{stage_k}]={actual_var_val:.6f} < lb={lb:.6f}, violation={lb - actual_var_val:.6f}")
																elif actual_val > 0:
																	LOG_WARN(f"      CONSTRAINT VIOLATION: lb - x = {actual_val:.6f} > 0, meaning x={actual_var_val:.6f} < lb={lb:.6f}")
															elif bound_type == "upper":
																# Upper bound constraint: x - ub <= 0, so x <= ub
																if actual_var_val > ub:
																	LOG_WARN(f"      VIOLATION: {var_name}[{stage_k}]={actual_var_val:.6f} > ub={ub:.6f}, violation={actual_var_val - ub:.6f}")
																elif actual_val > 0:
																	LOG_WARN(f"      CONSTRAINT VIOLATION: x - ub = {actual_val:.6f} > 0, meaning x={actual_var_val:.6f} > ub={ub:.6f}")
															# Check warmstart value
															if var_name in self.warmstart_values and stage_k < len(self.warmstart_values[var_name]):
																ws_val = self.warmstart_values[var_name][stage_k]
																LOG_WARN(f"      Warmstart value: {ws_val:.6f}")
																if bound_type == "lower" and ws_val < lb:
																	LOG_WARN(f"      WARMSTART VIOLATION: {var_name}[{stage_k}]={ws_val:.6f} < lb={lb:.6f}")
																elif bound_type == "upper" and ws_val > ub:
																	LOG_WARN(f"      WARMSTART VIOLATION: {var_name}[{stage_k}]={ws_val:.6f} > ub={ub:.6f}")
													except Exception as e2:
														LOG_DEBUG(f"      Could not get variable value: {e2}")
									except Exception:
										pass
									# Try to identify which constraint this is
									try:
										horizon_val = self.horizon if self.horizon is not None else 10
										dynamics_model = self._get_dynamics_model()
										# Use max_idx_abs which is defined in outer scope
										idx_to_check = max_idx_abs if 'max_idx_abs' in locals() or 'max_idx_abs' in globals() else (len(constraint_violations) - 1 if constraint_violations else 0)
										# Initial state constraints: 5 states fixed at k=0
										num_initial_state = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
										# Dynamics constraints: 5 per transition (k=0 to k=horizon-1) = 5*horizon = 50 for horizon=10
										num_dynamics_constraints = len(dynamics_model.get_dependent_vars()) * horizon_val if dynamics_model else 5 * horizon_val
										# Module constraints: Count actual constraints from all stages
										# Note: Some constraint modules may not return constraints for stage_idx >= horizon_val
										num_module_constraints = 0
										for stage_check in range(horizon_val + 1):
											cons_check = self.get_constraints(stage_check)
											num_module_constraints += len(cons_check)
										# Constraints are added in order: initial state, then module constraints per stage, then dynamics constraints
										LOG_WARN(f"  Constraint indexing: initial={num_initial_state}, module={num_module_constraints}, dynamics={num_dynamics_constraints}, total expected={num_initial_state+num_module_constraints+num_dynamics_constraints}, actual={len(constraint_violations)}")
										if idx_to_check < num_initial_state:
											dep_vars = dynamics_model.get_dependent_vars()
											var_name = dep_vars[idx_to_check] if idx_to_check < len(dep_vars) else 'unknown'
											LOG_WARN(f"    Constraint {idx_to_check} is initial state constraint for {var_name}")
										elif idx_to_check < num_initial_state + num_module_constraints:
											module_idx = idx_to_check - num_initial_state
											# Find which stage this constraint belongs to by counting actual constraints
											constraints_counted = 0
											stage_idx = 0
											constraint_in_stage = 0
											found_stage = False
											for stage_check in range(horizon_val + 1):
												cons_check = self.get_constraints(stage_check)
												if module_idx < constraints_counted + len(cons_check):
													stage_idx = stage_check
													constraint_in_stage = module_idx - constraints_counted
													found_stage = True
													break
												constraints_counted += len(cons_check)
											if not found_stage:
												# Fallback: use last stage if out of bounds
												stage_idx = horizon_val
												constraint_in_stage = module_idx - constraints_counted
											LOG_WARN(f"    Constraint {idx_to_check} is module constraint at stage {stage_idx}, constraint #{constraint_in_stage}")
										elif idx_to_check < num_initial_state + num_module_constraints + num_dynamics_constraints:
											dyn_idx = idx_to_check - num_initial_state - num_module_constraints
											num_states = len(dynamics_model.get_dependent_vars()) if dynamics_model else 5
											transition_k = dyn_idx // num_states
											state_idx = dyn_idx % num_states
											dep_vars = dynamics_model.get_dependent_vars()
											state_name = dep_vars[state_idx] if state_idx < len(dep_vars) else 'unknown'
											LOG_WARN(f"    Constraint {idx_to_check} is dynamics constraint at transition k={transition_k}->{transition_k+1}, state={state_name}")
										else:
											LOG_WARN(f"    Constraint {idx_to_check} is beyond expected range (could be bounds or other constraints)")
									except Exception as _e:
										LOG_DEBUG(f"  Could not identify constraint: {_e}")
									# Check violation using max_violation_abs if available
									if 'max_violation_abs' in locals() or 'max_violation_abs' in globals():
										max_violation_check = max_violation_abs
									else:
										max_violation_check = max_violation_abs if 'max_violation_abs' in dir() else 0.0
									if max_violation_check > 1.0:
										LOG_WARN(f"  SEVERE: Constraint value > 1.0 - problem is highly infeasible")
									elif max_violation_check > 1e-3:
										LOG_WARN(f"  Problem likely infeasible - constraint violation too large")
									# Check dynamics constraint residuals explicitly
									try:
										horizon_val = self.horizon if self.horizon is not None else 10
										dynamics_model = self._get_dynamics_model()
										dep_vars = dynamics_model.get_dependent_vars() if dynamics_model else []
										nx = len(dep_vars)
										LOG_WARN(f"  Checking dynamics constraint residuals at k=0..2 (expecting {nx} states per stage):")
										for k_check in range(min(3, horizon_val)):
											try:
												# Get x_k and x_k+1 from solution
												x_k_vals = []
												x_kp1_vals = []
												for var in dep_vars:
													try:
														x_k_vals.append(float(self.opti.debug.value(self.var_dict[var][k_check])))
														x_kp1_vals.append(float(self.opti.debug.value(self.var_dict[var][k_check + 1])))
													except Exception:
														x_k_vals.append(0.0)
														x_kp1_vals.append(0.0)
												# Get u_k from solution
												in_vars = dynamics_model.get_inputs() if dynamics_model else []
												u_k_vals = []
												for var in in_vars:
													try:
														u_k_vals.append(float(self.opti.debug.value(self.var_dict[var][k_check])))
													except Exception:
														u_k_vals.append(0.0)
												# Compute predicted next state
												x_k_dm = cs.DM(x_k_vals)
												u_k_dm = cs.DM(u_k_vals if len(u_k_vals) > 0 else [0.0])
												def _pget(key):
													try:
														return self.data.parameters.get(key)
													except Exception:
														return 0.0
												x_next_pred_sym = dynamics_model.symbolic_dynamics(x_k_dm, u_k_dm, _pget, self.timestep)
												x_next_pred_num = np.array(x_next_pred_sym.full()).flatten()
												# Compute residual
												residual = np.array(x_kp1_vals) - x_next_pred_num
												res_norm = np.linalg.norm(residual)
												max_res = np.max(np.abs(residual))
												max_res_idx = np.argmax(np.abs(residual))
												LOG_WARN(f"    k={k_check}: residual norm={res_norm:.6f}, max={max_res:.6f} at {dep_vars[max_res_idx] if max_res_idx < len(dep_vars) else 'unknown'}")
												if max_res > 1.0:
													LOG_WARN(f"      Full residual: {residual}, x_k={x_k_vals}, x_kp1={x_kp1_vals}, u_k={u_k_vals}")
											except Exception as _e:
												LOG_DEBUG(f"    Could not check dynamics residual at k={k_check}: {_e}")
									except Exception as _e:
										LOG_DEBUG(f"  Could not check dynamics residuals: {_e}")
					except Exception as const_err:
						LOG_DEBUG(f"Could not check constraint violations: {const_err}")
			except Exception as debug_err:
				LOG_DEBUG(f"Could not get debug values: {debug_err}")
			
			# Log current state and data
			LOG_WARN("Current state at failure:")
			# Get state from data - this is the authoritative source
			current_state = None
			if self.data is not None and hasattr(self.data, 'state'):
				current_state = self.data.state
			elif state is not None:
				current_state = state
			dynamics_model = self._get_dynamics_model()
			if current_state and dynamics_model:
				try:
					state_vars = {}
					for v in dynamics_model.get_all_vars():
						if current_state.has(v):
							state_vars[v] = current_state.get(v)
					LOG_WARN(f"  State variables: {state_vars}")
					# Check if initial state might violate constraints
					if 'x' in state_vars and 'y' in state_vars:
						LOG_WARN(f"  Initial position: ({state_vars['x']:.2f}, {state_vars['y']:.2f})")
					# Check if state is within bounds
					for var_name, var_val in state_vars.items():
						try:
							lb, ub, _ = dynamics_model.get_bounds(var_name)
							if var_val < lb or var_val > ub:
								LOG_WARN(f"  WARNING: State[{var_name}]={var_val:.3f} is OUTSIDE bounds [{lb:.3f}, {ub:.3f}]")
						except Exception:
							pass
				except Exception as state_err:
					LOG_DEBUG(f"Could not log state: {state_err}")
			
			LOG_WARN("Data at failure:")
			if self.data:
				has_ref = hasattr(self.data, 'reference_path') and self.data.reference_path is not None
				has_static = hasattr(self.data, 'static_obstacles') and self.data.static_obstacles is not None
				has_dynamic = hasattr(self.data, 'dynamic_obstacles') and self.data.dynamic_obstacles is not None
				LOG_WARN(f"  reference_path: {'present' if has_ref else 'missing'}")
				LOG_WARN(f"  static_obstacles: {'present' if has_static else 'missing'}")
				LOG_WARN(f"  dynamic_obstacles: {'present' if has_dynamic else 'missing'}")
				
				# Log static obstacle details
				if has_static and hasattr(self.data, 'static_obstacles'):
					obs_count = sum(1 for o in self.data.static_obstacles if o is not None)
					LOG_WARN(f"  Static obstacles: {obs_count} non-None obstacles")
					if obs_count > 0 and state:
						try:
							x0 = state.get('x')
							y0 = state.get('y')
							if x0 is not None and y0 is not None:
								# Check first stage
								if len(self.data.static_obstacles) > 0:
									obs0 = self.data.static_obstacles[0]
									if obs0 is not None and hasattr(obs0, 'halfspaces'):
										LOG_WARN(f"  Stage 0 has {len(obs0.halfspaces)} halfspace constraints")
						except Exception:
							pass
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
			
			LOG_WARN("=== DIAGNOSIS COMPLETE ===")
			LOG_WARN("Summary of potential issues:")
			LOG_WARN("  1. Check if initial state violates constraints (especially road boundaries)")
			LOG_WARN("  2. Check if warmstart is feasible (within bounds)")
			LOG_WARN("  3. Check if constraints are too tight for the problem")
			LOG_WARN("  4. Check if reference path alignment is correct")
			LOG_WARN("  5. Check if obstacle constraints are reasonable")
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
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return Trajectory(timestep=timestep_val, length=0)
		traj = Trajectory(timestep=timestep_val, length=horizon_val + 1)
		for k in range(horizon_val + 1):
			state_k = State(model_type=dynamics_model)
			for var_name in dynamics_model.get_dependent_vars():
				state_k.set(var_name, self.solution.value(self.var_dict[var_name][k]))
			if k < horizon_val:
				for var_name in dynamics_model.get_inputs():
					state_k.set(var_name, self.solution.value(self.var_dict[var_name][k]))
			traj.add_state(state_k)
		return traj

	def print_if_bound_limited(self):
		if not self.solution: return
		LOG_DEBUG("Checking variable bounds:")
		dynamics_model = self._get_dynamics_model()
		if dynamics_model is None:
			return
		for var_name, var in self.var_dict.items():
			values = self.solution.value(var)
			lb, ub, _ = dynamics_model.get_bounds(var_name)
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
		"""Set the dynamics model in data - DO NOT store separately."""
		if self.data is None:
			LOG_WARN("set_dynamics_model: data is None, cannot set dynamics_model")
			return
		self.data.dynamics_model = model
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

		# No need to set dynamics_model - it should be in data

		for module in self.module_manager.get_modules():
			module.reset()