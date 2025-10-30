import copy

import casadi as cs
import numpy as np

from planning.dynamic_models import DynamicsModel
from planning.types import State, Trajectory
from solver.base_solver import BaseSolver
from utils.const import OBJECTIVE, CONSTRAINT
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO

'''
Casadi solver used for trajectory optimization.
'''

class CasADiSolver(BaseSolver):
	def __init__(self, config):
		super().__init__(config)
		LOG_INFO(f"{(id(self))} CasADiSolver: Initializing solver")
		

	def intialize_solver(self, data):
		for var_name in data.dynamics_model.get_dependent_vars():
			self.var_dict[var_name] = self.opti.variable(data.horizon + 1)
			self.warmstart_values[var_name] = np.zeros(data.horizon + 1)
		for var_name in self.dynamics_model.get_inputs():
			self.var_dict[var_name] = self.opti.variable(data.horizon)
			self.warmstart_values[var_name] = np.zeros(data.horizon)
		opts = {
			'ipopt.print_level': 0,
			'print_time': 0,
			'ipopt.sb': 'yes',
			'ipopt.max_iter': 500,  # Reduced for faster convergence
			'ipopt.tol': 1e-3,  # Slightly relaxed for faster convergence
			'ipopt.acceptable_tol': 1e-2,  # Accept suboptimal solutions if close enough
			'ipopt.acceptable_iter': 15,  # Accept after 15 iterations if acceptable_tol met
			'ipopt.constr_viol_tol': 1e-4,  # Constraint violation tolerance
			'ipopt.mu_strategy': 'adaptive',
			'ipopt.hessian_approximation': 'limited-memory',
			'ipopt.warm_start_init_point': 'yes',
		}
		# Only set fast_step_computation if supported (may not be available in all IPOPT versions)
		try:
			opts['ipopt.fast_step_computation'] = 'yes'
		except:
			pass

		self.opti.solver('ipopt', opts)
		for k in range(data.horizon):
			x_k_list = [self.var_dict[var][k] for var in data.dynamics_model.get_dependent_vars()]
			u_k_list = [self.var_dict[var][k] for var in data.dynamics_model.get_inputs()]

			x_k = cs.vertcat(*x_k_list)
			u_k = cs.vertcat(*u_k_list)

			x_next_list = [self.var_dict[var][k + 1] for var in data.dynamics_model.get_dependent_vars()]
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
			x_next_pred = data.dynamics_model.symbolic_dynamics(x_k, u_k, _param_getter, data.timestep)

			self.opti.subject_to(x_next == x_next_pred)

	def initialize_rollout(self, state: State, shift_forward=True):
		self._set_initial_state(state)
		self._initialize_warmstart(state, shift_forward)

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

		# Use constant velocity profile (more realistic than braking)
		# Only apply light braking if velocity is high
		if current_vel > 5.0:
			v_profile = np.maximum(0.0, current_vel + DEFAULT_BRAKING * np.arange(self.horizon + 1) * self.timestep * 0.5)
		else:
			v_profile = np.full(self.horizon + 1, current_vel)
		
		psi_profile = np.full(self.horizon + 1, current_psi)  # Assume constant heading for initial guess

		self.warmstart_values['v'] = v_profile
		self.warmstart_values['psi'] = psi_profile

		# Integrate positions more accurately
		x_pos = np.zeros(self.horizon + 1)
		y_pos = np.zeros(self.horizon + 1)
		x_pos[0] = current_x
		y_pos[0] = current_y

		for k in range(1, self.horizon + 1):
			x_pos[k] = x_pos[k - 1] + v_profile[k - 1] * np.cos(psi_profile[k - 1]) * self.timestep
			y_pos[k] = y_pos[k - 1] + v_profile[k - 1] * np.sin(psi_profile[k - 1]) * self.timestep

		self.warmstart_values['x'] = x_pos
		self.warmstart_values['y'] = y_pos

		# Initialize other state variables if they exist
		for var_name in self.dynamics_model.get_dependent_vars():
			if var_name not in ['x', 'y', 'v', 'psi']:
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
		traj = Trajectory(timestep=self.timestep, length=self.horizon + 1)
		for k in range(self.horizon + 1):
			state_k = State(model_type=self.dynamics_model)
			for var_name in self.dynamics_model.get_dependent_vars():
				state_k.set(var_name, self.warmstart_values[var_name][k])
			if k < self.horizon:
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
		for var_name in self.dynamics_model.get_dependent_vars():
			value = state.get(var_name)
			if value is not None:
				self.opti.subject_to(self.var_dict[var_name][0] == value)
				self.warmstart_values[var_name][0] = value

	def solve(self):
		LOG_DEBUG("Attempting to solve in Casadi solver")
		total_objective = 0

		for stage_idx in range(self.horizon + 1):
			symbolic_state = State(self.dynamics_model)
			for var_name in self.dynamics_model.get_all_vars():
				if stage_idx < self.var_dict[var_name].shape[0]:
					symbolic_state.set(var_name, self.var_dict[var_name][stage_idx])

			objective_costs = self.get_objective_cost(symbolic_state, stage_idx)
			for cost_dict in objective_costs:
				for cost_val in cost_dict.values():
					total_objective += cost_val

			constraints = self.get_constraints(stage_idx)
			if constraints:
				for (c, lb, ub) in constraints:
					self.opti.subject_to(self.opti.bounded(lb, c, ub))

			# Collect penalty terms from modules using symbolic state
			for module in self.module_manager.get_modules():
				if hasattr(module, 'get_penalty'):
					penalty = module.get_penalty(symbolic_state, self.parameter_manager, stage_idx)
					if penalty is not None:
						total_objective += penalty

		self.opti.minimize(total_objective)

		try:
			self.solution = self.opti.solve()
			self.exit_flag = 1
			self._update_warmstart_from_solution()
			LOG_INFO("=== SOLUTION FOUND ===")
			return 1
		except RuntimeError as e:
			LOG_WARN(f"[ERROR] CasADi solver failed: {e}")
			self.exit_flag = -1
			self.info["status"] = "failed"
			self.info["error"] = str(e)
			return -1

	def get_output(self, k, var_name):
		if self.solution and var_name in self.var_dict:
			var = self.var_dict[var_name]
			if k < var.shape[0]:
				return self.solution.value(var[k])
		return None

	def get_reference_trajectory(self):
		if not self.solution:
			LOG_WARN("No solution available. Returning trajectory from warmstart.")
			return self._create_trajectory_from_warmstart()

		traj = Trajectory(timestep=self.timestep, length=self.horizon + 1)
		for k in range(self.horizon + 1):
			state_k = State(model_type=self.dynamics_model)
			for var_name in self.dynamics_model.get_dependent_vars():
				state_k.set(var_name, self.solution.value(self.var_dict[var_name][k]))
			if k < self.horizon:
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