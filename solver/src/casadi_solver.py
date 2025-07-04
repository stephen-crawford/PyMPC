import casadi as cs
import numpy as np

from planning.src.dynamic_models import DynamicsModel, numeric_rk4
from planning.src.types import State, Trajectory
from solver.src.base_solver import BaseSolver
from utils.utils import LOG_DEBUG, LOG_WARN, LOG_INFO, CONFIG

'''
Casadi solver used for trajectory optimization. 

Usage follows: 

1. Create solver object my_solver
2. Take a DynamicsModel and call my_solver.set_dynamics_model(model)
	> This will set the dynamics model for the solver and initialize the optimization variable dictionary to match
	> Will apply the upper and lower bounds for the state and control variables of the model as defined by the class 
3. Form an initial State instance for the trajectory you want to solve 
5. Call initialize_rollout(state, shift_forward)
6. Call solve()

'''
class CasADiSolver(BaseSolver):
	def __init__(self, timestep=0.1, horizon=30):
		super().__init__(timestep, horizon)

		# CasADi specific attributes
		self.opti = cs.Opti()  # CasADi Opti stack
		self.var_dict = {}  # Dictionary to store optimization variables

		self.dynamics_model = None

		self.warmstart_values = {}

		# Solution storage
		self.solution = None
		self.exit_flag = None
		self.info = {} # Used for delayed logging

		# Ego prediction trajectory
		self.ego_prediction = Trajectory()


	def set_dynamics_model(self, dynamics_model: DynamicsModel):
		"""
		Incorporates the given dynamics model into the optimization problem.
		"""

		self.dynamics_model = dynamics_model

		for var_name in self.dynamics_model.get_dependent_vars():
			self.var_dict[var_name] = self.opti.variable(self.horizon + 1) # This makes a horizon + 1 length symbolic vector
			self.warmstart_values[var_name] = np.zeros(self.horizon + 1)
		for var_name in self.dynamics_model.get_inputs():
			self.var_dict[var_name] = self.opti.variable(self.horizon)  # This makes a horizon length symbolic vector
			self.warmstart_values[var_name] = np.zeros(self.horizon)

		LOG_DEBUG(f"After setting dynamics model, vars are: {self.var_dict}")

		opts = {
			'ipopt.print_level': 0,
			'print_time': 0,
			'ipopt.sb': 'yes',
			'ipopt.max_iter': 1000,
			'ipopt.tol': 1e-4,
			'ipopt.mu_strategy': 'adaptive',
			'ipopt.hessian_approximation': 'limited-memory',
			'ipopt.warm_start_init_point': 'yes'
		}
		self.opti.solver('ipopt', opts)

		self._apply_model_bounds()

	def initialize_rollout(self, state: State, shift_forward=True):
		"""Initialize solver with warmstart from previous solution

		Args:
			state: Current state object
			shift_forward: Whether to shift previous solution forward
		"""
		self._set_initial_state(state)

		if self.solution is None:
			self._initialize_base_rollout(state)
			return

		# Shift previous solution if requested
		if shift_forward:
			LOG_DEBUG("Previous solution found so shifting forward")
			for var_name in self.dynamics_model.get_dependent_vars():
				if var_name in self.warmstart_values.keys():
					self.shift_warmstart(var_name)

					for k in range(self.horizon + 1):
						self.opti.set_initial(self.var_dict[var_name][k], self.warmstart_values[var_name][k])

			for var_name in self.dynamics_model.get_inputs():
				if var_name in self.warmstart_values.keys():
					self.shift_warmstart(var_name)

					for k in range(self.horizon):
						self.opti.set_initial(self.var_dict[var_name][k], self.warmstart_values[var_name][k])

	def _initialize_base_rollout(self, state: State):
		"""Initialize with a braking trajectory using proper numerical integration"""

		LOG_DEBUG("Initializing base rollout with ego prediction {}".format(self.ego_prediction))

		for i in range(1, self.horizon + 1):
			prev_state = self.ego_prediction.get_states()[i - 1]
			LOG_DEBUG("Current state when rolling out: {}".format(prev_state))

			# Get previous state variables
			prev_v = prev_state.get("v")
			prev_x = prev_state.get("x")
			prev_y = prev_state.get("y")
			prev_a = prev_state.get("a")
			prev_w = prev_state.get("w")
			prev_psi = prev_state.get("psi")

			# FIXED: Proper velocity integration (single timestep, not cumulative)
			v = prev_v + prev_a * self.timestep

			# FIXED: Use current velocity for position integration (or use average velocity)
			# Option 1: Use average velocity (more accurate for large timesteps)
			v_avg = (prev_v + v) / 2.0
			x = prev_x + v_avg * self.timestep * np.cos(prev_psi)
			y = prev_y + v_avg * self.timestep * np.sin(prev_psi)

			# FIXED: Update heading with angular velocity
			psi = prev_psi + prev_w * self.timestep

			# Normalize angle to [-pi, pi]
			psi = np.arctan2(np.sin(psi), np.cos(psi))

			# Create new state
			state_i = State(prev_state.get_model_type())
			state_i.set("x", x)
			state_i.set("y", y)
			state_i.set("v", v)
			state_i.set("psi", psi)
			state_i.set("a", prev_a)  # Keep same acceleration (or update if needed)
			state_i.set("w", prev_w)  # Keep same angular velocity (or update if needed)

			LOG_DEBUG("Adding state {} to ego prediction at step {}".format(state_i, i))
			self.ego_prediction.add_state(state_i)

		# Initialize warmstart values
		for k in range(self.horizon):
			pred_state = self.ego_prediction.get_states()[k]

			for var_name in self.dynamics_model.get_all_vars():
				if var_name in self.warmstart_values:
					value = pred_state.get(var_name)
					LOG_DEBUG("Loading {} from warmstart for stage {} value is {}".format(var_name, k, value))
					if value is not None:
						self.opti.set_initial(self.var_dict[var_name][k], value)

		# Debug logging
		LOG_DEBUG(f"Initial trajectory for horizon {self.horizon}:")
		for i, state in enumerate(self.ego_prediction.get_states()):
			if i <= self.horizon:
				LOG_DEBUG("state: {} at trajectory step {}".format(state, i))

	def _set_initial_state(self, state: State):
		"""
		Provide the initial state to the solver.
		"""
		# Extract initial state values using current model's variable names
		LOG_DEBUG("Setting initial state based on given state " + str(state))
		self.initial_state = state
		self.ego_prediction.add_state(state)
		for var_name in self.dynamics_model.get_dependent_vars():
			value = state.get(var_name)

			# If we have a valid value for this state variable
			if value is not None:
				# Constrain initial state
				self.opti.subject_to(self.var_dict[var_name][0] == value)

				# Also update the warmstart values to ensure consistency
				if var_name in self.warmstart_values.keys():
					self.warmstart_values[var_name][0] = value

	def _apply_model_bounds(self):
		"""Apply bounds from the dynamics model to optimization variables"""

		# Apply state variable bounds
		for var_name in self.dynamics_model.get_dependent_vars():
			var = self.var_dict[var_name]
			lb, ub, _ = self.dynamics_model.get_bounds(var_name)

			for k in range(self.horizon + 1):
				self.opti.subject_to(self.opti.bounded(lb, var[k], ub))


		for var_name in self.dynamics_model.get_inputs():
			var = self.var_dict[var_name]
			lb, ub, _ = self.dynamics_model.get_bounds(var_name)

			for k in range(self.horizon):
				self.opti.subject_to(self.opti.bounded(lb, var[k], ub))

		self._add_dynamics_constraints()

	def _add_dynamics_constraints(self):
		initial_ng = self.opti.ng
		for k in range(self.horizon):
			# Create input vector -- will end up being something like [a_0, a_1, a_2, ..., w_0, w_1, ...]
			u_k = []
			for u_name in self.dynamics_model.get_inputs():
				u_k.append(self.var_dict[u_name][k])

			# Create state vector -- ex: [x_0, x_1, ... y_0, y_1, ...]
			x_k = []
			for x_name in self.dynamics_model.get_dependent_vars():
				x_k.append(self.var_dict[x_name][k])

			# Combine into z_k [a_0, ... a_horizon-1, w_0, ..., w_horizon -1, x_0, ..., x_horizon-, etc.]
			z_k = cs.vertcat(*u_k, *x_k)

			# Load into dynamics model
			self.dynamics_model.load(z_k)

			# Symbolic predicted next state
			next_state = self.dynamics_model.discrete_dynamics(z_k, self.parameter_manager, self.timestep)

			# Add dynamics constraints
			for i, var in enumerate(self.dynamics_model.get_dependent_vars()):
				if (hasattr(self.dynamics_model, 'state_dimension_integrate') and
						self.dynamics_model.state_dimension_integrate is not None and
						i >= self.dynamics_model.state_dimension_integrate):
					continue
				self.opti.subject_to(self.var_dict[var][k + 1] == next_state[i])
				LOG_DEBUG(f"Added constraint {var}[{k + 1}] == {next_state[i]}")
			final_ng = self.opti.ng
			LOG_DEBUG(f"Dynamics constraints added: {final_ng - initial_ng}")


	def solve(self):
		"""Solve the optimization problem

		Returns:
			int: Exit flag (1 for success, -1 for failure)
		"""
		LOG_DEBUG("Attempting to solve in Casadi solver")
		try:
			# Get solver timeout parameter
			timeout = self.parameter_manager.get("solver_timeout")
			# Update solver options with timeout
			options = {}
			if timeout > 0:
				options['ipopt.max_cpu_time'] = timeout

			total_objective = 0
			for stage_idx in range(self.horizon + 1):
				LOG_DEBUG("Trying to get objective cost for stage " + str(stage_idx))
				objective_value = self.get_objective_cost(stage_idx)

				# Ensure the objective is a scalar
				if isinstance(objective_value, cs.MX) and objective_value.shape[0] > 1:
					# Sum all elements if it's a vector
					objective_value = cs.sum1(objective_value)

				total_objective += objective_value

			# Set the problem objective
			LOG_DEBUG("Total objective is" + str(total_objective))
			LOG_WARN(f"Total number of constraints is {self.opti.g.numel()}")
			self.opti.minimize(total_objective)

			# Solve the optimization problem
			LOG_DEBUG("Attempting to solve, " + str(self.opti))
			try:
				sol = self.opti.solve()
			except RuntimeError as e:

				LOG_WARN(f"[ERROR] CasADi solver failed at iteration: {self.opti.stats()['iter_count']} ")
				LOG_WARN("[ERROR] Exception:" + str(e))

				# Print all decision variable values at failure
				for i in range(self.opti.nx):
					LOG_WARN(f"x[{i}] = {self.opti.debug.value(self.opti.x[i])}")

				# You can also log constraint values here
				for i in range(self.opti.ng):
					LOG_WARN(f"g[{i}] = {self.opti.debug.value(self.opti.g[i])}")

				LOG_WARN("Solver failed. Dumping debug values...")
				for name in self.var_dict:
					try:
						val = self.opti.debug.value(self.var_dict[name])
						LOG_WARN(f"{name}: {val}")
					except RuntimeError:
						LOG_WARN(f"Could not extract value for {name}")

				raise

			# Store solution
			self.solution = sol
			self.exit_flag = 1
			self.info["status"] = "success"

			# Update warmstart values for next iteration
			LOG_DEBUG(f"Var dict is {self.var_dict}")
			for var_name in self.var_dict:
				if var_name in self.dynamics_model.get_dependent_vars():
					self.warmstart_values[var_name] = np.array(sol.value(self.var_dict[var_name]))
				elif var_name in self.dynamics_model.get_inputs():
					self.warmstart_values[var_name] = np.array(sol.value(self.var_dict[var_name]))

			LOG_DEBUG("Optimization solved successfully")
			LOG_INFO("=== SOLUTION FOUND ===")
			for var_name, var_vector in self.var_dict.items():
				try:
					values = self.solution.value(var_vector)
					values = np.round(values, decimals=4)  # Round for cleaner logs
					LOG_INFO(f"{var_name}: {values}")
				except Exception as e:
					LOG_WARN(f"Failed to log variable {var_name}: {e}")
			return 1
		except Exception as e:
			LOG_WARN(f"Solver failed: {str(e)}")
			self.exit_flag = -1
			self.info["status"] = "failed"
			self.info["error"] = str(e)
			return -1

	def get_output(self, k, var_name):
		"""Get the output value for a specific variable at time step k

		Args:
			k (int): Time step index (0 to horizon)
			var_name (str): Variable name

		Returns:
			float: Variable value at time step k, or None if unavailable
		"""
		if self.solution is None:
			return None

		# Check if the variable exists
		if var_name not in self.var_dict:
			return None

		# Check if k is valid for the variable
		var_length = self.horizon + 1 if var_name in self.dynamics_model.get_dependent_vars() else self.horizon
		if k < 0 or k >= var_length:
			return None
		# Return the value from the solution
		try:
			return float(self.solution.value(self.var_dict[var_name][k]))
		except:
			return None

	def get_reference_trajectory(self):
		"""Reconstruct the trajectory from the optimization result.

		Returns:
			Trajectory: A trajectory object containing states and controls from the solver output.
		"""
		traj = Trajectory(timestep=self.timestep, length=self.horizon + 1)

		if self.solution is None:
			LOG_WARN("No solution found. Cannot extract reference trajectory.")
			return traj

		# Reconstruct state trajectory
		for k in range(self.horizon + 1):
			state_k = State(model_type=self.dynamics_model)
			for var_name in self.dynamics_model.get_dependent_vars():
				try:
					value = float(self.solution.value(self.var_dict[var_name][k]))
					state_k.set(var_name, value)
				except Exception as e:
					LOG_WARN(f"Failed to extract value for {var_name}[{k}]: {e}")
					state_k.set(var_name, 0.0)
			for var_name in self.dynamics_model.get_inputs():
				if k in range(self.horizon):
					try:
						value = float(self.solution.value(self.var_dict[var_name][k]))
						state_k.set(var_name, value)
					except Exception as e:
						LOG_WARN(f"Failed to extract value for {var_name}[{k}]: {e}")
						state_k.set(var_name, 0.0)
			traj.add_state(state_k)

		return traj

	def print_if_bound_limited(self):
		"""Print which variables are at their bounds (for debugging)"""
		if self.solution is None:
			return

		LOG_DEBUG("Checking variable bounds:")
		for var_name, var in self.var_dict.items():
			# Get solution values
			try:
				values = self.solution.value(var)
				if isinstance(values, (float, int)):
					values = [values]  # Convert to list for consistency

				# Check for values close to typical bounds
				for i, val in enumerate(values):
					if abs(val) > 1e6:
						LOG_WARN(f"Variable {var_name}[{i}] has very large value: {val}")
			# You can add more specific bound checks here
			except:
				pass

	def explain_exit_flag(self, code=None):
		"""Explain the solver exit code

		Args:
			code (int, optional): Exit code to explain

		Returns:
			str: Human-readable explanation
		"""
		code = code if code is not None else self.exit_flag

		explanations = {
			1: "Optimization successful",
			-1: "Optimization failed" + (
				f": {self.info.get('error', '')}" if hasattr(self, 'info') and 'error' in self.info else ""),
			-2: "Problem is infeasible",
			-3: "Maximum iterations reached",
			-10: "Solver timeout reached"
		}

		return explanations.get(code, f"Unknown exit code: {code}")


	def reset(self):
		"""Reset the solver to its initial state."""

		# Clear and reinitialize variables and parameters
		self.opti = cs.Opti()

		self.solution = None
		self.exit_flag = None
		self.info = {}
		self.ego_prediction.reset()
		self.var_dict.clear()
		LOG_DEBUG("After reset, var dict is: " + str(self.var_dict.items()))
		self.set_dynamics_model(self.dynamics_model)
		for module in self.module_manager.modules:
			module.reset()
		LOG_DEBUG("CasADi solver reset.")

	# For compatibility with the model interface expected by modules
	def load(self, z):
		"""Compatibility method for BaseSolver's static methods
		In CasADi, this is a no-op as we're directly using the solver instance.
		"""
		pass

	def get(self, var_name, stage_idx=None):
		"""Get a variable value for the module interface

		Args:
			var_name (str): Name of the variable
			stage_idx (int, optional): Time stage (ignored, handled in the modules)

		Returns:
			CasADi symbolic variable
		"""
		if var_name in self.var_dict:
			if stage_idx is not None:
				var_length = self.horizon + 1 if var_name in self.dynamics_model.get_dependent_vars() else self.horizon
				if 0 <= stage_idx < var_length:
					return self.var_dict[var_name][stage_idx]
				else:
					LOG_WARN(f"Access out-of-bounds: {var_name}[{stage_idx}]")
					return None  # or throw
			return self.var_dict[var_name]

	def get_ego_prediction(self, k, var):
		LOG_DEBUG(f"Trying to get ego prediction for variable {var}, at {k}")
		if len(self.ego_prediction.states) <= k:
			return self.ego_prediction.get_states()[-1].get(var)
		LOG_WARN(f"Returning ego prediction {self.ego_prediction.get_states()[k].get(var)}")
		return self.ego_prediction.get_states()[k].get(var)

	def copy(self):
		"""Create a copy of the solver (for solver management)

		Returns:
			E
		"""
		# Create a new solver with same parameters
		new_solver = CasADiSolver(self.timestep, self.horizon)

		# Copy module manager
		new_solver.module_manager = self.module_manager

		# Copy parameter manager
		new_solver.parameter_manager = self.parameter_manager.copy()

		return new_solver

	def shift_warmstart(self, var_name):
		prev_values = self.warmstart_values[var_name]
		self.warmstart_values[var_name][:-1] = prev_values[1:]
		self.warmstart_values[var_name][-1] = prev_values[-1]