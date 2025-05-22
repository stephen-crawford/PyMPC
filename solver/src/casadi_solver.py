import casadi as cs
import numpy as np

from planning.src.dynamic_models import DynamicsModel
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
		self.info = {}

		# Ego prediction trajectory
		self.ego_prediction = Trajectory()

	def set_dynamics_model(self, dynamics_model: DynamicsModel):
		"""
		Incorporates the given dynamics model into the optimization problem.
		"""
		self.dynamics_model = dynamics_model

		for var_name in self.dynamics_model.get_vars():
			self.var_dict[var_name] = self.opti.variable(self.horizon + 1) # This makes a horizon + 1 length vector
			self.opti.set_initial(self.var_dict[var_name], 0.0)
			self.warmstart_values[var_name] = np.zeros(self.horizon + 1)

		for var_name in self.dynamics_model.get_inputs():
			self.var_dict[var_name] = self.opti.variable(self.horizon)
			self.opti.set_initial(self.var_dict[var_name], 0.0)
			self.warmstart_values[var_name] = np.zeros(self.horizon)

		LOG_DEBUG(f"After initialization, vars are: {self.var_dict}")
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
			for var_name in self.dynamics_model.get_vars():
				if var_name in self.warmstart_values:
					prev_values = self.warmstart_values[var_name]
					self.warmstart_values[var_name][:-1] = prev_values[1:]
					self.warmstart_values[var_name][-1] = prev_values[-1]

					for k in range(self.horizon + 1):
						self.opti.set_initial(self.var_dict[var_name][k], self.warmstart_values[var_name][k])

			for var_name in self.dynamics_model.get_inputs():
				if var_name in self.warmstart_values:
					if var_name in self.warmstart_values:
						prev_values = self.warmstart_values[var_name]
						self.warmstart_values[var_name][:-1] = prev_values[1:]
						self.warmstart_values[var_name][-1] = prev_values[-1]

						for k in range(self.horizon + 1):
							self.opti.set_initial(self.var_dict[var_name][k], self.warmstart_values[var_name][k])

	def solve(self):
		"""Solve the optimization problem

		Returns:
			int: Exit flag (1 for success, -1 for failure)
		"""

		try:
			# Get solver timeout parameter
			timeout = self.parameter_manager.get("solver_timeout")
			# Update solver options with timeout
			options = {}
			if timeout > 0:
				options['ipopt.max_cpu_time'] = timeout

			total_objective = 0
			for stage_idx in range(self.horizon):
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
				expr = total_objective
				LOG_WARN("Evaluating objective function at failure")
				LOG_WARN(cs.Function('obj', [self.opti.x], [expr])(self.opti.debug.value(self.opti.x)))
				index = 503
				LOG_WARN(
					f"offending x {self.opti.debug.x_describe(index)}, offending g: {self.opti.debug.g_describe(index)}")
				for i, var_name in enumerate(self.var_dict):
					var = self.var_dict[var_name]
					LOG_DEBUG(f"Initial guess for {var_name} = {self.opti.debug.value(var)}")

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
				if var_name in self.dynamics_model.get_vars():
					self.warmstart_values[var_name] = np.array(sol.value(self.var_dict[var_name]))
				elif var_name in self.dynamics_model.get_inputs():
					self.warmstart_values[var_name] = np.array(sol.value(self.var_dict[var_name]))

			LOG_DEBUG("Optimization solved successfully")
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
		var_length = self.horizon + 1 if var_name in self.dynamics_model.get_vars() else self.horizon
		if k < 0 or k >= var_length:
			return None
		# Return the value from the solution
		try:
			return float(self.solution.value(self.var_dict[var_name][k]))
		except:
			return None

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
		self.set_dynamics_model(self.dynamics_model)
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
			if stage_idx is not None and 0 <= stage_idx < self.horizon:
				return self.var_dict[var_name][stage_idx]
			return self.var_dict[var_name]
		return None

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

	def _apply_model_bounds(self):
		"""Apply bounds from the dynamics model to optimization variables"""

		# Apply state variable bounds
		for var_name in self.dynamics_model.get_vars():
			var = self.var_dict[var_name]
			lb, ub, _ = self.dynamics_model.get_bounds(var_name)

			for k in range(self.horizon + 1):
				self.opti.subject_to(self.opti.bounded(lb, var[k], ub))


		for var_name in self.dynamics_model.get_inputs():
			var = self.var_dict[var_name]
			lb, ub, _ = self.dynamics_model.get_bounds(var_name)

			for k in range(self.horizon):
				self.opti.subject_to(self.opti.bounded(lb, var[k], ub))

			if var_name == 'a':
				# Start with slight deceleration
				init_val = -0.5
			elif var_name == 'w':
				# Start with no steering
				init_val = 0.0
			else:
				init_val = (lb + ub) / 2.0

			# Clamp to bounds in case initial is outside of them -- this would only happen if acceleration cannot be neg. etc.
			init_val = max(lb, min(ub, init_val))
			self.opti.set_initial(var, init_val)

		self._add_dynamics_constraints()

	def _add_dynamics_constraints(self):
		for k in range(self.horizon):
			# Create input vector -- will end up being something like [a_0, a_1, a_2, ..., w_0, w_1, ...]
			u_k = []
			for u_name in self.dynamics_model.get_inputs():
				u_k.append(self.var_dict[u_name][k])

			# Create state vector -- ex: [x_0, x_1, ... y_0, y_1, ...]
			x_k = []
			for x_name in self.dynamics_model.get_vars():
				x_k.append(self.var_dict[x_name][k])

			# Combine into z_k [a_0, ... a_horizon-1, w_0, ..., w_horizon -1, x_0, ..., x_horizon-, etc.]
			z_k = cs.vertcat(*u_k, *x_k)

			# Load into dynamics model
			self.dynamics_model.load(z_k)

			# Symbolic predicted next state
			next_state = self.dynamics_model.discrete_dynamics(z_k, self.parameter_manager, self.timestep)

			# Add dynamics constraints
			for i, state_name in enumerate(self.dynamics_model.get_vars()):
				if (hasattr(self.dynamics_model, 'state_dimension_integrate') and
						self.dynamics_model.state_dimension_integrate is not None and
						i >= self.dynamics_model.state_dimension_integrate):
					continue
				LOG_DEBUG(f"Adding dynamics constraint {state_name} at {k + 1} must equal {next_state[i]}")
				self.opti.subject_to(self.var_dict[state_name][k + 1] == next_state[i])


	def _set_initial_state(self, state: State):
		"""
		Provide the initial state to the solver.
		"""
		# Extract initial state values using current model's variable names
		LOG_DEBUG("Setting initial state based on given state " + str(state))
		self.initial_state = state
		self.ego_prediction.add(state)
		for var_name in self.dynamics_model.get_vars():
			value = state.get(var_name)

			# If we have a valid value for this state variable
			if value is not None:
				# Constrain initial state
				self.opti.subject_to(self.var_dict[var_name][0] == value)

				# Also update the warmstart values to ensure consistency
				if var_name in self.warmstart_values:
					self.warmstart_values[var_name][0] = value

	def _initialize_base_rollout(self, state: State):
		"""Initialize with a braking trajectory"""

		LOG_DEBUG("Initializing base rollout")
		current_state = state.copy()  # Assuming State has a copy method
		self.ego_prediction = Trajectory()
		self.ego_prediction.add(current_state)

		psi = state.get('psi')

		for i in range(1, self.horizon + 1):
			prev_state = self.ego_prediction.get_states()[i - 1]
			prev_v = prev_state.get("v")
			prev_x = prev_state.get("x")
			prev_y = prev_state.get("y")
			prev_psi = prev_state.get("psi")

			a = -CONFIG["deceleration_at_infeasible"]
			v = self.get_ego_prediction(i, "v") + a * i * self.timestep
			a = ((v - self.get_ego_prediction(i, "v")) / (i * self.timestep))
			# Update velocity with deceleration
			v = max(prev_v + a * self.timestep, 0.0)

			# Update position
			x = prev_x + prev_v * self.timestep * np.cos(prev_psi)
			y = prev_y + prev_v * self.timestep * np.sin(prev_psi)

			state_i = State()
			state_i.x = x
			state_i.y = y
			state_i.psi = psi
			state_i.v = v
			state_i.a = a

			self.ego_prediction.add(state_i)

		for k in range(self.horizon + 1):
			pred_state = self.ego_prediction.get_states()[k]

			for var_name in self.dynamics_model.get_vars():
				if var_name in self.warmstart_values:
					value = pred_state.get(var_name)
					if value is not None:
						self.warmstart_values[var_name][k] = value
						# Set initial guess in optimizer
						self.opti.set_initial(self.var_dict[var_name][k], value)

		for k in range(self.horizon):
			for var_name in self.dynamics_model.get_inputs():
				if var_name in self.warmstart_values:
					if var_name in ['acceleration', 'a']:
						# Use the acceleration from prediction
						if k < len(self.ego_prediction.get_states()) - 1:
							next_v = self.ego_prediction.get_states()[k + 1].get('v')
							curr_v = self.ego_prediction.get_states()[k].get('v')
							a_value = (next_v - curr_v) / self.timestep
						else:
							a_value = -CONFIG["deceleration_at_infeasible"]

						self.warmstart_values[var_name][k] = a_value
						self.opti.set_initial(self.var_dict[var_name][k], a_value)
					else:
						# Other controls default to 0
						self.warmstart_values[var_name][k] = 0.0
						self.opti.set_initial(self.var_dict[var_name][k], 0.0)

		# Update ego prediction with a next state
		state = State()
		for var_name in self.dynamics_model.get_vars():
			if var_name in self.warmstart_values:
				state.set(var_name, self.warmstart_values[var_name])
		for var_name in self.dynamics_model.get_inputs():
			if var_name in self.warmstart_values:
				state.set(var_name, self.warmstart_values[var_name])
		self.ego_prediction.add(state)
