import casadi as cs
import numpy as np
from solver.src.base_solver import BaseSolver
from utils.const import CONSTRAINT, OBJECTIVE
from planning.src.types import State
from utils.utils import LOG_DEBUG, LOG_WARN


class CasADiSolver(BaseSolver):
	def __init__(self, timestep=0.1, horizon=30):
		super().__init__(timestep, horizon)

		# CasADi specific attributes
		self.opti = cs.Opti()  # CasADi Opti stack
		self.var_dict = {}  # Dictionary to store optimization variables
		self.param_dict = {}  # Dictionary to store optimization parameters

		# State and control dimensions
		self.nx = 6  # Default state dimension (x, y, psi, v, vx, vy)
		self.nu = 2  # Default control dimension (acceleration, steering)
		self.nvar = self.nx + self.nu  # Total variables per stage

		# Warmstart storage
		self.warmstart_values = {}

		# Solution storage
		self.solution = None
		self.exit_flag = None
		self.info = {}

		# Variables and parameters
		self.state_vars = ["x", "y", "psi", "v", "vx", "vy"]
		self.control_vars = ["acceleration", "steering"]
		self.var_names = self.state_vars + self.control_vars

		# Ego trajectory prediction
		self.ego_prediction = {}

		# Initialize with default settings
		self._initialize_problem()

	def _initialize_problem(self):
		"""Initialize the optimization problem structure"""
		print("Initializing problem...")
		print("State vars are: ", self.state_vars)

		# Clear existing variables to prevent duplicates
		self.var_dict.clear()

		# Create variables for each state across the horizon
		for var_name in self.state_vars:
			self.var_dict[var_name] = self.opti.variable(self.horizon + 1)
			# Initialize with zeros
			self.opti.set_initial(self.var_dict[var_name], 0.0)
			# Store for warmstart
			self.warmstart_values[var_name] = np.zeros(self.horizon + 1)
			# Store for ego prediction
			self.ego_prediction[var_name] = np.zeros(self.horizon + 1)

		print("VAR DICT NOW: ", self.var_dict)

		# Create variables for each control across the horizon
		for var_name in self.control_vars:
			self.var_dict[var_name] = self.opti.variable(self.horizon)
			# Initialize with zeros
			self.opti.set_initial(self.var_dict[var_name], 0.0)
			# Store for warmstart
			self.warmstart_values[var_name] = np.zeros(self.horizon)

		# Configure solver options
		opts = {
			'ipopt.print_level': 0,  # Suppress IPOPT output
			'print_time': 0,
			'ipopt.sb': 'yes',  # Suppress IPOPT banner
			'ipopt.max_iter': 500,
			'ipopt.tol': 1e-4
		}
		self.opti.solver('ipopt', opts)

	def set_dynamics_model(self, dynamics_model):
		"""Set the dynamics model for the optimization problem

		Args:
			dynamics_model: DynamicsModel object
		"""
		self.dynamics_model = dynamics_model

		# Update state and control dimensions based on the dynamics model
		self.nx = dynamics_model.nx
		self.nu = dynamics_model.nu
		self.nvar = self.nx + self.nu

		# Update variable names based on dynamics model
		self.state_vars = dynamics_model.get_vars()
		self.control_vars = dynamics_model.inputs
		self.var_names = self.state_vars + self.control_vars

		# Clear existing variables and recreate for consistency
		self.var_dict.clear()
		self.warmstart_values.clear()
		self.ego_prediction.clear()

		# Create parameter dictionary for dynamics model parameters
		self.param_dict["dynamics_params"] = self.opti.parameter()

		# Reinitialize the problem with the new variable structure
		self._initialize_problem()

		# Apply variable bounds from the dynamics model
		self._apply_model_bounds()

	def _apply_model_bounds(self):
		"""Apply bounds from the dynamics model to optimization variables"""
		if not hasattr(self, 'dynamics_model') or self.dynamics_model is None:
			return

		# Apply bounds to state variables
		print("state vars " + str(self.state_vars))
		print("control vars " + str(self.control_vars))
		for i, var_name in enumerate(self.state_vars):
			var = self.var_dict[var_name]
			lb, ub, _ = self.dynamics_model.get_bounds(var_name)

			# Apply bounds to all time steps
			for k in range(self.horizon + 1):
				self.opti.subject_to(self.opti.bounded(lb, var[k], ub))

		# Apply bounds to control variables
		for i, var_name in enumerate(self.control_vars):
			var = self.var_dict[var_name]
			lb, ub, _ = self.dynamics_model.get_bounds(var_name)

			# Apply bounds to all time steps
			for k in range(self.horizon):
				self.opti.subject_to(self.opti.bounded(lb, var[k], ub))

	def _setup_problem(self):
		"""Setup the optimization problem using module manager"""
		# Clear existing constraints
		self._initialize_problem()

		# Apply bounds from dynamics model if available
		self._apply_model_bounds()

		# Settings dictionary for modules
		settings = {
			"parameter_manager": self.parameter_manager,
			"horizon": self.horizon,
			"timestep": self.timestep,
			"N": self.horizon,  # For backward compatibility
			"dt": self.timestep,  # For dynamics model
			"params": self.parameter_manager  # For dynamics model
		}

		# Add dynamics constraints if dynamics model is set
		if hasattr(self, 'dynamics_model') and self.dynamics_model is not None:
			self._add_dynamics_constraints()

		# Set up objective function
		total_objective = 0
		for stage_idx in range(self.horizon):
			objective_value = self.get_objective_cost(0,
													  settings,
													  stage_idx
													  )

			# Ensure the objective is a scalar
			if isinstance(objective_value, cs.MX) and objective_value.shape[0] > 1:
				# Sum all elements if it's a vector
				objective_value = cs.sum1(objective_value)

			total_objective += objective_value

		# Set the problem objective
		print("Total objective is", total_objective)
		self.opti.minimize(total_objective)

		# Set up constraints
		for stage_idx in range(self.horizon):
			constraints = self.get_constraint_list(0,
												   settings,
												   stage_idx
												   )

			lower_bounds = self.get_constraint_lower_bounds_list()
			upper_bounds = self.get_constraint_upper_bounds_list()

			# Add constraints to the optimizer
			for i, constraint in enumerate(constraints):
				lb = lower_bounds[i] if i < len(lower_bounds) else -float('inf')
				ub = upper_bounds[i] if i < len(upper_bounds) else float('inf')

				if lb == ub:
					# Equality constraint
					self.opti.subject_to(constraint == lb)
				else:
					# Inequality constraint
					if lb > -float('inf'):
						self.opti.subject_to(constraint >= lb)
					if ub < float('inf'):
						self.opti.subject_to(constraint <= ub)

	def _add_dynamics_constraints(self):
		"""Add dynamics constraints to the optimization problem

		Args:
			settings: Dictionary of solver settings
		"""
		print("Adding dynamics constraints")

		# Create a parameter vector for the dynamics model if it doesn't exist
		if "dynamics_params" not in self.param_dict:
			self.param_dict["dynamics_params"] = self.opti.parameter()

		p = self.param_dict["dynamics_params"]

		# Apply dynamics constraints for each stage
		for k in range(self.horizon):
			# Create the state and control vector for the current stage
			z_k = []

			# First add control inputs (follows the order expected by DynamicsModel)
			for u_name in self.control_vars:
				z_k.append(self.var_dict[u_name][k])

			# Then add states
			for x_name in self.state_vars:
				z_k.append(self.var_dict[x_name][k])

			# Convert to CasADi vector
			z_k = cs.vertcat(*z_k)

			# Important: Make sure the dynamics model's _z is populated
			# This is the key fix - explicitly call load before discrete_dynamics
			self.dynamics_model.load(z_k)

			# Get next state from dynamics model
			x_next = self.dynamics_model.discrete_dynamics(z_k, p, self.timestep)

			# Constrain the next state in the optimization problem
			for i, state_name in enumerate(self.state_vars):
				# Skip states that shouldn't be integrated if specified in the model
				if hasattr(self.dynamics_model, 'nx_integrate') and \
						self.dynamics_model.nx_integrate is not None and \
						i >= self.dynamics_model.nx_integrate:
					continue

				# Add constraint: next state must match dynamics prediction
				self.opti.subject_to(self.var_dict[state_name][k + 1] == x_next[i])


	def reset(self):
		"""Reset the solver to its initial state."""
		# Recreate the Opti stack
		self.opti = cs.Opti()

		# Clear and reinitialize variables and parameters
		self.var_dict.clear()
		self.param_dict.clear()
		self.solution = None
		self.exit_flag = None
		self.info = {}

		# Reinitialize problem structure
		self._initialize_problem()

		# Reset module manager
		self.module_manager.reset_all()

		LOG_DEBUG("CasADi solver reset complete")

	def initialize_warmstart(self, state, shift_forward=True):
		"""Initialize solver with warmstart from previous solution

		Args:
			state: Current state object
			shift_forward: Whether to shift previous solution forward
		"""
		if self.solution is None:
			print("No previous solution found")
			# No previous solution, initialize with state
			self.initialize_base_rollout(state)
			return

		print("Previous solution found so updating")
		# Shift previous solution if requested
		if shift_forward:
			for var_name in self.state_vars:
				if var_name in self.warmstart_values:
					prev_values = self.warmstart_values[var_name]
					# Shift values by one step
					self.warmstart_values[var_name][:-1] = prev_values[1:]
					# Duplicate last value
					self.warmstart_values[var_name][-1] = prev_values[-1]

			for var_name in self.control_vars:
				if var_name in self.warmstart_values:
					prev_values = self.warmstart_values[var_name]
					if len(prev_values) > 1:
						# Shift values by one step
						self.warmstart_values[var_name][:-1] = prev_values[1:]
						# Duplicate last value
						self.warmstart_values[var_name][-1] = prev_values[-1]

		# Set first state values from current state
		# Make sure to use the variable names from state_vars (current model)
		print("populating with state values from curren state")
		for var_name in self.state_vars:
			if hasattr(state, "get") and state.has(var_name):
				self.warmstart_values[var_name][0] = state.get(var_name)
			elif hasattr(state, var_name):
				self.warmstart_values[var_name][0] = getattr(state, var_name)

		# Update ego prediction
		print("Updating ego prediction")
		for var_name in self.state_vars:
			if var_name in self.warmstart_values:
				self.ego_prediction[var_name] = self.warmstart_values[var_name].copy()

	def initialize_base_rollout(self, state):
		"""Initialize with a braking trajectory

		Args:
			state: Current state object
		"""
		# Initialize warmstart values arrays if they don't exist yet
		for var_name in self.state_vars:
			if var_name not in self.warmstart_values:
				self.warmstart_values[var_name] = np.zeros(self.horizon + 1)

		for var_name in self.control_vars:
			if var_name not in self.warmstart_values:
				self.warmstart_values[var_name] = np.zeros(self.horizon)

		# Get initial state values using current model's state variable names
		initial_values = {}
		for var_name in self.state_vars:
			if hasattr(state, "get") and state.has(var_name):
				initial_values[var_name] = state.get(var_name)
			elif hasattr(state, var_name):
				initial_values[var_name] = getattr(state, var_name)
			else:
				# Default to 0.0 if value not found
				initial_values[var_name] = 0.0

		# Extract common variables needed for trajectory calculation
		# For unicycle model, typical variables are x, y, heading/psi, v
		x_val = initial_values.get('x', 0.0)
		y_val = initial_values.get('y', 0.0)

		# Support both 'heading' and 'psi' for heading variable
		heading_var = 'psi' if 'psi' in self.state_vars else 'heading'
		heading_val = initial_values.get(heading_var, 0.0)

		v_val = initial_values.get('v', 0.0)

		# We might need velocity components
		vx_val = initial_values.get('vx', v_val * np.cos(heading_val) if v_val > 0.1 else 0.0)
		vy_val = initial_values.get('vy', v_val * np.sin(heading_val) if v_val > 0.1 else 0.0)

		# Create a simple braking trajectory
		decel = -1.0  # mild deceleration

		for k in range(self.horizon + 1):
			# Time at stage k
			t = k * self.timestep

			# Simple physics for braking
			v_at_t = max(0, v_val + decel * t)
			x_at_t = x_val + vx_val * t + 0.5 * decel * t * t * (vx_val / v_val if v_val > 0.1 else 0)
			y_at_t = y_val + vy_val * t + 0.5 * decel * t * t * (vy_val / v_val if v_val > 0.1 else 0)

			# Update warmstart values for all state variables
			for var_name in self.state_vars:
				if var_name == 'x':
					self.warmstart_values[var_name][k] = x_at_t
				elif var_name == 'y':
					self.warmstart_values[var_name][k] = y_at_t
				elif var_name in ['heading', 'psi']:
					self.warmstart_values[var_name][k] = heading_val
				elif var_name == 'v':
					self.warmstart_values[var_name][k] = v_at_t
				elif var_name == 'vx':
					self.warmstart_values[var_name][k] = v_at_t * np.cos(heading_val) if v_at_t > 0.1 else 0
				elif var_name == 'vy':
					self.warmstart_values[var_name][k] = v_at_t * np.sin(heading_val) if v_at_t > 0.1 else 0
				# For any other state variables, use zero or the initial value
				else:
					self.warmstart_values[var_name][k] = initial_values.get(var_name, 0.0)

			# Set control inputs if needed
			if k < self.horizon:
				for var_name in self.control_vars:
					# For acceleration-like controls
					if var_name in ['acceleration', 'a']:
						self.warmstart_values[var_name][k] = decel
					# For steering-like controls
					elif var_name in ['steering', 'w']:
						self.warmstart_values[var_name][k] = 0.0
					# For any other control variables
					else:
						self.warmstart_values[var_name][k] = 0.0

		# Update ego prediction
		for var_name in self.state_vars:
			if var_name in self.warmstart_values:
				self.ego_prediction[var_name] = self.warmstart_values[var_name].copy()

	def load_warmstart(self):
		"""Load warmstart values into the optimizer"""
		for var_name, values in self.warmstart_values.items():
			if var_name in self.var_dict:
				# Check for NaN values and replace them with zeros
				if isinstance(values, np.ndarray) and np.isnan(values).any():
					# Create a clean copy without NaN values
					clean_values = np.copy(values)
					clean_values[np.isnan(clean_values)] = 0.0
					self.opti.set_initial(self.var_dict[var_name], clean_values)
				else:
					# If no NaN values, proceed normally
					self.opti.set_initial(self.var_dict[var_name], values)

	def set_initial_state(self, state: State):
		"""Set initial state constraints

		Args:
			state: Initial state object
		"""
		# Extract initial state values using current model's variable names
		LOG_DEBUG("Setting initial state based on given state" + str(state))
		for var_name in self.state_vars:
			value = None

			# Try to get the value from state
			if hasattr(state, "get") and state.has(var_name):
				value = state.get(var_name)
			elif hasattr(state, var_name):
				value = getattr(state, var_name)

			# If we have a valid value for this state variable
			if value is not None:
				# Constrain initial state
				self.opti.subject_to(self.var_dict[var_name][0] == value)
				# Set initial guess
				self.opti.set_initial(self.var_dict[var_name][0], value)

				# Also update the warmstart values to ensure consistency
				if var_name in self.warmstart_values:
					self.warmstart_values[var_name][0] = value

	def solve(self):
		"""Solve the optimization problem

		Returns:
			int: Exit flag (1 for success, -1 for failure)
		"""
		# Set up the problem using the module manager

		try:
			# Get solver timeout parameter
			timeout = self.parameter_manager.get("solver_timeout")
			print("Solver timeout:", timeout)
			# Update solver options with timeout
			options = {}
			if timeout > 0:
				options['ipopt.max_cpu_time'] = timeout

			# Setup the problem using dynamics, objective, constraints
			self._setup_problem()

			# Solve the optimization problem
			sol = self.opti.solve()

			# Store solution
			self.solution = sol
			self.exit_flag = 1
			self.info["status"] = "success"

			# Update warmstart values for next iteration
			print("Var dict is ", self.var_dict)
			for var_name in self.var_dict:
				if var_name in self.state_vars:
					self.warmstart_values[var_name] = np.array(sol.value(self.var_dict[var_name]))
				elif var_name in self.control_vars:
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
		var_length = self.horizon + 1 if var_name in self.state_vars else self.horizon
		if k < 0 or k >= var_length:
			return None

		# Return the value from the solution
		try:
			return float(self.solution.value(self.var_dict[var_name][k]))
		except:
			return None

	def get_ego_prediction(self, k, var_name):
		"""Get ego prediction for visualization and warmstart

		Args:
			k (int): Time step index
			var_name (str): Variable name

		Returns:
			float: Predicted value at time step k
		"""
		if var_name in self.ego_prediction and k < len(self.ego_prediction[var_name]):
			return self.ego_prediction[var_name][k]
		return 0.0

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