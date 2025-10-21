"""
Main MPC planner implementation.

This module contains the main MPCCPlanner class that orchestrates
the MPC optimization with contouring control and scenario constraints.
"""

import numpy as np
import casadi as ca
from typing import List, Dict, Any, Optional, Tuple, Callable
import time

from .dynamics import BaseDynamics
from .objectives import BaseObjective, ContouringObjective, GoalObjective
from .constraints import BaseConstraint
from .solver import CasADiSolver


class MPCCPlanner:
	"""
  Model Predictive Contouring Control Planner.

  This planner combines contouring control with scenario constraints
  to provide robust path following in uncertain environments.
  """

	def __init__(self,
				 dynamics: BaseDynamics,
				 horizon_length: int = 20,
				 dt: float = 0.1,
				 solver_options: Optional[Dict[str, Any]] = None):
		"""
	Initialize the MPC planner.

	Args:
		dynamics: Vehicle dynamics model
		horizon_length: Prediction horizon length
		dt: Time step
		solver_options: Options for the optimization solver
	"""
		self.dynamics = dynamics
		self.horizon_length = horizon_length
		self.dt = dt
		self.solver_options = solver_options or {}

		# Initialize components
		self.objectives: List[BaseObjective] = []
		self.constraints: List[BaseConstraint] = []
		self.solver = CasADiSolver(**self.solver_options)

		# Problem setup status
		self.problem_setup = False
		self.last_solution = None

	def add_objective(self, objective: BaseObjective) -> None:
		"""
	Add an objective function to the planner.

	Args:
		objective: Objective function to add
	"""
		self.objectives.append(objective)
		self.problem_setup = False  # Need to re-setup problem

	def add_constraint(self, constraint: BaseConstraint) -> None:
		"""
	Add a constraint to the planner.

	Args:
		constraint: Constraint to add
	"""
		self.constraints.append(constraint)
		self.problem_setup = False  # Need to re-setup problem

	def remove_objective(self, index: int) -> None:
		"""
	Remove an objective function by index.

	Args:
		index: Index of objective to remove
	"""
		if 0 <= index < len(self.objectives):
			del self.objectives[index]
			self.problem_setup = False

	def remove_constraint(self, index: int) -> None:
		"""
	Remove a constraint by index.

	Args:
		index: Index of constraint to remove
	"""
		if 0 <= index < len(self.constraints):
			del self.constraints[index]
			self.problem_setup = False

	def setup_problem(self) -> None:
		"""
	Set up the optimization problem.
	"""
		if not self.objectives:
			raise ValueError("At least one objective function must be added")

		# Create combined objective function
		def combined_objective(X, U, opti):
			total_objective = 0
			for obj in self.objectives:
				if obj.is_active():
					total_objective += obj.compute_casadi(X, U, opti)
			return total_objective

		# Create constraint functions
		constraint_functions = []
		for constraint in self.constraints:
			if constraint.is_active():
				constraint_functions.append(
					lambda X, U, opti, c=constraint: c.add_to_opti(X, U, opti)
				)

		# Add dynamics constraints
		def dynamics_constraint(X, U, opti):
			for k in range(self.horizon_length):
				x_k = X[:, k]
				u_k = U[:, k]
				x_next = X[:, k + 1]

				# Add dynamics constraint: x_{k+1} = f(x_k, u_k)
				# This is a simplified version - in practice, you might want
				# to use the actual dynamics function
				dynamics_expr = self._get_dynamics_expression(x_k, u_k, opti)
				opti.subject_to(x_next == dynamics_expr)

		constraint_functions.append(dynamics_constraint)

		# Set up the solver
		self.solver.setup_problem(
			state_dim=self.dynamics.get_state_dimension(),
			control_dim=self.dynamics.get_control_dimension(),
			horizon_length=self.horizon_length,
			objective_function=combined_objective,
			constraints=constraint_functions
		)

		self.problem_setup = True

	def _get_dynamics_expression(self, x: ca.MX, u: ca.MX, opti: ca.Opti) -> ca.MX:
		"""
	Get CasADi expression for dynamics.

	Args:
		x: Current state
		u: Current control
		opti: CasADi Opti object

	Returns:
		Next state expression
	"""
		# This is a simplified version - in practice, you would implement
		# the actual dynamics using CasADi functions
		# For now, we'll use a simple linear dynamics model

		# Get linearized dynamics matrices
		A, B = self.dynamics.get_jacobian(
			np.zeros(self.dynamics.get_state_dimension()),
			np.zeros(self.dynamics.get_control_dimension())
		)

		# Discretize
		A_d, B_d = self.dynamics.discretize(A, B)

		return A_d @ x + B_d @ u

	def solve(self,
			  initial_state: np.ndarray,
			  reference_path: Optional[np.ndarray] = None,
			  **kwargs) -> Dict[str, Any]:
		"""
	Solve the MPC optimization problem.

	Args:
		initial_state: Initial state vector
		reference_path: Reference path for contouring control
		**kwargs: Additional solve parameters

	Returns:
		Dictionary containing solution information
	"""
		if not self.problem_setup:
			self.setup_problem()

		# Update reference path if provided
		if reference_path is not None:
			for obj in self.objectives:
				if isinstance(obj, ContouringObjective):
					obj.set_reference_path(reference_path)

		# Solve the optimization problem
		start_time = time.time()
		solution = self.solver.solve(initial_state, **kwargs)
		solve_time = time.time() - start_time

		# Store solution
		self.last_solution = solution
		solution['solve_time'] = solve_time

		return solution

	def get_solution(self) -> Optional[Dict[str, Any]]:
		"""
	Get the last solution.

	Returns:
		Last solution or None if not solved
	"""
		return self.last_solution

	def get_optimal_control(self) -> Optional[np.ndarray]:
		"""
	Get the optimal control sequence.

	Returns:
		Optimal control sequence or None if not solved
	"""
		if self.last_solution is None:
			return None

		return self.last_solution.get('controls')

	def get_optimal_trajectory(self) -> Optional[np.ndarray]:
		"""
	Get the optimal state trajectory.

	Returns:
		Optimal state trajectory or None if not solved
	"""
		if self.last_solution is None:
			return None

		return self.last_solution.get('states')

	def is_feasible(self) -> bool:
		"""
	Check if the last solution is feasible.

	Returns:
		True if feasible, False otherwise
	"""
		if self.last_solution is None:
			return False

		return self.last_solution.get('status') in ['optimal', 'feasible']

	def get_solve_time(self) -> float:
		"""
	Get the time taken for the last solve.

	Returns:
		Solve time in seconds
	"""
		if self.last_solution is None:
			return 0.0

		return self.last_solution.get('solve_time', 0.0)

	def reset(self) -> None:
		"""
	Reset the planner state.
	"""
		self.problem_setup = False
		self.last_solution = None
		self.solver.reset()

	def update_dynamics(self, dynamics: BaseDynamics) -> None:
		"""
	Update the dynamics model.

	Args:
		dynamics: New dynamics model
	"""
		self.dynamics = dynamics
		self.problem_setup = False

	def set_horizon_length(self, horizon_length: int) -> None:
		"""
	Set the prediction horizon length.

	Args:
		horizon_length: New horizon length
	"""
		self.horizon_length = horizon_length
		self.problem_setup = False

	def get_horizon_length(self) -> int:
		"""
	Get the current horizon length.

	Returns:
		Current horizon length
	"""
		return self.horizon_length

	def get_objective_count(self) -> int:
		"""
	Get the number of objective functions.

	Returns:
		Number of objectives
	"""
		return len(self.objectives)

	def get_constraint_count(self) -> int:
		"""
	Get the number of constraints.

	Returns:
		Number of constraints
	"""
		return len(self.constraints)

	def get_iterations(self) -> int:
		"""
	Get the number of iterations from the last solve.

	Returns:
		Number of iterations
	"""
		if hasattr(self.solver, 'get_iterations'):
			return self.solver.get_iterations()
		return 0

	def get_status(self) -> str:
		"""
	Get the status from the last solve.

	Returns:
		Status string
	"""
		if hasattr(self.solver, 'get_status'):
			return self.solver.get_status()
		return "unknown"
