"""
Base class for optimization solvers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseSolver(ABC):
	"""
    Abstract base class for optimization solvers.

    All solvers should inherit from this class and implement
    the required methods for solving optimization problems.
    """

	def __init__(self, **kwargs):
		"""
        Initialize the solver.

        Args:
            **kwargs: Solver-specific parameters
        """
		self.solver_options = kwargs
		self.solution = None
		self.solve_time = 0.0
		self.iterations = 0
		self.status = "not_solved"

	@abstractmethod
	def setup_problem(self,
					  state_dim: int,
					  control_dim: int,
					  horizon_length: int,
					  objective_function: callable,
					  constraints: list,
					  **kwargs) -> None:
		"""
        Set up the optimization problem.

        Args:
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            horizon_length: Prediction horizon length
            objective_function: Objective function to minimize
            constraints: List of constraint functions
            **kwargs: Additional problem parameters
        """
		pass

	@abstractmethod
	def solve(self, initial_state: np.ndarray, **kwargs) -> Dict[str, Any]:
		"""
        Solve the optimization problem.

        Args:
            initial_state: Initial state vector
            **kwargs: Additional solve parameters

        Returns:
            Dictionary containing solution information
        """
		pass

	def get_solution(self) -> Optional[Dict[str, Any]]:
		"""
        Get the last solution.

        Returns:
            Dictionary containing solution information or None if not solved
        """
		return self.solution

	def get_solve_time(self) -> float:
		"""
        Get the time taken for the last solve.

        Returns:
            Solve time in seconds
        """
		return self.solve_time

	def get_iterations(self) -> int:
		"""
        Get the number of iterations for the last solve.

        Returns:
            Number of iterations
        """
		return self.iterations

	def get_status(self) -> str:
		"""
        Get the status of the last solve.

        Returns:
            Status string
        """
		return self.status

	def is_feasible(self) -> bool:
		"""
        Check if the last solution is feasible.

        Returns:
            True if feasible, False otherwise
        """
		return self.status in ["optimal", "feasible"]

	def reset(self) -> None:
		"""
        Reset the solver state.
        """
		self.solution = None
		self.solve_time = 0.0
		self.iterations = 0
		self.status = "not_solved"
