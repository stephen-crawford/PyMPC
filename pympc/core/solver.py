"""
Optimization solvers for MPC.

This module provides solver backends for the MPC optimization problem.
"""

import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Callable


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


class CasADiSolver(BaseSolver):
    """
    CasADi-based solver for MPC optimization.

    This solver uses CasADi's Opti interface for nonlinear optimization.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CasADi solver.

        Args:
            **kwargs: Solver options
        """
        super().__init__(**kwargs)
        self.opti = None
        self.state_dim = None
        self.control_dim = None
        self.horizon_length = None
        self.objective_function = None
        self.constraints = None

    def setup_problem(self,
                      state_dim: int,
                      control_dim: int,
                      horizon_length: int,
                      objective_function: callable,
                      constraints: list,
                      **kwargs) -> None:
        """
        Set up the CasADi optimization problem.

        Args:
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            horizon_length: Prediction horizon length
            objective_function: Objective function to minimize
            constraints: List of constraint functions
            **kwargs: Additional problem parameters
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon_length = horizon_length
        self.objective_function = objective_function
        self.constraints = constraints

        # Create Opti instance
        self.opti = ca.Opti()

        # Decision variables
        X = self.opti.variable(state_dim, horizon_length + 1)
        U = self.opti.variable(control_dim, horizon_length)

        # Set up objective
        objective = objective_function(X, U, self.opti)
        self.opti.minimize(objective)

        # Add constraints
        for constraint_func in constraints:
            constraint_func(X, U, self.opti)

        # Set up solver
        solver_options = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-6,
            'print_time': 0
        }
        solver_options.update(self.solver_options.get('ipopt', {}))

        self.opti.solver('ipopt', solver_options)

    def solve(self, initial_state: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Solve the optimization problem.

        Args:
            initial_state: Initial state vector
            **kwargs: Additional solve parameters

        Returns:
            Dictionary containing solution information
        """
        if self.opti is None:
            raise RuntimeError("Problem not set up. Call setup_problem() first.")

        # Set initial state constraint
        self.opti.set_initial(self.opti.variable(self.state_dim, self.horizon_length + 1)[:, 0], initial_state)
        self.opti.subject_to(self.opti.variable(self.state_dim, self.horizon_length + 1)[:, 0] == initial_state)

        # Solve
        import time
        start_time = time.time()
        
        try:
            sol = self.opti.solve()
            self.solve_time = time.time() - start_time
            self.status = "optimal"
            
            # Extract solution
            X_opt = sol.value(self.opti.variable(self.state_dim, self.horizon_length + 1))
            U_opt = sol.value(self.opti.variable(self.control_dim, self.horizon_length))
            
            self.solution = {
                'states': X_opt,
                'controls': U_opt,
                'status': self.status,
                'solve_time': self.solve_time,
                'iterations': sol.stats()['iter_count'] if 'iter_count' in sol.stats() else 0
            }
            
            return self.solution
            
        except Exception as e:
            self.solve_time = time.time() - start_time
            self.status = "failed"
            self.solution = {
                'states': None,
                'controls': None,
                'status': self.status,
                'solve_time': self.solve_time,
                'error': str(e)
            }
            return self.solution

    def reset(self) -> None:
        """
        Reset the solver state.
        """
        super().reset()
        self.opti = None
        self.state_dim = None
        self.control_dim = None
        self.horizon_length = None
        self.objective_function = None
        self.constraints = None
