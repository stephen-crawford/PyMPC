"""
MPC planner implementation.

This module contains the main MPC planner that coordinates between
dynamics models, solvers, and modules (constraints/objectives).
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from .dynamics import BaseDynamics, create_dynamics_model
from .solver import BaseSolver, create_solver


class Planner:

    
    def __init__(self, solver):
        """
        Initialize MPC planner.
        
        Args:
            dynamics: Dynamics model
            N: Prediction horizon length
            dt: Time step
            solver_type: Type of solver to use
            solver_options: Solver options
        """

        self.dynamics = None
        self.N = None
        self.dt = None
        
        # Create solver
        self.solver = solver
        
        # Modules
        self.objectives: List[Any] = []
        self.constraints: List[Any] = []
        
        # State
        self.current_state: Optional[np.ndarray] = None
        self.last_solution: Optional[Dict[str, Any]] = None
        self.solve_count: int = 0
        self.success_count: int = 0
        
        # Timing
        self.total_solve_time: float = 0.0
        self.avg_solve_time: float = 0.0

    def build_problem(self, dynamics: BaseDynamics, N: int, dt: float, objectives: List[Any], constraints: List[Any]) -> None:
        self.dynamics = dynamics
        self.N = N
        self.dt = dt
        self.objectives = objectives
        self.constraints = constraints

        self.solver.set_dt(self.dt)
        self.solver.set_N(self.N)
        self.solver.modules_manager.add_modules(objectives)
        self.solver.modules_manager.add_modules(constraints)
    
    def solve(self, x0: np.ndarray, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Solve the MPC problem.
        
        Args:
            x0: Initial state
            **kwargs: Additional parameters
            
        Returns:
            Solution dictionary or None if failed
        """
        self.current_state = x0.copy()
        self.solve_count += 1
        
        # Set up optimization problem
        self._setup_optimization_problem(x0, **kwargs)
        
        # Solve
        start_time = time.time()
        success, result = self.solver.solve(x0, **kwargs)
        solve_time = time.time() - start_time
        
        # Update statistics
        self.total_solve_time += solve_time
        self.avg_solve_time = self.total_solve_time / self.solve_count
        
        if success:
            self.success_count += 1
            self.last_solution = result
            result['solve_time'] = solve_time
            return result
        else:
            print(f"MPC solve failed: {result.get('error', 'Unknown error')}")
            return None
    
    def _setup_optimization_problem(self, x0: np.ndarray, **kwargs) -> None:
        """
        Set up the optimization problem with objectives and constraints.
        
        Args:
            x0: Initial state
            **kwargs: Additional parameters
        """
        # Reset solver
        self.solver.reset()
        
        # Add objectives
        total_objective = None
        for k in range(self.N):
            # Get variables for this time step
            x_k = self.solver.x_vars[:, k]
            u_k = self.solver.u_vars[:, k]
            
            # Add objectives from modules
            for objective in self.objectives:
                if hasattr(objective, 'add_objective'):
                    obj = objective.add_objective(x_k, u_k, k, **kwargs)
                    if total_objective is None:
                        total_objective = obj
                    else:
                        total_objective += obj
        
        # Add constraints
        for k in range(self.N):
            x_k = self.solver.x_vars[:, k]
            u_k = self.solver.u_vars[:, k]
            
            for constraint in self.constraints:
                if hasattr(constraint, 'add_constraints'):
                    constraints = constraint.add_constraints(x_k, u_k, k, **kwargs)
                    for constr in constraints:
                        self.solver.add_constraint(constr)
        
        # Set objective
        if total_objective is not None:
            self.solver.add_objective(total_objective)
    
    def get_solution(self, k: int, var_name: str) -> float:
        """
        Get solution value for variable at step k.
        
        Args:
            k: Time step
            var_name: Variable name
            
        Returns:
            Solution value
        """
        return self.solver.get_solution(k, var_name)
    
    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get planned trajectory.
        
        Returns:
            (x_trajectory, y_trajectory)
        """
        return self.solver.get_trajectory()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get solver statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'solve_count': self.solve_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(self.solve_count, 1),
            'total_solve_time': self.total_solve_time,
            'avg_solve_time': self.avg_solve_time
        }
    
    def reset(self) -> None:
        """Reset planner state."""
        self.current_state = None
        self.last_solution = None
        self.solve_count = 0
        self.success_count = 0
        self.total_solve_time = 0.0
        self.avg_solve_time = 0.0
        
        # Reset solver
        self.solver.reset()
        
        # Reset modules
        for objective in self.objectives:
            if hasattr(objective, 'reset'):
                objective.reset()
        
        for constraint in self.constraints:
            if hasattr(constraint, 'reset'):
                constraint.reset()
    
    def visualize(self, **kwargs) -> None:
        """
        Visualize planner state and computations.
        
        Args:
            **kwargs: Visualization options
        """
        print(f"MPC Planner Statistics:")
        print(f"  Solve count: {self.solve_count}")
        print(f"  Success count: {self.success_count}")
        print(f"  Success rate: {self.success_count / max(self.solve_count, 1):.2%}")
        print(f"  Average solve time: {self.avg_solve_time:.4f}s")
        
        if self.current_state is not None:
            print(f"  Current state: {self.current_state}")
        
        if self.last_solution is not None:
            print(f"  Last solution cost: {self.last_solution.get('cost', 'N/A')}")
        
        # Visualize modules
        for objective in self.objectives:
            if hasattr(objective, 'visualize'):
                objective.visualize(**kwargs)
        
        for constraint in self.constraints:
            if hasattr(constraint, 'visualize'):
                constraint.visualize(**kwargs)


class PlannerBuilder:
    """Builder class for creating MPC planners with modules."""
    
    def __init__(self, dynamics_type: str = "bicycle", N: int = 20, 
                 dt: float = 0.1, solver_type: str = "casadi"):
        """
        Initialize builder.
        
        Args:
            dynamics_type: Type of dynamics model
            N: Prediction horizon length
            dt: Time step
            solver_type: Type of solver
        """
        self.dynamics_type = dynamics_type
        self.N = N
        self.dt = dt
        self.solver_type = solver_type
        self.solver_options = {}
        
        # Modules
        self.objectives = []
        self.constraints = []
    
    def set_solver_options(self, options: Dict[str, Any]) -> 'PlannerBuilder':
        """
        Set solver options.
        
        Args:
            options: Solver options
            
        Returns:
            Self for chaining
        """
        self.solver_options.update(options)
        return self
    
    def add_objective(self, objective: Any) -> 'PlannerBuilder':
        """
        Add objective function.
        
        Args:
            objective: Objective function module
            
        Returns:
            Self for chaining
        """
        self.objectives.append(objective)
        return self
    
    def add_constraint(self, constraint: Any) -> 'PlannerBuilder':
        """
        Add constraint.
        
        Args:
            constraint: Constraint module
            
        Returns:
            Self for chaining
        """
        self.constraints.append(constraint)
        return self
    
    def build(self) -> Planner:
        """
        Build the MPC planner.
        
        Returns:
            Configured MPC planner
        """
        # Create dynamics model
        dynamics = create_dynamics_model(self.dynamics_type, dt=self.dt)
        
        # Create planner
        planner = Planner(
            dynamics=dynamics,
            N=self.N,
            dt=self.dt,
            solver_type=self.solver_type,
            solver_options=self.solver_options
        )
        
        # Add modules
        for objective in self.objectives:
            planner.add_objective(objective)
        
        for constraint in self.constraints:
            planner.add_constraint(constraint)
        
        return planner


def create_mpc_planner(dynamics_type: str = "bicycle", 
                      N: int = 20,
                      dt: float = 0.1,
                      solver_type: str = "casadi",
                      solver_options: Optional[Dict[str, Any]] = None) -> Planner:
    """
    Create an MPC planner with default configuration.
    
    Args:
        dynamics_type: Type of dynamics model
        N: Prediction horizon length
        dt: Time step
        solver_type: Type of solver
        solver_options: Solver options
        
    Returns:
        Configured MPC planner
    """
    # Create dynamics model
    dynamics = create_dynamics_model(dynamics_type, dt=dt)
    
    # Create planner
    planner = Planner(
        dynamics=dynamics,
        N=N,
        dt=dt,
        solver_type=solver_type,
        solver_options=solver_options or {}
    )
    
    return planner
