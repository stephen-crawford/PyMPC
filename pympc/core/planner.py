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


class MPCCPlanner:
    """
    Model Predictive Contouring Control (MPCC) planner.
    
    This planner coordinates between dynamics models, solvers, and modules
    to solve MPC problems with contouring control capabilities.
    """
    
    def __init__(self, dynamics: BaseDynamics, horizon_length: int = 20, 
                 dt: float = 0.1, solver_type: str = "casadi",
                 solver_options: Optional[Dict[str, Any]] = None):
        """
        Initialize MPCC planner.
        
        Args:
            dynamics: Dynamics model
            horizon_length: Prediction horizon length
            dt: Time step
            solver_type: Type of solver to use
            solver_options: Solver options
        """
        self.dynamics = dynamics
        self.horizon_length = horizon_length
        self.dt = dt
        
        # Create solver
        self.solver = create_solver(
            solver_type, dynamics, horizon_length, dt, 
            **(solver_options or {})
        )
        
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
    
    def add_objective(self, objective: Any) -> None:
        """
        Add objective function.
        
        Args:
            objective: Objective function module
        """
        self.objectives.append(objective)
    
    def add_constraint(self, constraint: Any) -> None:
        """
        Add constraint.
        
        Args:
            constraint: Constraint module
        """
        self.constraints.append(constraint)
    
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
        for k in range(self.horizon_length):
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
        for k in range(self.horizon_length):
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


class MPCPlannerBuilder:
    """Builder class for creating MPC planners with modules."""
    
    def __init__(self, dynamics_type: str = "bicycle", horizon_length: int = 20, 
                 dt: float = 0.1, solver_type: str = "casadi"):
        """
        Initialize builder.
        
        Args:
            dynamics_type: Type of dynamics model
            horizon_length: Prediction horizon length
            dt: Time step
            solver_type: Type of solver
        """
        self.dynamics_type = dynamics_type
        self.horizon_length = horizon_length
        self.dt = dt
        self.solver_type = solver_type
        self.solver_options = {}
        
        # Modules
        self.objectives = []
        self.constraints = []
    
    def set_solver_options(self, options: Dict[str, Any]) -> 'MPCPlannerBuilder':
        """
        Set solver options.
        
        Args:
            options: Solver options
            
        Returns:
            Self for chaining
        """
        self.solver_options.update(options)
        return self
    
    def add_objective(self, objective: Any) -> 'MPCPlannerBuilder':
        """
        Add objective function.
        
        Args:
            objective: Objective function module
            
        Returns:
            Self for chaining
        """
        self.objectives.append(objective)
        return self
    
    def add_constraint(self, constraint: Any) -> 'MPCPlannerBuilder':
        """
        Add constraint.
        
        Args:
            constraint: Constraint module
            
        Returns:
            Self for chaining
        """
        self.constraints.append(constraint)
        return self
    
    def build(self) -> MPCCPlanner:
        """
        Build the MPC planner.
        
        Returns:
            Configured MPC planner
        """
        # Create dynamics model
        dynamics = create_dynamics_model(self.dynamics_type, dt=self.dt)
        
        # Create planner
        planner = MPCCPlanner(
            dynamics=dynamics,
            horizon_length=self.horizon_length,
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
                      horizon_length: int = 20,
                      dt: float = 0.1,
                      solver_type: str = "casadi",
                      solver_options: Optional[Dict[str, Any]] = None) -> MPCCPlanner:
    """
    Create an MPC planner with default configuration.
    
    Args:
        dynamics_type: Type of dynamics model
        horizon_length: Prediction horizon length
        dt: Time step
        solver_type: Type of solver
        solver_options: Solver options
        
    Returns:
        Configured MPC planner
    """
    # Create dynamics model
    dynamics = create_dynamics_model(dynamics_type, dt=dt)
    
    # Create planner
    planner = MPCCPlanner(
        dynamics=dynamics,
        horizon_length=horizon_length,
        dt=dt,
        solver_type=solver_type,
        solver_options=solver_options or {}
    )
    
    return planner


def create_contouring_mpc(reference_path: np.ndarray,
                         dynamics_type: str = "contouring_bicycle",
                         horizon_length: int = 20,
                         dt: float = 0.1,
                         contouring_weight: float = 2.0,
                         lag_weight: float = 1.0,
                         progress_weight: float = 1.5,
                         road_width: float = 6.0,
                         safety_margin: float = 0.5) -> MPCCPlanner:
    """
    Create an MPC planner configured for contouring control.
    
    Args:
        reference_path: Reference path as Nx2 array
        dynamics_type: Type of dynamics model
        horizon_length: Prediction horizon length
        dt: Time step
        contouring_weight: Weight for contouring error
        lag_weight: Weight for lag error
        progress_weight: Weight for progress
        road_width: Road width
        safety_margin: Safety margin
        
    Returns:
        Configured MPC planner for contouring control
    """
    # Create planner
    planner = create_mpc_planner(
        dynamics_type=dynamics_type,
        horizon_length=horizon_length,
        dt=dt
    )
    
    # Add contouring objective (placeholder - would need actual implementation)
    # contouring_obj = ContouringObjective(
    #     contouring_weight=contouring_weight,
    #     lag_weight=lag_weight,
    #     progress_weight=progress_weight
    # )
    # contouring_obj.set_reference_path(reference_path)
    # planner.add_objective(contouring_obj)
    
    # Add contouring constraints (placeholder - would need actual implementation)
    # contouring_const = ContouringConstraints(
    #     road_width=road_width,
    #     safety_margin=safety_margin
    # )
    # contouring_const.set_reference_path(reference_path)
    # planner.add_constraint(contouring_const)
    
    return planner
