"""
PyMPC - Python Model Predictive Control Framework

A comprehensive MPC framework for autonomous vehicle control with support for:
- Model Predictive Contouring Control (MPCC)
- Scenario-based robust MPC
- Various constraint types (contouring, obstacle avoidance, etc.)
- Multiple objective functions
- Comprehensive test framework

Main Components:
- Core: Dynamics models, planners, solvers
- Modules: Constraints and objectives
- Utils: Mathematical utilities, logging, visualization
- Testing: Unified test framework
"""

# Core imports
from .core.dynamics import BaseDynamics, BicycleModel, KinematicModel
from .core.planner import MPCCPlanner
from .core.solver import BaseSolver, CasADiSolver
from .core.modules_manager import ModuleManager, BaseModule
from .core.parameters_manager import ParameterManager

# Constraint imports
from .modules.constraints import (
    BaseConstraint, ContouringConstraints, ScenarioConstraints,
    EllipsoidConstraints, GaussianConstraints, LinearizedConstraints,
    DecompositionConstraints, GuidanceConstraints
)

# Objective imports
from .modules.objectives import (
    BaseObjective, ContouringObjective, GoalObjective, PathReferenceVelocityObjective
)

# Utility imports
from .utils import (
    Spline, MPCLogger, get_logger, setup_logging_config,
    LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR
)

# Testing imports
try:
    from .test import (
        UnifiedTestRunner, TestConfig, TestResult,
        UnifiedConstraintFramework
    )
except ImportError:
    # Fallback for missing test modules
    pass

# Version information
__version__ = "1.0.0"
__author__ = "PyMPC Development Team"
__email__ = "pympc@example.com"

# Main classes for easy access
__all__ = [
    # Core classes
    'BaseDynamics', 'BicycleModel', 'KinematicModel',
    'MPCCPlanner', 'BaseSolver', 'CasADiSolver',
    'ModuleManager', 'BaseModule', 'ParameterManager',
    
    # Constraint classes
    'BaseConstraint', 'ContouringConstraints', 'ScenarioConstraints',
    'EllipsoidConstraints', 'GaussianConstraints', 'LinearizedConstraints',
    'DecompositionConstraints', 'GuidanceConstraints',
    
    # Objective classes
    'BaseObjective', 'ContouringObjective', 'GoalObjective', 'PathReferenceVelocityObjective',
    
    # Utility classes
    'Spline', 'MPCLogger', 'get_logger', 'setup_logging_config',
    'LOG_DEBUG', 'LOG_INFO', 'LOG_WARN', 'LOG_ERROR',
    
    # Testing classes
    'UnifiedTestRunner', 'TestConfig', 'TestResult',
    'UnifiedConstraintFramework', 'ConstraintTestConfig', 'ConstraintTestResult',
    'TestConfigManager', 'TestScenario',
    
    # Version info
    '__version__', '__author__', '__email__'
]


def create_mpc_planner(dynamics_type: str = "bicycle", 
                      N: int = 20,
                      timestep: float = 0.1,
                      solver_options: dict = None) -> MPCCPlanner:
    """
    Create an MPC planner with default configuration.
    
    Args:
        dynamics_type: Type of dynamics model ("bicycle" or "kinematic")
        N: Prediction horizon length
        timestep: Time step
        solver_options: Solver options
        
    Returns:
        Configured MPC planner
    """
    # Create dynamics model
    if dynamics_type == "bicycle":
        dynamics = BicycleModel(dt=timestep)
    elif dynamics_type == "kinematic":
        dynamics = KinematicModel(dt=timestep)
    else:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")
    
    # Create solver
    solver = CasADiSolver(**(solver_options or {}))
    
    # Create planner
    planner = MPCCPlanner(
        dynamics=dynamics,
        N=N,
        dt=timestep,
        solver_options=solver_options or {}
    )
    
    return planner


def create_contouring_mpc(reference_path, 
                         contouring_weight: float = 2.0,
                         lag_weight: float = 1.0,
                         progress_weight: float = 1.5,
                         road_width: float = 6.0,
                         safety_margin: float = 0.5) -> MPCCPlanner:
    """
    Create an MPC planner configured for contouring control.
    
    Args:
        reference_path: Reference path as Nx2 array
        contouring_weight: Weight for contouring error
        lag_weight: Weight for lag error
        progress_weight: Weight for progress
        road_width: Road width
        safety_margin: Safety margin
        
    Returns:
        Configured MPC planner for contouring control
    """
    # Create planner
    planner = create_mpc_planner()
    
    # Add contouring objective
    contouring_obj = ContouringObjective(
        contouring_weight=contouring_weight,
        lag_weight=lag_weight,
        progress_weight=progress_weight
    )
    contouring_obj.set_reference_path(reference_path)
    planner.add_objective(contouring_obj)
    
    # Add contouring constraints
    contouring_const = ContouringConstraints(
        road_width=road_width,
        safety_margin=safety_margin
    )
    contouring_const.set_reference_path(reference_path)
    planner.add_constraint(contouring_const)
    
    return planner


def create_goal_reaching_mpc(goal_position,
                            distance_weight: float = 1.0,
                            velocity_weight: float = 0.1,
                            orientation_weight: float = 0.1) -> MPCCPlanner:
    """
    Create an MPC planner configured for goal reaching.
    
    Args:
        goal_position: Target goal position [x, y]
        distance_weight: Weight for distance to goal
        velocity_weight: Weight for velocity at goal
        orientation_weight: Weight for orientation at goal
        
    Returns:
        Configured MPC planner for goal reaching
    """
    # Create planner
    planner = create_mpc_planner()
    
    # Add goal objective
    goal_obj = GoalObjective(
        goal_position=goal_position,
        distance_weight=distance_weight,
        velocity_weight=velocity_weight,
        orientation_weight=orientation_weight
    )
    planner.add_objective(goal_obj)
    
    return planner


def create_obstacle_avoidance_mpc(goal_position, obstacles,
                                 distance_weight: float = 1.0,
                                 safety_margin: float = 1.0) -> MPCCPlanner:
    """
    Create an MPC planner configured for obstacle avoidance.
    
    Args:
        goal_position: Target goal position [x, y]
        obstacles: List of obstacle dictionaries
        distance_weight: Weight for distance to goal
        safety_margin: Safety margin around obstacles
        
    Returns:
        Configured MPC planner for obstacle avoidance
    """
    # Create planner
    planner = create_mpc_planner()
    
    # Add goal objective
    goal_obj = GoalObjective(goal_position=goal_position, distance_weight=distance_weight)
    planner.add_objective(goal_obj)
    
    # Add obstacle constraints
    for obstacle in obstacles:
        if obstacle['type'] == 'circle':
            ellipsoid_const = EllipsoidConstraints(safety_margin=safety_margin)
            ellipsoid_const.add_circular_obstacle(obstacle['position'], obstacle['radius'])
            planner.add_constraint(ellipsoid_const)
        elif obstacle['type'] == 'ellipse':
            ellipsoid_const = EllipsoidConstraints(safety_margin=safety_margin)
            ellipsoid_const.add_elliptical_obstacle(
                obstacle['position'], obstacle['semi_major'], obstacle['semi_minor']
            )
            planner.add_constraint(ellipsoid_const)
    
    return planner


def run_simple_test(test_name: str = "simple_test",
                   test_type: str = "goal_reaching",
                   output_dir: str = "test_outputs") -> dict:
    """
    Run a simple MPC test.
    
    Args:
        test_name: Name of the test
        test_type: Type of test to run
        output_dir: Output directory for results
        
    Returns:
        Test results dictionary
    """
    from .test.unified_test_runner import UnifiedTestRunner, TestConfig
    
    # Create test configuration
    config = TestConfig(
        test_name=test_name,
        test_type=test_type,
        vehicle_type="bicycle",
        N=15,
        timestep=0.1,
        max_steps=100,
        gif_generation=True,
        num_obstacles=3
    )
    
    # Create test runner
    runner = UnifiedTestRunner(output_dir=output_dir)
    
    # Run test
    result = runner.run_test(config)
    
    return {
        'test_name': result.test_name,
        'success': result.success,
        'execution_time': result.execution_time,
        'final_position': result.final_position.tolist() if result.final_position is not None else None,
        'final_error': result.final_error,
        'constraint_violations': result.constraint_violations,
        'objective_value': result.objective_value
    }


# Example usage function
def example_usage():
    """Demonstrate basic usage of the PyMPC framework."""
    import numpy as np
    
    print("PyMPC Example Usage")
    print("===================")
    
    # Create a simple reference path
    t = np.linspace(0, 4*np.pi, 50)
    reference_path = np.column_stack([t * np.cos(t/4), t * np.sin(t/4)])
    
    # Create contouring MPC
    print("Creating contouring MPC...")
    planner = create_contouring_mpc(reference_path)
    
    # Set up initial state
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, psi, v]
    
    # Solve MPC problem
    print("Solving MPC problem...")
    try:
        solution = planner.solve(initial_state)
        
        if solution and solution.get('status') == 'optimal':
            print("MPC solve successful!")
            print(f"Solve time: {solution.get('solve_time', 0):.4f} seconds")
            
            # Get optimal trajectory
            trajectory = solution.get('states')
            controls = solution.get('controls')
            
            if trajectory is not None:
                print(f"Trajectory shape: {trajectory.shape}")
                print(f"Final position: {trajectory[-1, :2]}")
            
            if controls is not None:
                print(f"Control shape: {controls.shape}")
                print(f"First control: {controls[:, 0]}")
        else:
            print("MPC solve failed!")
            
    except Exception as e:
        print(f"Error during MPC solve: {e}")
    
    # Run a simple test
    print("\nRunning simple test...")
    test_results = run_simple_test("example_test", "goal_reaching")
    print(f"Test results: {test_results}")


if __name__ == "__main__":
    example_usage()