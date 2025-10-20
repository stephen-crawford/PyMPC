from planning.src.planner import Planner
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.constraints.scenario_constraints import ScenarioConstraints
from planner_modules.src.objectives.goal_objective import GoalObjective
from planner_modules.src.objectives.contouring_objective import ContouringObjective
from planner_modules.src.constraints.guidance_constraints import GuidanceConstraints
from solver.src.casadi_solver import CasADiSolver
from planning.src.types import Data, ReferencePath
import numpy as np


def main():
    # Create solver
    solver = CasADiSolver(timestep=0.1, horizon=10)
    
    # Create vehicle model
    model = ContouringSecondOrderUnicycleModel()
    solver.set_dynamics_model(model)
    
    # Create planner
    planner = Planner(solver, model)
    
    # Add modules
    goal_module = GoalObjective(solver)
    contouring_objective = ContouringObjective(solver)
    contouring_constraints = ContouringConstraints(solver)
    scenario_constraints = ScenarioConstraints(solver)
    # guidance_constraints = GuidanceConstraints(solver)  # Skip for now due to complexity
    
    # Add modules to solver
    solver.module_manager.add_module(goal_module)
    solver.module_manager.add_module(contouring_objective)
    solver.module_manager.add_module(contouring_constraints)
    solver.module_manager.add_module(scenario_constraints)
    # solver.module_manager.add_module(guidance_constraints)  # Skip for now
    
    # Create test data
    data = Data()
    data.start = [0.0, 0.0]
    data.goal = [10.0, 10.0]
    data.goal_received = True
    data.planning_start_time = 0.0
    
    # Create reference path
    t = np.linspace(0, 1, 50)
    x_path = np.linspace(0, 10, 50)
    y_path = 2 * np.sin(2 * np.pi * t)
    s_path = np.linspace(0, 1, 50)
    
    reference_path = ReferencePath()
    reference_path.set('x', x_path)
    reference_path.set('y', y_path)
    reference_path.set('s', s_path)
    data.reference_path = reference_path
    
    # Create road boundaries
    road_width = 4.0
    left_bound = ReferencePath()
    left_bound.set('x', x_path)
    left_bound.set('y', y_path + road_width/2)
    right_bound = ReferencePath()
    right_bound.set('x', x_path)
    right_bound.set('y', y_path - road_width/2)
    
    data.left_bound = left_bound
    data.right_bound = right_bound
    
    # Initialize and solve
    planner.initialize(data)
    result = planner.solve_mpc(data)
    
    print(f"MPC Solution: {'Success' if result.success else 'Failed'}")
    if result.success:
        print(f"Trajectory length: {len(result.trajectory_history)}")
        if result.trajectory_history:
            print(f"First trajectory point: {result.trajectory_history[0]}")

