"""
Example usage of Safe Horizon Constraint in main.py
"""
from planning.src.planner import Planner
from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.objectives.goal_objective import GoalObjective
from planner_modules.src.constraints.guidance_constraints import GuidanceConstraints
from planner_modules.src.constraints.safe_horizon_constraint import SafeHorizonConstraint
from solver.src.casadi_solver import CasADiSolver


def main():
    """Main function demonstrating Safe Horizon Constraint integration."""
    
    # Create solver
    solver = CasADiSolver()
    module_manager = solver.get_module_manager()

    # Create modules
    goal_module = GoalObjective(solver)
    contouring_module = ContouringConstraints(solver)
    guidance_constraints = GuidanceConstraints(solver)
    
    # Create Safe Horizon Constraint
    safe_horizon_constraint = SafeHorizonConstraint(solver)

    # Add modules to manager
    modules = [goal_module, contouring_module, guidance_constraints, safe_horizon_constraint]
    for module in modules:
        module_manager.add_module(module)

    # Create planner
    planner = Planner(solver)

    # Run MPC with Safe Horizon constraints
    planner.solve_mpc()


if __name__ == "__main__":
    main()
