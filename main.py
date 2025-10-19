from planning.src.planner import Planner
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.objectives.goal_objective import GoalObjective
from planner_modules.src.constraints.guidance_constraints import GuidanceConstraints
from solver.src.casadi_solver import CasADiSolver
from planning.src.types import Data


def main():
    solver = CasADiSolver()
    module_manager = solver.get_module_manager()

    goal_module = GoalObjective(solver)
    contouring_module = ContouringConstraints(solver)
    guidance_constraints = GuidanceConstraints(solver)

    modules = [goal_module, contouring_module, guidance_constraints]
    for module in modules:
        module_manager.add_module(module)

    # Create a model instance for the planner
    model = ContouringSecondOrderUnicycleModel()
    planner = Planner(solver, model)

    # Create some basic data for the planner
    data = Data()
    data.start = [0.0, 0.0]
    data.goal = [10.0, 10.0]
    data.goal_received = True
    data.planning_start_time = 0.0

    planner.initialize(data)
    planner.solve_mpc(data)

