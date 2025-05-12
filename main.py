from planning.src.planner import Planner
from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
from planner_modules.src.objectives.goal_objective import GoalObjective
from planner_modules.src.constraints.guidance_constraints import GuidanceConstraints
from solver.src.casadi_solver import CasADiSolver


def main():

	solver = CasADiSolver()
	module_manager = solver.get_module_manager()

	goal_module = GoalObjective(solver)
	contouring_module = ContouringConstraints(solver)
	guidance_constraints = GuidanceConstraints(solver)

	modules = [goal_module, contouring_module, guidance_constraints]
	for module in modules:
		module_manager.add_module(module)

	planner = Planner(solver)

	planner.solve_mpc()

