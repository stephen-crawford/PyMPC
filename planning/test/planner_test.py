import unittest
from unittest.mock import MagicMock
import time

from planning.src.planner import Planner, PlannerOutput
from planning.src.types import Trajectory
from solver.src.base_solver import BaseSolver
from solver.src.parameter_manager import ParameterManager
from utils.const import OBJECTIVE, CONSTRAINT


# Dummy state and data classes
class DummyState:
    def reset(self): pass

class DummyData:
    def __init__(self):
        self.planning_start_time = time.time()
    def reset(self): pass

# Dummy module
class DummyModule:
    def initialize(self, solver): pass
    def is_data_ready(self, data): return True
    def update(self, state, data, module_data): pass
    def set_parameters(self, data, module_data, k): pass
    def optimize(self, state, data, module_data): return -1
    def visualize(self, data): pass
    def save_data(self, saver): pass
    def reset(self): pass
    def is_objective_reached(self, state, data): return True
    def on_data_received(self, data, data_name): pass

# Dummy CasADi solver mock
class DummyCasadiSolver(BaseSolver):
    def __init__(self):
        super().__init__()
        self.timestep = 0.1
        self.horizon = 5
        self.params = ParameterManager()
        self.params.add("solver_timeout")
        self.params.set_parameter("solver_timeout", "10")

    def add(self): pass
    def reset(self): pass
    def initialize_warmstart(self, state, shift): pass
    def initialize_with_braking(self, state): pass
    def set_xinit(self, state): pass
    def get_ego_prediction(self, k, var): return k * 1.0
    def load_warmstart(self): pass
    def solve(self):
        print("returning 1")
        return 1
    def explain_exit_flag(self, flag): return "Success"
    def get_output(self, k, var): return k * 2.0
    def print_if_bound_limited(self): pass


    def set_initial_state(self, state):
        pass

    def define_parameters(self):

        # Define parameters for objectives and constraints (in order)
        for module in self.module_manager.modules:
            if module.module_type == OBJECTIVE:
                module.define_parameters(self.module_manager, self.parameter_manager)

        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                module.define_parameters(self.module_manager, self.parameter_manager)

    def objective(self, z, p, model, settings, stage_idx):
        cost = 0.0

        params = settings["params"]
        params.load(p)
        model.load(z)

        for module in self.module_manager.modules:
            if module.module_type == OBJECTIVE:
                cost += module.get_value(model, params, settings, stage_idx)

        # if stage_idx == 0:
        # print(cost)

        return cost

    def constraints(self, module_manager, z, p, model, settings, stage_idx):
        constraints = []

        params = settings["params"]
        params.load(p)
        model.load(z)

        for module in module_manager.modules:
            if module.module_type == CONSTRAINT:
                constraints += module.get_constraints(model, params, settings, stage_idx)

        return constraints

    def constraint_upper_bounds(self):
        ub = []
        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                ub += module.get_upper_bound()
        return ub

    def constraint_lower_bounds(self):
        lb = []
        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                lb += module.get_lower_bound()
        return lb

    def constraint_number(self):
        nh = 0
        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                nh += module.nh
        return nh

# Dummy OSQP solver mock (same interface)
class DummyOSQPSolver(DummyCasadiSolver):
    def solve(self): return 1  # Simulate success

class PlannerTest(unittest.TestCase):


    def setUp(self):
        self.state = DummyState()
        self.data = DummyData()
        self.modules = [DummyModule()]

    def get_output(self, k, var_name):
        pass

    def explain_exit_flag(self, code):
        pass

    def test_planner_with_casadi_solver(self):
        solver = DummyCasadiSolver()
        planner = Planner(solver)

        output = planner.solve_mpc(self.state, self.data)
        self.assertTrue(output.success)
        self.assertIsInstance(output.trajectory, Trajectory)
        self.assertGreater(len(output.trajectory.positions), 0)

    def test_planner_with_osqp_solver(self):
        solver = DummyOSQPSolver()
        planner = Planner(solver)

        output = planner.solve_mpc(self.state, self.data)
        self.assertTrue(output.success)
        self.assertGreater(len(output.trajectory.positions), 0)

    def test_planner_failure_handling(self):
        class FailingSolver(DummyCasadiSolver):
            def solve(self): return -2
            def explain_exit_flag(self, flag): return "Mocked failure"

        planner = Planner(FailingSolver())
        output = planner.solve_mpc(self.state, self.data)
        self.assertFalse(output.success)

if __name__ == '__main__':
    unittest.main()
