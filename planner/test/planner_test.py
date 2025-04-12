import unittest
from unittest.mock import MagicMock
import time

from planner.src.planner import Planner, PlannerOutput
from planner.src.types import Trajectory

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
    def is_objective_reached(self, planner, state, data): return True
    def on_data_received(self, data, data_name): pass

# Dummy CasADi solver mock
class DummyCasadiSolver:
    def __init__(self):
        self.dt = 0.1
        self.N = 5
        self.params = MagicMock()
        self.params.solver_timeout = 0.1

    def reset(self): pass
    def initialize_warmstart(self, state, shift): pass
    def initialize_with_braking(self, state): pass
    def set_xinit(self, state): pass
    def get_ego_prediction(self, k, var): return k * 1.0
    def load_warmstart(self): pass
    def solve(self): return 1
    def explain_exit_flag(self, flag): return "Success"
    def get_output(self, k, var): return k * 2.0
    def print_if_bound_limited(self): pass

# Dummy OSQP solver mock (same interface)
class DummyOSQPSolver(DummyCasadiSolver):
    def solve(self): return 1  # Simulate success

class PlannerTest(unittest.TestCase):

    def setUp(self):
        self.state = DummyState()
        self.data = DummyData()
        self.modules = [DummyModule()]

    def test_planner_with_casadisolver(self):
        solver = DummyCasadiSolver()
        planner = Planner(solver, self.modules)

        output = planner.solve_mpc(self.state, self.data)
        self.assertTrue(output.success)
        self.assertIsInstance(output.trajectory, Trajectory)
        self.assertGreater(len(output.trajectory.positions), 0)

    def test_planner_with_osqpsolver(self):
        solver = DummyOSQPSolver()
        planner = Planner(solver, self.modules)

        output = planner.solve_mpc(self.state, self.data)
        self.assertTrue(output.success)
        self.assertGreater(len(output.trajectory.positions), 0)

    def test_planner_failure_handling(self):
        class FailingSolver(DummyCasadiSolver):
            def solve(self): return -2
            def explain_exit_flag(self, flag): return "Mocked failure"

        planner = Planner(FailingSolver(), self.modules)
        output = planner.solve_mpc(self.state, self.data)
        self.assertFalse(output.success)

if __name__ == '__main__':
    unittest.main()
