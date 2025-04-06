import numpy as np
import osqp

from solver.solver_interface import BaseSolver


class OSQPSolver(BaseSolver):
    def __init__(self, dt=0.1, N=20):
        super().__init__(dt, N)
        self.solver = osqp.OSQP()
        self._setup_problem()

    def _setup_problem(self):
        P = np.eye(self.N)
        q = np.zeros(self.N)
        A = np.eye(self.N)
        l = np.zeros(self.N)
        u = np.ones(self.N)

        self.solver.setup(P, q, A, l, u, verbose=False)

    def set_xinit(self, state):
        pass  # OSQP doesn’t need this

    def solve(self):
        result = self.solver.solve()
        return 1 if result.info.status == 'solved' else -1

    def get_output(self, k, var_name):
        return self.solver.solution.x[k]

    def reset(self):
        self._setup_problem()

    def explain_exit_flag(self, code):
        return "Solved successfully" if code == 1 else "Solver failed"
