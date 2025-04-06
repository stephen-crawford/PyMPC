import casadi as ca

from solver.solver_interface import BaseSolver


class CasADiSolver(BaseSolver):
    def __init__(self, dt=0.1, N=20):
        super().__init__(dt, N)
        self.opti = ca.Opti()
        self.solution = None
        self._setup_problem()

    def _setup_problem(self):
        self.X = self.opti.variable(3, self.N + 1)
        self.U = self.opti.variable(2, self.N)
        self.X0 = self.opti.parameter(3)

        cost = 0
        for k in range(self.N):
            cost += ca.sumsqr(self.X[:, k]) + ca.sumsqr(self.U[:, k])
            self.opti.subject_to(self.X[:, k + 1] == self._dynamics(self.X[:, k], self.U[:, k]))

        self.opti.minimize(cost)
        self.opti.solver("ipopt", {"expand": True}, {"max_iter": 100})

    def _dynamics(self, x, u):
        theta = x[2]
        dx = ca.vertcat(
            u[0] * ca.cos(theta),
            u[0] * ca.sin(theta),
            u[1]
        )
        return x + self.dt * dx

    def set_xinit(self, state):
        self.opti.set_value(self.X0, state)

    def solve(self):
        try:
            self.opti.set_initial(self.X, 0)
            self.opti.set_initial(self.U, 0)
            self.opti.subject_to(self.X[:, 0] == self.X0)
            self.solution = self.opti.solve()
            return 1
        except RuntimeError:
            return -1

    def get_output(self, k, var_name):
        return self.solution.value(self.X[0, k]) if var_name == "x" else self.solution.value(self.X[1, k])

    def reset(self):
        self.opti = ca.Opti()
        self.solution = None
        self._setup_problem()

    def explain_exit_flag(self, code):
        return "Solved successfully" if code == 1 else "Solver failed"
