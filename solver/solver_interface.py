from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, dt, N):
        self.dt = dt
        self.N = N
        self.params = {"solver_timeout": 0.1}

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def set_xinit(self, state):
        pass

    @abstractmethod
    def get_output(self, k, var_name):
        pass

    @abstractmethod
    def explain_exit_flag(self, code):
        pass
