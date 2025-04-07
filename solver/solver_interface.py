from abc import ABC, abstractmethod


def set_solver_parameter(params, key, value, k, index=0, settings=None):
    if settings is None:
        raise ValueError("settings must be provided")

    bundles = settings["params"].parameter_bundles
    length = settings["params"].length()

    if key not in bundles:
        raise KeyError(f"Parameter '{key}' not found in parameter bundles.")

    indices = bundles[key]

    function_name = key.replace("_", " ").title().replace(" ", "")

    # Handling single index case
    if len(indices) == 1:
        param_index = k * length + indices[0]
        params.all_parameters[param_index] = value
    else:
        # Handling multiple indices case
        if index < 0 or index >= len(indices):
            raise IndexError(f"Index {index} out of bounds for parameter bundle '{key}'")

        param_index = k * length + indices[index]
        params.all_parameters[param_index] = value


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
