import casadi as cd
import numpy as np

from solver_generator.util.files import model_map_path, write_to_yaml
from solver_generator.util.logging import print_warning

from solver_generator.spline import Spline2D

import casadi as ca


def casadi_discrete_dynamics(z, p, model, settings, nx=None, integration_step=None):
    """
    Discretizes a continuous-time model using RK4, implemented with CasADi.

    @param z: CasADi vector containing [u, x]
    @param p: Parameters
    @param model: Object containing .continuous_model (Callable: f(x, u, p))
    @param settings: Dictionary with 'integrator_step' key
    @param nx: Number of states
    @param integration_step: Optional step size override
    @return: Discretized state after one integration step
    """
    if nx is None:
        nx = model.nx

    if integration_step is None:
        integration_step = settings["integrator_step"]

    # Extract control and state
    u = z[0:model.nu]
    x = z[model.nu:model.nu + nx]

    # Define the continuous dynamics function
    # Make sure the continuous_model returns a CasADi MX/SX object, not a numpy array
    def f(x_val, u_val):
        # Ensure result is a CasADi vector
        result = model.continuous_model(x_val, u_val)
        # If result is a numpy array, convert it to CasADi vector
        if isinstance(result, np.ndarray):
            return ca.vertcat(*[ca.MX(r) if not isinstance(r, (ca.MX, ca.SX)) else r for r in result])
        return result

    dt = ca.MX(integration_step)

    # RK4 integration
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)

    # Update the state
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


def numpy_to_casadi(x: np.array) -> cd.SX:
    result = None
    for param in x:
        if result is None:
            result = param
        else:
            result = cd.vertcat(result, param)
    return result


class DynamicsModel:

    def __init__(self):
        self.nu = 0  # number of control variables
        self.nx = 0  # number of states

        self.states = []
        self.inputs = []

        self.lower_bound = []
        self.upper_bound = []

        self.params = None
        self.nx_integrate = None

    def discrete_dynamics(self, z, p, settings, **kwargs):
        # Load parameters and settings
        params = settings["params"]
        params.load(p)
        self.load(z)
        self.load_settings(settings)

        # Get integration step (default to 0.1 if not provided)
        integration_step = settings.get("dt", 0.1)
        if integration_step is None:
            integration_step = 0.1

        # Determine how many states to integrate
        nx_integrate = self.nx if self.nx_integrate is None else self.nx_integrate

        # Call the discretization function
        integrated_states = casadi_discrete_dynamics(z, p, self, settings,
                                                     nx=nx_integrate,
                                                     integration_step=integration_step,
                                                     **kwargs)

        # Apply model-specific discrete dynamics
        integrated_states = self.model_discrete_dynamics(z, integrated_states, **kwargs)
        return integrated_states

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        return integrated_states

    def get_nvar(self):
        return self.nu + self.nx

    def get_xinit(self):
        return range(self.nu, self.get_nvar())

    def acados_symbolics(self):
        x = cd.SX.sym("x", self.nx)  # [px, py, vx, vy]
        u = cd.SX.sym("u", self.nu)  # [ax, ay]
        z = cd.vertcat(u, x)
        self.load(z)
        return z

    def get_acados_dynamics(self):
        self._x_dot = cd.SX.sym("x_dot", self.nx)

        # Correctly pass parameters to continuous_model
        f_expl = numpy_to_casadi(self.continuous_model(self._z[self.nu:], self._z[:self.nu]))
        f_impl = self._x_dot - f_expl
        return f_expl, f_impl

    def get_x(self):
        return self._z[self.nu:]

    def get_u(self):
        return self._z[:self.nu]

    def get_acados_x_dot(self):
        return self._x_dot

    def get_acados_u(self):
        return self._z[:self.nu]

    def load(self, z):
        self._z = z

    def load_settings(self, settings):
        self.params = settings["params"]
        self.settings = settings

    def save_map(self):
        file_path = model_map_path()

        map = dict()
        for idx, state in enumerate(self.states):
            map[state] = ["x", idx + self.nu, self.get_bounds(state)[0], self.get_bounds(state)[1]]

        for idx, input in enumerate(self.inputs):
            map[input] = ["u", idx, self.get_bounds(input)[0], self.get_bounds(input)[1]]

        write_to_yaml(file_path, map)

    def integrate(self, z, settings, integration_step):
        # This function should handle params correctly
        return self.discrete_dynamics(z, settings["params"].get_p(), settings, integration_step=integration_step)

    def do_not_use_integration_for_last_n_states(self, n):
        self.nx_integrate = self.nx - n

    def get(self, state_or_input):
        if state_or_input in self.states:
            i = self.states.index(state_or_input)
            return self._z[self.nu + i]
        elif state_or_input in self.inputs:
            i = self.inputs.index(state_or_input)
            return self._z[i]
        else:
            raise IOError(
                f"Requested a state or input `{state_or_input}' that was neither a state nor an input for the selected model")

    def set_bounds(self, lower_bound, upper_bound):
        assert len(lower_bound) == len(upper_bound) == len(self.lower_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_bounds(self, state_or_input):
        if state_or_input in self.states:
            i = self.states.index(state_or_input)
            return (
                self.lower_bound[self.nu + i],
                self.upper_bound[self.nu + i],
                self.upper_bound[self.nu + i] - self.lower_bound[self.nu + i],
            )
        elif state_or_input in self.inputs:
            i = self.inputs.index(state_or_input)
            return (
                self.lower_bound[i],
                self.upper_bound[i],
                self.upper_bound[i] - self.lower_bound[i],
            )
        else:
            raise IOError(
                f"Requested a state or input `{state_or_input}' that was neither a state nor an input for the selected model")

    # This is the method that needs to be implemented by subclasses
    def continuous_model(self, x, u):
        """
    Compute the continuous-time dynamics for the model.

    @param x: State vector
    @param u: Control vector
    @return: Derivative of state vector
    """
        raise NotImplementedError("Subclasses must implement continuous_model")


class SecondOrderUnicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.nx = 4  # number of states

        self.states = ["x", "y", "psi", "v"]
        self.inputs = ["a", "w"]

        self.lower_bound = [-2.0, -2.0, -200.0, -200.0, -np.pi * 4, -2.0]
        self.upper_bound = [2.0, 2.0, 200.0, 200.0, np.pi * 4, 3.0]

    def continuous_model(self, x, u):
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return np.array([v * cd.cos(psi), v * cd.sin(psi), w, a])


class ContouringSecondOrderUnicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.nx = 5  # number of states

        self.states = ["x", "y", "psi", "v", "spline"]
        self.inputs = ["a", "w"]

        # w = 0.8
        self.lower_bound = [-2.0, -0.8, -2000.0, -2000.0, -np.pi * 4, -0.01, -1.0]
        self.upper_bound = [2.0, 0.8, 2000.0, 2000.0, np.pi * 4, 3.0, 10000.0]

    def continuous_model(self, x, u):
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return np.array([v * cd.cos(psi), v * cd.sin(psi), w, a, v])


class ContouringSecondOrderUnicycleModelCurvatureAware(DynamicsModel):

    def __init__(self):
        super().__init__()
        print_warning(
            "ContouringSecondOrderUnicycleModelCurvatureAware is not supported in Acados as discrete dynamics are necessary for the spline state")
        self.nu = 2  # number of control variables
        self.nx = 5  # number of states

        self.states = ["x", "y", "psi", "v", "spline"]
        self.inputs = ["a", "w"]

        self.do_not_use_integration_for_last_n_states(n=1)

        self.lower_bound = [-4.0, -0.8, -2000.0, -2000.0, -np.pi * 4, -0.01, -1.0]
        self.upper_bound = [4.0, 0.8, 2000.0, 2000.0, np.pi * 4, 3.0, 10000.0]

    def continuous_model(self, x, u):
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return np.array([v * cd.cos(psi), v * cd.sin(psi), w, a])

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        x = self.get_x()

        pos_x = x[0]
        pos_y = x[1]
        s = x[-1]

        # CA-MPC
        path = Spline2D(self.params, self.settings["contouring"]["get_num_segments"], s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
        path_ddx, path_ddy = path.deriv2(s)

        # Contour = n_vec
        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        dp = np.array([integrated_states[0] - pos_x, integrated_states[1] - pos_y])
        t_vec = np.array([path_dx_normalized, path_dy_normalized])
        n_vec = np.array([path_dy_normalized, -path_dx_normalized])

        vt_t = dp.dot(t_vec)
        vn_t = dp.dot(n_vec)

        R = 1.0 / path.get_curvature(s)  # max(R) = 1 / 0.0001
        R = cd.fmax(R, 1e5)

        theta = cd.atan2(vt_t, R - contour_error - vn_t)

        return cd.vertcat(integrated_states, s + R * theta)


class ContouringSecondOrderUnicycleModelWithSlack(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.nx = 6  # number of states

        self.states = ["x", "y", "psi", "v", "spline", "slack"]
        self.inputs = ["a", "w"]

        self.lower_bound = [-2.0, -0.8, -2000.0, -2000.0, -np.pi * 4, -0.01, -1.0, 0.0]  # v -0.01
        self.upper_bound = [2.0, 0.8, 2000.0, 2000.0, np.pi * 4, 3.0, 10000.0, 5000.0]  # w 0.8

    def continuous_model(self, x, u):
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return np.array([v * cd.cos(psi), v * cd.sin(psi), w, a, v, 0.0])

    # NOTE: No initialization for slack variable
    def get_xinit(self):
        return range(self.nu, self.get_nvar() - 1)


# Bicycle model with dynamic steering
class BicycleModel2ndOrder(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 3
        self.nx = 6

        self.states = ["x", "y", "psi", "v", "delta", "spline"]
        self.inputs = ["a", "w", "slack"]

        # Prius limits: https:#github.com/oscardegroot/lmpcc/blob/prius/lmpccsolver/scripts/systems.py
        # w [-0.2, 0.2] | a [-1.0 1.0]
        # w was 0.5
        # delta was 0.45

        # NOTE: the angle of the vehicle should not be limited to -pi, pi, as the solution will not shift when it is at the border!
        # a was 3.0
        self.lower_bound = [-3.0, -1.5, 0.0, -1.0e6, -1.0e6, -np.pi * 4, -0.01, -0.55, -1.0]
        self.upper_bound = [3.0, 1.5, 1.0e2, 1.0e6, 1.0e6, np.pi * 4, 5.0, 0.55, 5000.0]

    def continuous_model(self, x, u):
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]
        delta = x[4]

        wheel_base = 2.79  # between front wheel center and rear wheel center
        wheel_tread = 1.64  # between left wheel center and right wheel center
        front_overhang = 1.0  # between front wheel center and vehicle front
        rear_overhang = 1.1  # between rear wheel center and vehicle rear
        left_overhang = 0.128  # between left wheel center and vehicle left
        right_overhang = 0.128  # between right wheel center and vehicle right

        # NOTE: Mass is equally distributed according to the parameters
        lr = wheel_base / 2.0
        lf = wheel_base / 2.0
        ratio = lr / (lr + lf)
        self.width = 2.25

        beta = cd.arctan(ratio * cd.tan(delta))

        return np.array([v * cd.cos(psi + beta), v * cd.sin(psi + beta), (v / lr) * cd.sin(beta), a, w, v])


# Bicycle model with dynamic steering
class BicycleModel2ndOrderCurvatureAware(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 3
        self.nx = 6

        self.states = ["x", "y", "psi", "v", "delta", "spline"]
        self.inputs = ["a", "w", "slack"]

        self.do_not_use_integration_for_last_n_states(n=1)

        # Prius limits: https:#github.com/oscardegroot/lmpcc/blob/prius/lmpccsolver/scripts/systems.py
        # w [-0.2, 0.2] | a [-1.0 1.0]
        # w was 0.5
        # delta was 0.45

        # NOTE: the angle of the vehicle should not be limited to -pi, pi, as the solution will not shift when it is at the border!
        # a was 3.0
        # delta was -0.45, 0.45
        self.lower_bound = [-3.0, -1.5, 0.0, -1.0e6, -1.0e6, -np.pi * 4, -0.01, -0.55, -1.0]
        self.upper_bound = [3.0, 1.5, 1.0e2, 1.0e6, 1.0e6, np.pi * 4, 8.0, 0.55, 5000.0]

    def continuous_model(self, x, u):
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]
        delta = x[4]

        wheel_base = 2.79  # between front wheel center and rear wheel center

        # NOTE: Mass is equally distributed according to the parameters
        self.lr = wheel_base / 2.0
        self.lf = wheel_base / 2.0
        ratio = self.lr / (self.lr + self.lf)

        self.width = 2.25

        beta = cd.arctan(ratio * cd.tan(delta))

        return np.array([v * cd.cos(psi + beta), v * cd.sin(psi + beta), (v / self.lr) * cd.sin(beta), a, w])

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        x = self.get_x()

        pos_x = x[0]
        pos_y = x[1]
        psi = x[2]
        vel = x[3]
        s = x[-1]

        # CA-MPC
        path = Spline2D(self.params, self.settings["contouring"]["get_num_segments"], s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
        path_ddx, path_ddy = path.deriv2(s)

        # Contour = n_vec
        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        dp = np.array([integrated_states[0] - pos_x, integrated_states[1] - pos_y])
        t_vec = np.array([path_dx_normalized, path_dy_normalized])
        n_vec = np.array([path_dy_normalized, -path_dx_normalized])

        vt_t = dp.dot(t_vec)
        vn_t = dp.dot(n_vec)

        R = 1.0 / path.get_curvature(s)  # max(R) = 1 / 0.0001
        R = cd.fmax(R, 1e5)

        theta = cd.atan2(vt_t, R - contour_error - vn_t)

        return cd.vertcat(integrated_states, s + R * theta)