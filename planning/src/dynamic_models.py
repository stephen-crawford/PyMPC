from math import pi

import casadi as cd
import numpy as np
from scipy.interpolate import interpolate

from utils.utils import model_map_path, write_to_yaml, LOG_DEBUG, LOG_WARN, LOG_INFO


def safe_casadi_discrete_dynamics(model, z, p, nx=None, integration_step=None):
    """
    Symbolic rk4 using Casadi.
    """
    try:
        if nx is None:
            nx = model.state_dimension

        if integration_step is None:
            integration_step = 0.1

        # Ensure integration step is reasonable
        integration_step = min(max(integration_step, 0.01), 0.5)

        # Extract control and state with bounds checking
        if z.shape[0] < model.nu + nx:
            # Create safe fallback values
            return cd.DM.zeros(nx, 1)

        u = z[0:model.nu]
        x = z[model.nu:model.nu + nx]

        # Define the continuous dynamics function with additional safety
        def f(x_val, u_val, params):
            try:
                result = model.continuous_model(x_val, u_val, params)
                # Skip NaN check for symbolic expressions
                if isinstance(result, np.ndarray) and np.isnan(result).any():
                    LOG_WARN("NaN in continuous model output")
                    return cd.DM.zeros(nx, 1)
                return result
            except Exception as e:
                LOG_WARN(f"Error in continuous model: {e}")
                return cd.DM.zeros(nx, 1)

        # Use a safer dt value
        dt = cd.MX(integration_step)

        # RK4 integration with additional error checking
        h = dt
        try:
            k1 = f(x, u, p)
            k2 = f(x + h / 2 * k1, u, p)
            k3 = f(x + h / 2 * k2, u, p)
            k4 = f(x + h * k3, u, p)
        except Exception as e:
            LOG_WARN(f"Error in RK4 steps: {e}")
            return x  # Return original state if integration fails

        # Compute next state with safeguards against NaN
        x_next = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Only perform NaN checks and bounds enforcement for numeric types, not symbolic
        if isinstance(x_next, cd.DM):  # Check if it's a numeric type
            # Project extreme values and check for NaN
            for i in range(x_next.size1()):
                if not x_next[i].is_regular():  # Only for numeric values
                    x_next[i] = x[i]  # Replace NaN with original value
                x_next[i] = cd.fmin(cd.fmax(x_next[i], -1e6), 1e6)  # Bound values

        return x_next

    except Exception as e:
        LOG_WARN(f"Error in safe_casadi_discrete_dynamics: {e}")
        # Return original state as fallback
        if nx is None:
            nx = model.state_dimension
        return z[model.nu:model.nu + nx]


def numeric_rk4(next_state, vehicle, params, timestep):
        if isinstance(next_state, cd.MX):
            LOG_DEBUG(f"Got symbolic result from discrete_dynamics. Using RK4 integration directly.")

            u = vehicle.get_z()[0:vehicle.nu]
            x = vehicle.get_z()[vehicle.nu:vehicle.nu + vehicle.state_dimension]

            # Get parameters
            p = params

            # Use continuous model directly
            k1 = vehicle.continuous_model(x, u, p)
            k2 = vehicle.continuous_model(x + timestep / 2 * k1, u, p)
            k3 = vehicle.continuous_model(x + timestep / 2 * k2, u, p)
            k4 = vehicle.continuous_model(x + timestep * k3, u, p)

            # Compute next state
            next_state = x + timestep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return next_state
        else:
            raise NotImplementedError


def numpy_to_casadi(x: np.array) -> cd.SX:
    return cd.vertcat(*x.tolist())

class DynamicsModel:

    def __init__(self):
        self.settings = None
        self._z = None
        self.nu = 0  # number of control variables
        self.state_dimension = 0  # number of states

        self.dependent_vars = []
        self.inputs = []

        self.lower_bound = []
        self.upper_bound = []

        self.params = None
        self.state_dimension_integrate = None


    def get_dependent_vars(self):
        return self.dependent_vars

    def get_inputs(self):
        return self.inputs

    def get_all_vars(self):
        return self.inputs + self.dependent_vars

    def discrete_dynamics(self, z, p, timestep, **kwargs):
        try:
            # LOG_INFO(f"Getting discrete dynamics for {z}, {p}")
            # Load the z vector
            self.load(z)

            if timestep is None:
                timestep = 0.1

            # Determine how many states to integrate
            state_dimension = self.state_dimension if self.state_dimension_integrate is None else self.state_dimension_integrate

            # Call the discretization function with explicit error checking
            integrated_states = safe_casadi_discrete_dynamics(self, z, p, nx=state_dimension, integration_step=timestep)
            try:
                if isinstance(z, (cd.MX, cd.SX)):
                    # Skip model_discrete_dynamics for symbolic expressions
                    return integrated_states
                else:
                    integrated_states = self.model_discrete_dynamics(z, integrated_states, **kwargs)
                    # Check for NaN in the result - only for numeric types
                    if isinstance(integrated_states, cd.DM) and not integrated_states.is_regular():
                        LOG_WARN("NaN detected in model_discrete_dynamics output")
            except Exception as e:
                LOG_WARN(f"Error in model_discrete_dynamics: {e}")
                # Return the states from RK4 if model-specific dynamics fails
                pass

            return integrated_states
        except Exception as e:
            LOG_WARN(f"Error in discrete_dynamics: {e}")
            raise

    def load(self, z):
        """
        Load state and control variables from the z vector.

        Parameters:
        ----------
        z : CasADi MX or SX vector or numpy array
            Vector containing control inputs followed by states [u, x]
        """
        # Store the full vector
        self._z = z

        # For debugging, you can print or verify the shape
        # import casadi as cd
        # if isinstance(z, (cd.MX, cd.SX)):
        #     print(f"Loaded symbolic z with shape {z.shape}")
        # else:
        #     print(f"Loaded numeric z with shape {len(z)}")


    def get_z(self):
        return self._z

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        """
        Apply model-specific discrete dynamics logic after integration.
        This method can be overridden by subclasses to implement model-specific behavior.

        Parameters:
        ----------
        z : CasADi MX or SX vector or numpy array
            Vector containing control inputs followed by states [u, x]
        integrated_states : CasADi MX or SX vector or numpy array
            The states after integration

        Returns:
        -------
        CasADi MX or SX vector or numpy array
            The updated states after model-specific logic
        """
        return integrated_states

    def get_nvar(self):
        return self.nu + self.state_dimension

    def get_xinit(self):
        return range(self.nu, self.get_nvar())

    def get_x(self):
        return self._z[self.nu:]

    def get_u(self):
        return self._z[:self.nu]

    def load_settings(self, settings):
        self.params = settings["params"]
        self.settings = settings

    def save_map(self):
        file_path = model_map_path()
        mapped = {}
        for idx, state in enumerate(self.dependent_vars):
            mapped[state] = ["x", idx + self.nu, self.get_bounds(state)[0], self.get_bounds(state)[1]]

        for idx, ins in enumerate(self.inputs):
            mapped[ins] = ["u", idx, self.get_bounds(ins)[0], self.get_bounds(ins)[1]]

        write_to_yaml(file_path, mapped)

    def integrate(self, z, settings, integration_step):
        # This function should handle params correctly
        return self.discrete_dynamics(z, settings["params"].get_p(), settings, integration_step=integration_step)

    def debug_z_vector(self):
        """
        Helper method to print information about the current z vector.
        Useful for debugging purposes.

        Returns:
        -------
        dict
            Dictionary with control and state values from _z
        """
        if self._z is None:
            print("Warning: _z is None. load() has not been called yet.")
            return None

        import casadi as cd

        result = {}

        # Check if _z is a numeric or symbolic vector
        is_symbolic = isinstance(self._z, (cd.MX, cd.SX))

        if is_symbolic:
            print(f"_z is symbolic with shape {self._z.shape}")
            # Can't extract numeric values from symbolic expressions
            return {"type": "symbolic", "shape": self._z.shape}
        else:
            # Try to treat _z as a numeric array or list
            try:
                # Extract control inputs
                controls = {}
                for i, input_name in enumerate(self.inputs):
                    if i < len(self._z):
                        controls[input_name] = self._z[i]

                # Extract states
                states = {}
                for i, state_name in enumerate(self.dependent_vars):
                    idx = self.nu + i
                    if idx < len(self._z):
                        states[state_name] = self._z[idx]

                result = {
                    "type": "numeric",
                    "controls": controls,
                    "states": states,
                    "length": len(self._z),
                    "expected_length": self.nu + self.state_dimension
                }

                print(f"_z contains {len(self._z)} elements (expected {self.nu + self.state_dimension})")
                print(f"Controls: {controls}")
                print(f"States: {states}")
                return result
            except Exception as e:
                print(f"Error parsing _z: {e}")
                return {"type": "error", "error": str(e)}

    def do_not_use_integration_for_last_n_states(self, n):
        self.state_dimension_integrate = self.state_dimension - n

    def get(self, state_or_input, default=None):
        if state_or_input in self.dependent_vars:
            i = self.dependent_vars.index(state_or_input)
            return self._z[self.nu + i]
        elif state_or_input in self.inputs:
            i = self.inputs.index(state_or_input)
            return self._z[i]
        elif hasattr(self, state_or_input):
            return getattr(self, state_or_input)
        elif default is not None:
            return default
        else:
            raise IOError(
                f"Requested a state or input `{state_or_input}' that was neither a state nor an input for the selected model")

    def set_bounds(self, lower_bound, upper_bound):
        assert len(lower_bound) == len(upper_bound) == len(self.lower_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_bounds(self, state_or_input):
        if state_or_input in self.dependent_vars:
            i = self.dependent_vars.index(state_or_input)
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

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  nx={self.state_dimension}, nu={self.nu},\n"
            f"  dependent_vars={self.dependent_vars},\n"
            f"  inputs={self.inputs},\n"
            f"  lower_bound={self.lower_bound},\n"
            f"  upper_bound={self.upper_bound}\n"
            f")"
        )

    # This is the method that needs to be implemented by subclasses
    def continuous_model(self, x, u, p):
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
        self.state_dimension = 4  # number of states

        self.width = 0.5
        self.length = 0.5

        self.dependent_vars = ["x", "y", "psi", "v"]
        self.inputs = ["a", "w"]
        self.lower_bound = [-2.0, -2.0, -200.0, -200.0, -np.pi * 4, -2.0]
        self.upper_bound = [2.0, 2.0, 200.0, 200.0, np.pi * 4, 3.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a)

class ContouringSecondOrderUnicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.state_dimension = 5  # number of states

        self.dependent_vars = ["x", "y", "psi", "v", "spline"]
        self.inputs = ["a", "w"]

        self.width = 0.5
        self.length = 0.5

        self.lr = .5
        self.lf = .5

        # w = 0.8
        self.lower_bound = [-2.0, -0.8, -2000.0, -2000.0, -np.pi * 6, -0.01, -1.0]
        self.upper_bound = [2.0, 0.8, 2000.0, 2000.0, np.pi * 6, 3.0, 10000.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a, v)

class ContouringSecondOrderUnicycleModelCurvatureAware(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.state_dimension = 5  # number of states

        self.dependent_vars = ["x", "y", "psi", "v", "spline"]
        self.inputs = ["a", "w"]

        self.do_not_use_integration_for_last_n_states(n=1)

        self.lower_bound = [-4.0, -0.8, -2000.0, -2000.0, -pi * 4, -0.01, -1.0]
        self.upper_bound = [4.0, 0.8, 2000.0, 2000.0, pi * 4, 3.0, 10000.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a)

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        x = self.get_x()

        pos_x = x[0]
        pos_y = x[1]
        s = x[-1]


        path = ScipySplineWrapper(self.params, self.settings["contouring"]["get_num_segments"], s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
        path_ddx, path_ddy = path.deriv2(s)

        # Contour = n_vec
        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        # dp = np.array([integrated_states[0] - pos_x, integrated_states[1] - pos_y])
        # t_vec = np.array([path_dx_normalized, path_dy_normalized])
        # n_vec = np.array([path_dy_normalized, -path_dx_normalized])


        dp = integrated_states[0:2] - cd.vertcat(pos_x, pos_y)
        t_vec = cd.vertcat(path_dx_normalized, path_dy_normalized)
        vn_t = dp.T @ cd.vertcat(path_dy_normalized, -path_dx_normalized)
        vt_t = dp.dot(t_vec)

        R = 1.0 / path.get_curvature(s)  # max(R) = 1 / 0.0001
        R = cd.fmax(R, 1e5)

        theta = cd.atan2(vt_t, R - contour_error - vn_t)

        return cd.vertcat(integrated_states, s + R * theta)


class ContouringSecondOrderUnicycleModelWithSlack(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 2  # number of control variables
        self.state_dimension = 6  # number of states

        self.dependent_vars = ["x", "y", "psi", "v", "spline", "slack"]
        self.inputs = ["a", "w"]

        self.lower_bound = [-2.0, -0.8, -2000.0, -2000.0, -np.pi * 4, -0.01, -1.0, 0.0]  # v -0.01
        self.upper_bound = [2.0, 0.8, 2000.0, 2000.0, np.pi * 4, 3.0, 10000.0, 5000.0]  # w 0.8

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]

        return cd.vertcat(v * cd.cos(psi), v * cd.sin(psi), w, a, v, 0.0)

    # NOTE: No initialization for slack variable
    def get_xinit(self):
        return range(self.nu, self.get_nvar() - 1)


# Bicycle model with dynamic steering
class SecondOrderBicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 3
        self.state_dimension = 6

        self.dependent_vars = ["x", "y", "psi", "v", "delta", "spline"]
        self.inputs = ["a", "w", "slack"]

        # Prius limits: https:#github.com/oscardegroot/lmpcc/blob/prius/lmpccsolver/scripts/systems.py
        # w [-0.2, 0.2] | a [-1.0 1.0]
        # w was 0.5
        # delta was 0.45

        # NOTE: the angle of the vehicle should not be limited to -pi, pi, as the solution will not shift when it is at the border!
        # a was 3.0
        self.lower_bound = [-3.0, -1.5, 0.0, -1.0e6, -1.0e6, -np.pi * 4, -0.01, -0.55, -1.0]
        self.upper_bound = [3.0, 1.5, 1.0e2, 1.0e6, 1.0e6, np.pi * 4, 5.0, 0.55, 5000.0]

    def continuous_model(self, x, u, p):
        self.params = p
        a = u[0]
        w = u[1]
        psi = x[2]
        v = x[3]
        delta = x[4]

        wheel_base = self.params("wheel_base") # 2.79  between front wheel center and rear wheel center
        wheel_tread = self.params("wheel_tread") # 1.64  between left wheel center and right wheel center
        front_overhang = self.params("front_overhang") # 1.0  between front wheel center and vehicle front
        rear_overhang = self.params("rear_overhang") # 1.1  # between rear wheel center and vehicle rear
        left_overhang = self.params("left_overhang") # 0.128  # between left wheel center and vehicle left
        right_overhang = self.params("right_overhang") # 0.128  # between right wheel center and vehicle right

        # NOTE: Mass is equally distributed according to the parameters
        lr = wheel_base / 2.0
        lf = wheel_base / 2.0
        ratio = lr / (lr + lf)
        self.width = 2.25

        beta = cd.arctan(ratio * cd.tan(delta))

        return cd.vertcat(v * cd.cos(psi + beta), v * cd.sin(psi + beta), (v / lr) * cd.sin(beta), a, w, v)


class ScipySplineWrapper:
    """Wrapper class to handle scipy splines for path representation with segment support"""

    def __init__(self, params, get_num_segments_fn, s):
        """Initialize spline from parameters with segment support

        Args:
            params: Path parameters (assumed to contain x and y coordinates)
            get_num_segments_fn: Function to get number of segments
            s: Path parameter
        """
        self.params = params
        self.num_segments = get_num_segments_fn()

        # Extract x and y coordinates from params
        # This assumes params contains path points in some format
        # You'll need to adjust this based on your actual param structure
        x_coords = params["x_points"]
        y_coords = params["y_points"]

        # Handle segmentation
        self.segment_length = 1.0 / self.num_segments

        # Create parameter vector for the entire path
        t = np.linspace(0, 1, len(x_coords))

        # Create cubic splines for the entire path
        self.tck_x = interpolate.splrep(t, x_coords, s=0)
        self.tck_y = interpolate.splrep(t, y_coords, s=0)

        # Pre-compute some values for efficiency
        points_per_segment = len(x_coords) // self.num_segments
        self.segment_indices = [i * points_per_segment for i in range(self.num_segments + 1)]
        self.segment_indices[-1] = len(x_coords)  # Ensure the last segment includes all remaining points

    def at(self, s):
        """Get x,y position on spline at parameter s"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # Evaluate splines
        x = float(interpolate.splev(t, self.tck_x))
        y = float(interpolate.splev(t, self.tck_y))

        return x, y

    def deriv_normalized(self, s):
        """Get normalized tangent vector at parameter s"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # Get derivatives
        dx = float(interpolate.splev(t, self.tck_x, der=1))
        dy = float(interpolate.splev(t, self.tck_y, der=1))

        # Normalize
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm < 1e-10:
            norm = 1e-10  # Avoid division by zero

        dx_normalized = dx / norm
        dy_normalized = dy / norm

        return dx_normalized, dy_normalized

    def deriv2(self, s):
        """Get second derivatives at parameter s"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # Get second derivatives
        ddx = float(interpolate.splev(t, self.tck_x, der=2))
        ddy = float(interpolate.splev(t, self.tck_y, der=2))

        return ddx, ddy

    def get_curvature(self, s):
        """Calculate curvature at parameter s with improved numerical stability"""
        # Identify which segment this s belongs to
        segment_idx = min(int(s * self.num_segments), self.num_segments - 1)

        # Map s to parameter range [0,1] for the entire spline
        t = min(max(s, 0), 1)  # Clamp to [0,1]

        # First derivatives
        dx = float(interpolate.splev(t, self.tck_x, der=1))
        dy = float(interpolate.splev(t, self.tck_y, der=1))

        # Second derivatives
        ddx = float(interpolate.splev(t, self.tck_x, der=2))
        ddy = float(interpolate.splev(t, self.tck_y, der=2))

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = abs(dx * ddy - dy * ddx)
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2)

        # Improved numerical stability
        if denominator < 1e-6:
            # For very small denominators, return a stable but small curvature
            return 1e-4  # Minimum curvature (maximum radius)

        curvature = numerator / denominator

        # Limit maximum curvature to avoid numerical issues
        # Use min to cap the maximum curvature (minimum radius)
        return min(max(curvature, 1e-4), 10.0)  # Bounded between 1e-4 and 10

# Bicycle model with dynamic steering
class CurvatureAwareSecondOrderBicycleModel(DynamicsModel):

    def __init__(self):
        super().__init__()
        self.nu = 3
        self.state_dimension = 6

        self.dependent_vars = ["x", "y", "psi", "v", "delta", "spline"]
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

    def continuous_model(self, x, u, p):
        self.params = p
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

        return cd.vertcat(v * cd.cos(psi + beta), v * cd.sin(psi + beta), (v / self.lr) * cd.sin(beta), a, w)

    def model_discrete_dynamics(self, z, integrated_states, **kwargs):
        x = self.get_x()

        pos_x = x[0]
        pos_y = x[1]
        s = x[-1]

        # Ensure s is within valid range
        s = cd.fmin(cd.fmax(s, 0.0), 0.999)  # Clamp slightly inside [0,1]

        path = ScipySplineWrapper(self.params, self.settings["contouring"]["get_num_segments"], s)
        path_x, path_y = path.at(s)
        path_dx_normalized, path_dy_normalized = path.deriv_normalized(s)
        path_ddx, path_ddy = path.deriv2(s)

        # Contour = n_vec
        contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)

        dp = integrated_states[0:2] - cd.vertcat(pos_x, pos_y)
        t_vec = cd.vertcat(path_dx_normalized, path_dy_normalized)

        # Safer dot product calculations
        vn_t = dp.T @ cd.vertcat(path_dy_normalized, -path_dx_normalized)
        vt_t = dp.dot(t_vec)

        # Get curvature with improved numerical stability
        curvature = path.get_curvature(s)

        # Safeguard radius calculation with a minimum curvature
        curvature = cd.fmax(curvature, 1e-5)  # Ensure curvature is not too small
        R = 1.0 / curvature

        # Use a more conservative radius limit
        # Avoid extremely large radii that can cause precision issues
        R = cd.fmin(R, 1e4)  # Cap at 10 km radius

        # Safeguard theta calculation
        # Add a small epsilon to denominator to prevent division by zero
        theta = cd.atan2(vt_t, cd.fmax(R - contour_error - vn_t, 1e-6))

        # Bound theta to prevent extreme values
        theta = cd.fmin(cd.fmax(theta, -0.5), 0.5)

        # Ensure s + R * theta stays within bounds
        new_s = cd.fmin(cd.fmax(s + R * theta, 0.0), 0.999)

        return cd.vertcat(integrated_states, new_s)