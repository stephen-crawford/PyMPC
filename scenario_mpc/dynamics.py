"""
Ego vehicle dynamics model.

Implements the unicycle model with acceleration and steering rate inputs.
Supports both numeric and symbolic (CasADi) computation.
"""

import numpy as np
from typing import Union, Optional
from .types import EgoState, EgoInput

# Try to import CasADi for symbolic computation
try:
    import casadi as cs
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


class EgoDynamics:
    """
    Unicycle dynamics model for ego vehicle.

    State: x = [x, y, theta, v]
    Input: u = [a, w] (acceleration, angular velocity)

    Continuous dynamics:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = w
        dv/dt = a
    """

    def __init__(self, dt: float = 0.1):
        """
        Initialize dynamics model.

        Args:
            dt: Timestep for discrete integration [s]
        """
        self.dt = dt
        self.state_dim = 4  # [x, y, theta, v]
        self.input_dim = 2  # [a, w]

    def continuous_dynamics(
        self,
        state: Union[np.ndarray, "cs.MX"],
        input: Union[np.ndarray, "cs.MX"]
    ) -> Union[np.ndarray, "cs.MX"]:
        """
        Compute continuous-time state derivative.

        Args:
            state: State vector [x, y, theta, v]
            input: Input vector [a, w]

        Returns:
            State derivative [dx, dy, dtheta, dv]
        """
        # Extract state components
        theta = state[2]
        v = state[3]

        # Extract inputs
        a = input[0]
        w = input[1]

        # Check if using CasADi
        if HAS_CASADI and (isinstance(state, cs.MX) or isinstance(input, cs.MX)):
            return cs.vertcat(
                v * cs.cos(theta),
                v * cs.sin(theta),
                w,
                a
            )
        else:
            return np.array([
                v * np.cos(theta),
                v * np.sin(theta),
                w,
                a
            ])

    def discrete_dynamics(
        self,
        state: Union[np.ndarray, "cs.MX"],
        input: Union[np.ndarray, "cs.MX"],
        dt: Optional[float] = None
    ) -> Union[np.ndarray, "cs.MX"]:
        """
        Compute discrete-time state update using RK4 integration.

        Args:
            state: Current state [x, y, theta, v]
            input: Control input [a, w]
            dt: Timestep (uses self.dt if not provided)

        Returns:
            Next state after dt
        """
        if dt is None:
            dt = self.dt

        # RK4 integration
        k1 = self.continuous_dynamics(state, input)
        k2 = self.continuous_dynamics(state + dt / 2 * k1, input)
        k3 = self.continuous_dynamics(state + dt / 2 * k2, input)
        k4 = self.continuous_dynamics(state + dt * k3, input)

        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def propagate(
        self,
        state: EgoState,
        input: EgoInput,
        dt: Optional[float] = None
    ) -> EgoState:
        """
        Propagate ego state forward one timestep.

        Args:
            state: Current ego state
            input: Control input
            dt: Timestep (uses self.dt if not provided)

        Returns:
            Next ego state
        """
        x = state.to_array()
        u = input.to_array()
        x_next = self.discrete_dynamics(x, u, dt)
        return EgoState.from_array(x_next)

    def rollout(
        self,
        initial_state: EgoState,
        inputs: list,
        dt: Optional[float] = None
    ) -> list:
        """
        Roll out trajectory from initial state with given inputs.

        Args:
            initial_state: Starting ego state
            inputs: List of EgoInput for each timestep
            dt: Timestep (uses self.dt if not provided)

        Returns:
            List of EgoState including initial state (length N+1)
        """
        states = [initial_state]
        current = initial_state

        for u in inputs:
            current = self.propagate(current, u, dt)
            states.append(current)

        return states

    def get_jacobians(
        self,
        state: np.ndarray,
        input: np.ndarray
    ) -> tuple:
        """
        Compute Jacobians of discrete dynamics for linearization.

        Args:
            state: State vector [x, y, theta, v]
            input: Input vector [a, w]

        Returns:
            (A, B) where x_next ≈ A @ x + B @ u + c
        """
        theta = state[2]
        v = state[3]
        dt = self.dt

        # Jacobian w.r.t. state (using RK4 approximation)
        A = np.eye(4)
        A[0, 2] = -v * np.sin(theta) * dt
        A[0, 3] = np.cos(theta) * dt
        A[1, 2] = v * np.cos(theta) * dt
        A[1, 3] = np.sin(theta) * dt

        # Jacobian w.r.t. input
        B = np.zeros((4, 2))
        B[2, 1] = dt  # theta depends on w
        B[3, 0] = dt  # v depends on a

        return A, B


def create_obstacle_mode_models(dt: float = 0.1) -> dict:
    """
    Create standard obstacle mode models.

    Returns dictionary of common modes for obstacle prediction.
    """
    from .types import ModeModel

    modes = {}

    # Constant velocity mode
    A_cv = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    b_cv = np.zeros(4)
    G_cv = np.array([
        [0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt]
    ]) * 0.5  # Scale process noise

    modes["constant_velocity"] = ModeModel(
        mode_id="constant_velocity",
        A=A_cv,
        b=b_cv,
        G=G_cv,
        description="Constant velocity motion"
    )

    # Decelerating mode
    A_dec = A_cv.copy()
    b_dec = np.array([0, 0, -0.5 * dt, -0.5 * dt])  # Deceleration

    modes["decelerating"] = ModeModel(
        mode_id="decelerating",
        A=A_dec,
        b=b_dec,
        G=G_cv,
        description="Decelerating motion"
    )

    # Left turn mode
    omega = 0.3  # Turn rate [rad/s]
    cos_w = np.cos(omega * dt)
    sin_w = np.sin(omega * dt)

    A_left = np.array([
        [1, 0, dt * cos_w, -dt * sin_w],
        [0, 1, dt * sin_w, dt * cos_w],
        [0, 0, cos_w, -sin_w],
        [0, 0, sin_w, cos_w]
    ])
    b_left = np.zeros(4)

    modes["turn_left"] = ModeModel(
        mode_id="turn_left",
        A=A_left,
        b=b_left,
        G=G_cv,
        description="Left turning motion"
    )

    # Right turn mode
    A_right = np.array([
        [1, 0, dt * cos_w, dt * sin_w],
        [0, 1, -dt * sin_w, dt * cos_w],
        [0, 0, cos_w, sin_w],
        [0, 0, -sin_w, cos_w]
    ])
    b_right = np.zeros(4)

    modes["turn_right"] = ModeModel(
        mode_id="turn_right",
        A=A_right,
        b=b_right,
        G=G_cv,
        description="Right turning motion"
    )

    # Lane change left
    A_lc = A_cv.copy()
    b_lc_left = np.array([0, 0.3 * dt, 0, 0])  # Lateral drift left

    modes["lane_change_left"] = ModeModel(
        mode_id="lane_change_left",
        A=A_lc,
        b=b_lc_left,
        G=G_cv,
        description="Lane change left"
    )

    # Lane change right
    b_lc_right = np.array([0, -0.3 * dt, 0, 0])  # Lateral drift right

    modes["lane_change_right"] = ModeModel(
        mode_id="lane_change_right",
        A=A_lc,
        b=b_lc_right,
        G=G_cv,
        description="Lane change right"
    )

    return modes
