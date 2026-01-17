"""
Optimization solver for scenario-based MPC.

Provides a clean interface for solving the trajectory optimization problem
with scenario constraints.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .types import (
    EgoState,
    EgoInput,
    CollisionConstraint,
    MPCResult,
)
from .config import ScenarioMPCConfig


@dataclass
class SolverOptions:
    """Solver configuration options."""
    max_iter: int = 500
    tolerance: float = 1e-4
    print_level: int = 0
    warm_start: bool = True


class ScenarioMPCSolver:
    """
    Nonlinear optimization solver for scenario-based MPC.

    Formulates and solves:
        min  J(x, u)
        s.t. x_{k+1} = f(x_k, u_k)      (dynamics)
             x_0 = x_init                (initial state)
             a^T @ p_k >= b_k            (collision constraints)
             u_min <= u <= u_max         (input bounds)
             x_min <= x <= x_max         (state bounds)
    """

    def __init__(self, config: ScenarioMPCConfig, options: Optional[SolverOptions] = None):
        """
        Initialize solver.

        Args:
            config: MPC configuration
            options: Solver options
        """
        self.config = config
        self.options = options or SolverOptions()

        # Check for CasADi
        try:
            import casadi as cs
            self.cs = cs
            self.has_casadi = True
        except ImportError:
            self.has_casadi = False

        # Warmstart storage
        self.X_warm = None
        self.U_warm = None

    def solve(
        self,
        initial_state: EgoState,
        goal: np.ndarray,
        constraints: List[CollisionConstraint],
        reference_velocity: float = 2.0,
        reference_trajectory: Optional[List[EgoState]] = None
    ) -> MPCResult:
        """
        Solve the optimization problem.

        Args:
            initial_state: Initial ego state
            goal: Goal position [x, y]
            constraints: List of collision constraints
            reference_velocity: Desired velocity
            reference_trajectory: Optional reference for initialization

        Returns:
            MPCResult with solution
        """
        if self.has_casadi:
            return self._solve_casadi(
                initial_state, goal, constraints,
                reference_velocity, reference_trajectory
            )
        else:
            return self._solve_scipy(
                initial_state, goal, constraints,
                reference_velocity, reference_trajectory
            )

    def _solve_casadi(
        self,
        initial_state: EgoState,
        goal: np.ndarray,
        constraints: List[CollisionConstraint],
        reference_velocity: float,
        reference_trajectory: Optional[List[EgoState]]
    ) -> MPCResult:
        """Solve using CasADi/IPOPT."""
        cs = self.cs
        cfg = self.config
        N = cfg.horizon
        dt = cfg.dt

        # Create optimization problem
        opti = cs.Opti()

        # Decision variables
        # States: [x, y, theta, v] for N+1 timesteps
        X = opti.variable(4, N + 1)
        # Controls: [a, w] for N timesteps
        U = opti.variable(2, N)

        # Extract state components for readability
        x = X[0, :]
        y = X[1, :]
        theta = X[2, :]
        v = X[3, :]

        # Extract control components
        a = U[0, :]
        w = U[1, :]

        # === Constraints ===

        # Initial state
        opti.subject_to(X[:, 0] == initial_state.to_array())

        # Dynamics (RK4 integration)
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = self._rk4_step(x_k, u_k, dt)
            opti.subject_to(X[:, k + 1] == x_next)

        # Input bounds
        opti.subject_to(opti.bounded(
            cfg.min_acceleration, a, cfg.max_acceleration
        ))
        opti.subject_to(opti.bounded(
            -cfg.max_steering_rate, w, cfg.max_steering_rate
        ))

        # State bounds
        opti.subject_to(opti.bounded(-0.1, v, 5.0))  # Velocity limits

        # Collision constraints (scenario-based)
        for constraint in constraints:
            k = constraint.k
            if k > N:
                continue

            # Linear constraint: a^T @ p >= b
            opti.subject_to(
                constraint.a[0] * x[k] + constraint.a[1] * y[k] >= constraint.b
            )

        # === Objective ===
        cost = 0

        # Terminal goal cost
        cost += cfg.goal_weight * 10 * (
            (x[N] - goal[0])**2 + (y[N] - goal[1])**2
        )

        # Running goal cost
        for k in range(N):
            cost += cfg.goal_weight * (
                (x[k] - goal[0])**2 + (y[k] - goal[1])**2
            )

        # Velocity tracking
        for k in range(N + 1):
            cost += cfg.velocity_weight * (v[k] - reference_velocity)**2

        # Control effort
        for k in range(N):
            cost += cfg.acceleration_weight * a[k]**2
            cost += cfg.steering_weight * w[k]**2

        # Control smoothness (jerk penalty)
        for k in range(N - 1):
            cost += 0.01 * (a[k+1] - a[k])**2
            cost += 0.01 * (w[k+1] - w[k])**2

        opti.minimize(cost)

        # === Solver setup ===
        opts = {
            'ipopt.print_level': self.options.print_level,
            'print_time': 0,
            'ipopt.max_iter': self.options.max_iter,
            'ipopt.tol': self.options.tolerance,
            'ipopt.warm_start_init_point': 'yes' if self.options.warm_start else 'no',
        }
        opti.solver('ipopt', opts)

        # === Initial guess ===
        if self.options.warm_start and self.X_warm is not None:
            # Shift warmstart forward
            X_init = np.zeros((4, N + 1))
            X_init[:, :-1] = self.X_warm[:, 1:]
            X_init[:, -1] = self.X_warm[:, -1]

            U_init = np.zeros((2, N))
            U_init[:, :-1] = self.U_warm[:, 1:]
            U_init[:, -1] = self.U_warm[:, -1]

            # Update initial state
            X_init[:, 0] = initial_state.to_array()

            opti.set_initial(X, X_init)
            opti.set_initial(U, U_init)

        elif reference_trajectory is not None:
            # Use reference trajectory
            for k in range(min(N + 1, len(reference_trajectory))):
                opti.set_initial(X[:, k], reference_trajectory[k].to_array())

        # === Solve ===
        try:
            sol = opti.solve()

            # Extract solution
            X_opt = sol.value(X)
            U_opt = sol.value(U)

            # Store for warmstart
            self.X_warm = X_opt
            self.U_warm = U_opt

            # Build result
            ego_trajectory = [
                EgoState.from_array(X_opt[:, k]) for k in range(N + 1)
            ]
            control_inputs = [
                EgoInput.from_array(U_opt[:, k]) for k in range(N)
            ]

            return MPCResult(
                success=True,
                ego_trajectory=ego_trajectory,
                control_inputs=control_inputs,
                cost=sol.value(cost)
            )

        except Exception as e:
            # Solver failed
            return self._generate_fallback(initial_state)

    def _rk4_step(self, x, u, dt):
        """RK4 integration step using CasADi."""
        cs = self.cs

        def f(state, control):
            """Continuous dynamics."""
            return cs.vertcat(
                state[3] * cs.cos(state[2]),  # dx = v * cos(theta)
                state[3] * cs.sin(state[2]),  # dy = v * sin(theta)
                control[1],                    # dtheta = w
                control[0]                     # dv = a
            )

        k1 = f(x, u)
        k2 = f(x + dt/2 * k1, u)
        k3 = f(x + dt/2 * k2, u)
        k4 = f(x + dt * k3, u)

        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def _solve_scipy(
        self,
        initial_state: EgoState,
        goal: np.ndarray,
        constraints: List[CollisionConstraint],
        reference_velocity: float,
        reference_trajectory: Optional[List[EgoState]]
    ) -> MPCResult:
        """Fallback solver using scipy."""
        from scipy.optimize import minimize

        cfg = self.config
        N = cfg.horizon

        # Decision variables: [U_flat] = [a_0, w_0, a_1, w_1, ...]
        n_vars = 2 * N

        # Initial guess
        x0 = np.zeros(n_vars)

        def objective(z):
            """Objective function."""
            # Simulate trajectory
            traj = self._simulate_trajectory(initial_state, z, N)

            cost = 0.0

            # Goal cost
            for state in traj:
                cost += cfg.goal_weight * (
                    (state.x - goal[0])**2 + (state.y - goal[1])**2
                )

            # Velocity cost
            for state in traj:
                cost += cfg.velocity_weight * (state.v - reference_velocity)**2

            # Control cost
            for k in range(N):
                a = z[2*k]
                w = z[2*k + 1]
                cost += cfg.acceleration_weight * a**2
                cost += cfg.steering_weight * w**2

            return cost

        # Bounds
        bounds = []
        for _ in range(N):
            bounds.append((cfg.min_acceleration, cfg.max_acceleration))  # a
            bounds.append((-cfg.max_steering_rate, cfg.max_steering_rate))  # w

        # Solve
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.options.max_iter}
        )

        # Extract solution
        z_opt = result.x
        traj = self._simulate_trajectory(initial_state, z_opt, N)
        inputs = [
            EgoInput(a=z_opt[2*k], delta=z_opt[2*k+1])
            for k in range(N)
        ]

        return MPCResult(
            success=result.success,
            ego_trajectory=traj,
            control_inputs=inputs,
            cost=result.fun
        )

    def _simulate_trajectory(
        self,
        initial_state: EgoState,
        z: np.ndarray,
        N: int
    ) -> List[EgoState]:
        """Simulate trajectory given control sequence."""
        dt = self.config.dt
        traj = [initial_state]
        state = initial_state.to_array()

        for k in range(N):
            a = z[2*k]
            w = z[2*k + 1]
            u = np.array([a, w])

            # Euler integration
            state = state + dt * np.array([
                state[3] * np.cos(state[2]),
                state[3] * np.sin(state[2]),
                w,
                a
            ])

            traj.append(EgoState.from_array(state))

        return traj

    def _generate_fallback(self, initial_state: EgoState) -> MPCResult:
        """Generate safe fallback (braking)."""
        N = self.config.horizon
        dt = self.config.dt

        traj = [initial_state]
        inputs = []
        state = initial_state

        for _ in range(N):
            # Brake
            inp = EgoInput(a=-1.0, delta=0.0)
            inputs.append(inp)

            # Propagate
            x = state.to_array()
            x_next = x + dt * np.array([
                x[3] * np.cos(x[2]),
                x[3] * np.sin(x[2]),
                0,
                -1.0
            ])
            x_next[3] = max(0, x_next[3])  # No negative velocity
            state = EgoState.from_array(x_next)
            traj.append(state)

        return MPCResult(
            success=False,
            ego_trajectory=traj,
            control_inputs=inputs,
            cost=float('inf')
        )

    def reset_warmstart(self):
        """Reset warmstart data."""
        self.X_warm = None
        self.U_warm = None
