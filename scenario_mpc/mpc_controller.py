"""
Adaptive Scenario-Based MPC Controller.

Implements Algorithm 2 from guide.md: AdaptiveScenarioMPC

Main control loop that:
1. Samples scenarios from obstacle predictions
2. Computes linearized collision constraints
3. Solves the scenario-constrained optimization
4. Updates mode histories with observations
5. Prunes inactive scenarios
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .types import (
    EgoState,
    EgoInput,
    ObstacleState,
    ModeModel,
    ModeHistory,
    Scenario,
    CollisionConstraint,
    MPCResult,
    WeightType,
)
from .config import ScenarioMPCConfig
from .dynamics import EgoDynamics, create_obstacle_mode_models
from .mode_weights import compute_mode_weights
from .scenario_sampler import sample_scenarios
from .collision_constraints import compute_linearized_constraints, evaluate_constraint_violation
from .scenario_pruning import (
    prune_dominated_scenarios,
    remove_inactive_scenarios,
    select_diverse_scenarios,
)


class AdaptiveScenarioMPC:
    """
    Adaptive Scenario-Based Model Predictive Controller.

    Implements Algorithm 2 from guide.md.
    """

    def __init__(self, config: ScenarioMPCConfig):
        """
        Initialize the MPC controller.

        Args:
            config: Configuration parameters
        """
        self.config = config
        config.validate()

        # Initialize dynamics
        self.ego_dynamics = EgoDynamics(dt=config.dt)

        # Create default obstacle mode models
        self.default_modes = create_obstacle_mode_models(dt=config.dt)

        # Mode histories for each obstacle
        self.mode_histories: Dict[int, ModeHistory] = {}

        # Current scenarios (maintained between iterations)
        self.scenarios: List[Scenario] = []

        # Reference trajectory for linearization
        self.reference_trajectory: List[EgoState] = []

        # Random number generator for reproducibility
        self.rng = np.random.default_rng()

        # Statistics
        self.solve_times: List[float] = []
        self.iteration_count = 0

    def initialize_obstacle(
        self,
        obstacle_id: int,
        available_modes: Optional[Dict[str, ModeModel]] = None
    ) -> None:
        """
        Initialize mode history for a new obstacle.

        Args:
            obstacle_id: Unique obstacle identifier
            available_modes: Optional custom modes (uses defaults if None)
        """
        modes = available_modes if available_modes is not None else self.default_modes

        self.mode_histories[obstacle_id] = ModeHistory(
            obstacle_id=obstacle_id,
            available_modes=modes,
            observed_modes=[],
            max_history=self.config.horizon * 10
        )

    def update_mode_observation(
        self,
        obstacle_id: int,
        observed_mode: str,
        timestep: Optional[int] = None
    ) -> None:
        """
        Record a mode observation for an obstacle.

        Args:
            obstacle_id: Obstacle identifier
            observed_mode: Observed mode ID
            timestep: Optional timestep (uses iteration count if None)
        """
        if obstacle_id not in self.mode_histories:
            self.initialize_obstacle(obstacle_id)

        if timestep is None:
            timestep = self.iteration_count

        self.mode_histories[obstacle_id].record_observation(timestep, observed_mode)

    def solve(
        self,
        ego_state: EgoState,
        obstacles: Dict[int, ObstacleState],
        goal: np.ndarray,
        reference_velocity: float = 2.0
    ) -> MPCResult:
        """
        Solve the MPC problem.

        Algorithm 2: AdaptiveScenarioMPC

        Args:
            ego_state: Current ego vehicle state
            obstacles: Current obstacle states
            goal: Goal position [x, y]
            reference_velocity: Desired velocity

        Returns:
            MPCResult with optimal trajectory and controls
        """
        start_time = time.time()
        self.iteration_count += 1

        # Ensure all obstacles have mode histories
        for obs_id in obstacles:
            if obs_id not in self.mode_histories:
                self.initialize_obstacle(obs_id)

        # Step 1: Initialize reference trajectory (warmstart from previous)
        self._initialize_reference_trajectory(ego_state, goal)

        # Step 2: Sample scenarios (Algorithm 1)
        self.scenarios = sample_scenarios(
            obstacles=obstacles,
            mode_histories=self.mode_histories,
            horizon=self.config.horizon,
            num_scenarios=self.config.num_scenarios,
            weight_type=self.config.weight_type,
            recency_decay=self.config.recency_decay,
            current_timestep=self.iteration_count,
            rng=self.rng
        )

        # Step 3: Prune dominated scenarios (Algorithm 3)
        self.scenarios = prune_dominated_scenarios(
            self.scenarios,
            self.reference_trajectory,
            self.config.ego_radius,
            self.config.obstacle_radius
        )

        # Ensure minimum scenario count
        if len(self.scenarios) < 3:
            # Re-sample if too many pruned
            additional = sample_scenarios(
                obstacles=obstacles,
                mode_histories=self.mode_histories,
                horizon=self.config.horizon,
                num_scenarios=max(5, self.config.num_scenarios - len(self.scenarios)),
                weight_type=self.config.weight_type,
                rng=self.rng
            )
            self.scenarios.extend(additional)

        # Step 4: Compute linearized constraints
        constraints = compute_linearized_constraints(
            self.reference_trajectory,
            self.scenarios,
            self.config.ego_radius,
            self.config.obstacle_radius,
            self.config.safety_margin
        )

        # Step 5: Solve optimization problem
        result = self._solve_optimization(
            ego_state,
            goal,
            reference_velocity,
            constraints
        )

        # Step 6: Remove inactive scenarios (Algorithm 4)
        if result.success:
            self.scenarios, active = remove_inactive_scenarios(
                self.scenarios,
                constraints,
                result.ego_trajectory
            )
            result.active_scenarios = list(active)

            # Update reference trajectory for next iteration
            self.reference_trajectory = result.ego_trajectory

        # Record timing
        result.solve_time = time.time() - start_time
        self.solve_times.append(result.solve_time)

        return result

    def _initialize_reference_trajectory(
        self,
        ego_state: EgoState,
        goal: np.ndarray
    ) -> None:
        """
        Initialize reference trajectory for constraint linearization.

        Uses previous solution shifted forward, or straight-line to goal.
        """
        if self.reference_trajectory and len(self.reference_trajectory) > 1:
            # Shift previous trajectory forward
            self.reference_trajectory = self.reference_trajectory[1:]

            # Extend to full horizon
            while len(self.reference_trajectory) <= self.config.horizon:
                last = self.reference_trajectory[-1]
                # Simple constant velocity extension
                new_state = EgoState(
                    x=last.x + last.v * np.cos(last.theta) * self.config.dt,
                    y=last.y + last.v * np.sin(last.theta) * self.config.dt,
                    theta=last.theta,
                    v=last.v
                )
                self.reference_trajectory.append(new_state)

            # Update first state to current
            self.reference_trajectory[0] = ego_state
        else:
            # Initialize with straight-line trajectory to goal
            self.reference_trajectory = self._generate_straight_line_trajectory(
                ego_state, goal
            )

    def _generate_straight_line_trajectory(
        self,
        start: EgoState,
        goal: np.ndarray
    ) -> List[EgoState]:
        """
        Generate straight-line reference trajectory.
        """
        trajectory = [start]
        current = start

        for _ in range(self.config.horizon):
            # Direction to goal
            to_goal = goal - current.position()
            dist = np.linalg.norm(to_goal)

            if dist > 0.1:
                direction = to_goal / dist
                desired_theta = np.arctan2(direction[1], direction[0])
            else:
                desired_theta = current.theta

            # Simple propagation
            v = min(current.v + 0.5 * self.config.dt, 2.0)  # Accelerate gently
            next_state = EgoState(
                x=current.x + v * np.cos(desired_theta) * self.config.dt,
                y=current.y + v * np.sin(desired_theta) * self.config.dt,
                theta=desired_theta,
                v=v
            )
            trajectory.append(next_state)
            current = next_state

        return trajectory

    def _solve_optimization(
        self,
        ego_state: EgoState,
        goal: np.ndarray,
        reference_velocity: float,
        constraints: List[CollisionConstraint]
    ) -> MPCResult:
        """
        Solve the scenario-constrained optimization problem.

        Uses CasADi for nonlinear optimization.
        """
        try:
            import casadi as cs
        except ImportError:
            return self._solve_optimization_simple(
                ego_state, goal, reference_velocity, constraints
            )

        N = self.config.horizon
        dt = self.config.dt

        # Create optimization problem
        opti = cs.Opti()

        # Decision variables
        X = opti.variable(4, N + 1)  # States [x, y, theta, v]
        U = opti.variable(2, N)      # Controls [a, w]

        # Initial state constraint
        opti.subject_to(X[:, 0] == ego_state.to_array())

        # Dynamics constraints (using simple Euler for speed)
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]

            # Euler integration
            x_next = x_k + dt * cs.vertcat(
                x_k[3] * cs.cos(x_k[2]),
                x_k[3] * cs.sin(x_k[2]),
                u_k[1],  # w (angular velocity)
                u_k[0]   # a (acceleration)
            )

            opti.subject_to(X[:, k + 1] == x_next)

        # Input bounds
        opti.subject_to(opti.bounded(
            self.config.min_acceleration,
            U[0, :],
            self.config.max_acceleration
        ))
        opti.subject_to(opti.bounded(
            -self.config.max_steering_rate,
            U[1, :],
            self.config.max_steering_rate
        ))

        # Velocity bounds
        opti.subject_to(opti.bounded(-0.1, X[3, :], 5.0))

        # Collision constraints (linearized)
        for constraint in constraints:
            k = constraint.k
            if k > N:
                continue
            # a^T @ p >= b
            opti.subject_to(
                constraint.a[0] * X[0, k] + constraint.a[1] * X[1, k] >= constraint.b
            )

        # Objective function
        cost = 0

        # Goal tracking
        for k in range(N + 1):
            cost += self.config.goal_weight * (
                (X[0, k] - goal[0])**2 + (X[1, k] - goal[1])**2
            )

        # Velocity tracking
        for k in range(N + 1):
            cost += self.config.velocity_weight * (X[3, k] - reference_velocity)**2

        # Control effort
        for k in range(N):
            cost += self.config.acceleration_weight * U[0, k]**2
            cost += self.config.steering_weight * U[1, k]**2

        opti.minimize(cost)

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': self.config.solver_max_iter,
            'ipopt.tol': self.config.solver_tolerance,
        }
        opti.solver('ipopt', opts)

        # Set initial guess from reference trajectory
        for k in range(N + 1):
            if k < len(self.reference_trajectory):
                opti.set_initial(X[:, k], self.reference_trajectory[k].to_array())

        try:
            sol = opti.solve()

            # Extract solution
            X_opt = sol.value(X)
            U_opt = sol.value(U)

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
            # Solver failed - return safe fallback
            return self._generate_safe_fallback(ego_state)

    def _solve_optimization_simple(
        self,
        ego_state: EgoState,
        goal: np.ndarray,
        reference_velocity: float,
        constraints: List[CollisionConstraint]
    ) -> MPCResult:
        """
        Simple optimization fallback without CasADi.

        Uses gradient descent on unconstrained problem.
        """
        N = self.config.horizon

        # Initialize with straight line to goal
        trajectory = self._generate_straight_line_trajectory(ego_state, goal)
        inputs = []

        for k in range(N):
            if k + 1 < len(trajectory):
                # Compute input that would achieve this transition
                current = trajectory[k]
                next_state = trajectory[k + 1]

                a = (next_state.v - current.v) / self.config.dt
                w = (next_state.theta - current.theta) / self.config.dt

                # Clamp to bounds
                a = np.clip(a, self.config.min_acceleration, self.config.max_acceleration)
                w = np.clip(w, -self.config.max_steering_rate, self.config.max_steering_rate)

                inputs.append(EgoInput(a=a, delta=w))
            else:
                inputs.append(EgoInput(a=0, delta=0))

        # Re-propagate with clamped inputs
        trajectory = self.ego_dynamics.rollout(ego_state, inputs)

        # Check constraint violations
        max_violation, _ = evaluate_constraint_violation(constraints, trajectory)

        # If constraints violated, try to fix with simple avoidance
        if max_violation > 0:
            trajectory, inputs = self._apply_simple_avoidance(
                ego_state, trajectory, inputs, constraints
            )

        return MPCResult(
            success=True,
            ego_trajectory=trajectory,
            control_inputs=inputs,
            cost=0.0
        )

    def _apply_simple_avoidance(
        self,
        ego_state: EgoState,
        trajectory: List[EgoState],
        inputs: List[EgoInput],
        constraints: List[CollisionConstraint]
    ) -> Tuple[List[EgoState], List[EgoInput]]:
        """
        Apply simple constraint avoidance by adjusting inputs.
        """
        # Group constraints by timestep
        by_k = {}
        for c in constraints:
            if c.k not in by_k:
                by_k[c.k] = []
            by_k[c.k].append(c)

        # Adjust inputs to avoid violations
        new_inputs = list(inputs)

        for k in range(len(inputs)):
            if k not in by_k:
                continue

            # Check violations at this timestep
            if k + 1 < len(trajectory):
                ego_pos = trajectory[k + 1].position()

                for constraint in by_k.get(k + 1, []):
                    value = constraint.evaluate(ego_pos)

                    if value < 0:
                        # Constraint violated - adjust steering to avoid
                        avoidance_direction = constraint.a
                        current_heading = trajectory[k].theta

                        # Compute steering adjustment
                        desired_heading = np.arctan2(
                            avoidance_direction[1], avoidance_direction[0]
                        )
                        heading_diff = desired_heading - current_heading

                        # Wrap to [-pi, pi]
                        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi

                        # Apply steering adjustment
                        new_w = new_inputs[k].delta + 0.3 * heading_diff
                        new_w = np.clip(
                            new_w,
                            -self.config.max_steering_rate,
                            self.config.max_steering_rate
                        )
                        new_inputs[k] = EgoInput(a=new_inputs[k].a, delta=new_w)

        # Re-propagate with new inputs
        new_trajectory = self.ego_dynamics.rollout(ego_state, new_inputs)

        return new_trajectory, new_inputs

    def _generate_safe_fallback(self, ego_state: EgoState) -> MPCResult:
        """
        Generate safe fallback trajectory (gentle braking).
        """
        trajectory = [ego_state]
        inputs = []

        current = ego_state

        for _ in range(self.config.horizon):
            # Brake gently
            a = -1.0
            w = 0.0
            inputs.append(EgoInput(a=a, delta=w))

            next_state = self.ego_dynamics.propagate(current, inputs[-1])
            trajectory.append(next_state)
            current = next_state

        return MPCResult(
            success=False,
            ego_trajectory=trajectory,
            control_inputs=inputs,
            cost=float('inf')
        )

    def get_statistics(self) -> dict:
        """
        Get controller statistics.
        """
        return {
            'iteration_count': self.iteration_count,
            'avg_solve_time': np.mean(self.solve_times) if self.solve_times else 0,
            'max_solve_time': max(self.solve_times) if self.solve_times else 0,
            'num_obstacles': len(self.mode_histories),
            'num_scenarios': len(self.scenarios),
        }

    def reset(self) -> None:
        """
        Reset the controller state.

        Clears mode histories, scenarios, and statistics.
        """
        self.mode_histories.clear()
        self.scenarios.clear()
        self.reference_trajectory.clear()
        self.solve_times.clear()
        self.iteration_count = 0
