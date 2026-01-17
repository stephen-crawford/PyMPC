"""
Scenario-based constraint module for MPC.

This module integrates the scenario_mpc approach with the existing
constraint module framework, enabling scenario-based collision avoidance.
"""

import numpy as np
from typing import List, Dict, Optional, Any
import casadi as cd

from modules.constraints.base_constraint import BaseConstraint
from planning.scenario_adapter import (
    ScenarioDataAdapter,
    state_to_ego_state,
    dynamic_obstacle_to_obstacle_state,
    collision_constraint_to_dict,
)
from utils.utils import LOG_DEBUG, LOG_INFO, LOG_WARN
from utils.const import CONSTRAINT


class ScenarioConstraint(BaseConstraint):
    """
    Scenario-based collision avoidance constraint module.

    Implements the scenario-based MPC approach from guide.md:
    - Samples scenarios from mode distributions
    - Computes linearized collision constraints
    - Supports scenario pruning for computational efficiency

    Integrates with the existing ModuleManager and solver framework.
    """

    def __init__(self, settings=None):
        """
        Initialize the scenario constraint module.

        Args:
            settings: Optional settings dictionary
        """
        super().__init__(settings)
        self.name = "scenario_constraint"
        self.module_type = CONSTRAINT

        # Scenario MPC components (lazy loaded)
        self._scenario_mpc = None
        self._data_adapter = None
        self._initialized = False

        # Configuration parameters
        self._num_scenarios = self.get_config_value("scenario_constraint.num_scenarios", 10)
        self._confidence_level = self.get_config_value("scenario_constraint.confidence_level", 0.95)
        self._ego_radius = self.get_config_value("scenario_constraint.ego_radius", 0.5)
        self._obstacle_radius = self.get_config_value("scenario_constraint.obstacle_radius", 0.5)
        self._weight_type = self.get_config_value("scenario_constraint.weight_type", "frequency")
        self._enable_pruning = self.get_config_value("scenario_constraint.enable_pruning", True)

        # Cache for computed constraints
        self._cached_constraints: List[Dict] = []
        self._current_timestep = 0

        LOG_INFO(f"ScenarioConstraint initialized: {self._num_scenarios} scenarios, "
                 f"confidence={self._confidence_level}")

    def _lazy_init(self):
        """Lazily initialize scenario MPC components."""
        if self._initialized:
            return

        try:
            from scenario_mpc import (
                AdaptiveScenarioMPC,
                ScenarioMPCConfig,
                WeightType,
            )
            from scenario_mpc.dynamics import create_obstacle_mode_models

            # Map weight type string to enum
            weight_map = {
                "uniform": WeightType.UNIFORM,
                "recency": WeightType.RECENCY,
                "frequency": WeightType.FREQUENCY,
            }
            weight_type = weight_map.get(self._weight_type, WeightType.FREQUENCY)

            # Create configuration
            config = ScenarioMPCConfig(
                horizon=20,  # Will be updated from data
                dt=0.1,      # Will be updated from data
                num_scenarios=self._num_scenarios,
                confidence_level=self._confidence_level,
                weight_type=weight_type,
                ego_radius=self._ego_radius,
                obstacle_radius=self._obstacle_radius,
            )

            self._scenario_mpc = AdaptiveScenarioMPC(config)
            self._available_modes = create_obstacle_mode_models(config.dt)
            self._initialized = True
            LOG_DEBUG("ScenarioConstraint: Lazy initialization complete")

        except ImportError as e:
            LOG_WARN(f"ScenarioConstraint: Could not import scenario_mpc: {e}")
            self._initialized = False

    def update(self, state, data):
        """
        Update constraint module with current state and data.

        Args:
            state: Current ego state
            data: Current data object
        """
        LOG_DEBUG(f"ScenarioConstraint.update called")

        # Lazy initialize
        self._lazy_init()
        if not self._initialized or self._scenario_mpc is None:
            LOG_WARN("ScenarioConstraint: Not initialized, skipping update")
            return

        # Create data adapter
        self._data_adapter = ScenarioDataAdapter(data)
        self._data_adapter.set_available_modes(self._available_modes)

        # Update scenario MPC config with actual horizon/timestep
        horizon = self._data_adapter.get_horizon()
        timestep = self._data_adapter.get_timestep()
        self._scenario_mpc.config.horizon = horizon
        self._scenario_mpc.config.dt = timestep

        # Convert obstacles and initialize mode histories
        obstacles = {}
        if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
            for i, obs in enumerate(data.dynamic_obstacles):
                obs_id = obs.index if hasattr(obs, 'index') else i
                obs_state = dynamic_obstacle_to_obstacle_state(obs)
                obstacles[obs_id] = obs_state

                # Initialize obstacle in scenario MPC
                if obs_id not in self._scenario_mpc.mode_histories:
                    self._scenario_mpc.initialize_obstacle(obs_id)

                # Update mode observation (default to constant_velocity)
                self._scenario_mpc.update_mode_observation(
                    obs_id,
                    "constant_velocity",
                    timestep=self._current_timestep
                )

        # Convert ego state
        ego_state = state_to_ego_state(state)

        # Get goal
        goal = self._data_adapter.get_goal()
        if goal is None:
            goal = np.array([10.0, 0.0])  # Default goal

        # Pre-compute scenarios and constraints
        if obstacles:
            self._precompute_constraints(ego_state, obstacles, goal)

        self._current_timestep += 1

    def _precompute_constraints(self, ego_state, obstacles, goal):
        """
        Pre-compute scenario-based constraints.

        Args:
            ego_state: Current ego state
            obstacles: Dictionary of obstacle states
            goal: Goal position
        """
        from scenario_mpc.scenario_sampler import sample_scenarios
        from scenario_mpc.collision_constraints import compute_linearized_constraints
        from scenario_mpc.scenario_pruning import prune_dominated_scenarios

        LOG_DEBUG(f"Pre-computing scenario constraints for {len(obstacles)} obstacles")

        # Get mode histories
        histories = {}
        for obs_id in obstacles:
            history = self._scenario_mpc.mode_histories.get(obs_id)
            if history is not None:
                histories[obs_id] = history

        if not histories:
            self._cached_constraints = []
            return

        # Sample scenarios
        try:
            scenarios = sample_scenarios(
                obstacles,
                histories,
                horizon=self._scenario_mpc.config.horizon,
                num_scenarios=self._num_scenarios,
            )
        except Exception as e:
            LOG_WARN(f"ScenarioConstraint: Error sampling scenarios: {e}")
            self._cached_constraints = []
            return

        # Generate reference trajectory (simple straight line for constraint computation)
        ego_traj = self._generate_reference_trajectory(ego_state)

        # Prune dominated scenarios if enabled
        if self._enable_pruning and len(scenarios) > 1:
            try:
                scenarios = prune_dominated_scenarios(
                    scenarios,
                    ego_traj,
                    ego_radius=self._ego_radius,
                    obstacle_radius=self._obstacle_radius,
                )
            except Exception as e:
                LOG_DEBUG(f"ScenarioConstraint: Pruning failed: {e}")

        # Compute linearized constraints
        try:
            constraints = compute_linearized_constraints(
                ego_traj,
                scenarios,
                ego_radius=self._ego_radius,
                obstacle_radius=self._obstacle_radius,
            )
            # Convert to dict format
            self._cached_constraints = [
                collision_constraint_to_dict(c) for c in constraints
            ]
            LOG_DEBUG(f"Computed {len(self._cached_constraints)} scenario constraints")

        except Exception as e:
            LOG_WARN(f"ScenarioConstraint: Error computing constraints: {e}")
            self._cached_constraints = []

    def _generate_reference_trajectory(self, ego_state):
        """
        Generate a simple reference trajectory for constraint computation.

        Args:
            ego_state: Current ego state

        Returns:
            List of EgoState objects
        """
        from scenario_mpc.types import EgoState

        horizon = self._scenario_mpc.config.horizon
        dt = self._scenario_mpc.config.dt

        trajectory = [ego_state]
        state = ego_state

        for _ in range(horizon):
            # Simple constant velocity propagation
            next_state = EgoState(
                x=state.x + state.v * np.cos(state.theta) * dt,
                y=state.y + state.v * np.sin(state.theta) * dt,
                theta=state.theta,
                v=state.v
            )
            trajectory.append(next_state)
            state = next_state

        return trajectory

    def calculate_constraints(self, state, data, stage_idx):
        """
        Calculate symbolic constraint expressions for this stage.

        CRITICAL: Returns CasADi symbolic expressions for MPC optimization.

        Args:
            state: Dictionary of symbolic state variables from solver
            data: Data object
            stage_idx: Current stage index

        Returns:
            List of constraint dictionaries or symbolic expressions
        """
        if not self._cached_constraints:
            return []

        # Filter constraints for this stage
        stage_constraints = [
            c for c in self._cached_constraints if c.get('k', 0) == stage_idx
        ]

        if not stage_constraints:
            return []

        # Get symbolic position variables
        x_sym = state.get('x') if isinstance(state, dict) else state.get('x')
        y_sym = state.get('y') if isinstance(state, dict) else state.get('y')

        if x_sym is None or y_sym is None:
            LOG_DEBUG(f"ScenarioConstraint: No symbolic variables at stage {stage_idx}")
            return stage_constraints  # Return as-is for fallback handling

        # Convert to symbolic CasADi constraints
        symbolic_constraints = []
        for c in stage_constraints:
            a1 = c['a1']
            a2 = c['a2']
            b = c['b']

            # Constraint: a1*x + a2*y >= b
            # In form for solver: a1*x + a2*y - b >= 0
            expr = a1 * x_sym + a2 * y_sym - b

            symbolic_constraints.append({
                'type': 'symbolic_expression',
                'expression': expr,
                'lower_bound': 0.0,  # expr >= 0
                'upper_bound': float('inf'),
                'metadata': {
                    'obstacle_id': c.get('obstacle_id'),
                    'scenario_id': c.get('scenario_id'),
                    'stage': stage_idx,
                }
            })

        LOG_DEBUG(f"ScenarioConstraint: {len(symbolic_constraints)} constraints at stage {stage_idx}")
        return symbolic_constraints

    def is_data_ready(self, data) -> bool:
        """
        Check if required data is available.

        Args:
            data: Data object

        Returns:
            True if data is ready
        """
        # Scenario constraints can work with or without obstacles
        return True

    def get_parameters(self, data, stage_idx) -> Dict[str, Any]:
        """
        Get parameters for this stage.

        Args:
            data: Data object
            stage_idx: Stage index

        Returns:
            Dictionary of parameters
        """
        # Count constraints at this stage
        num_constraints = sum(
            1 for c in self._cached_constraints if c.get('k', 0) == stage_idx
        )
        return {
            'scenario_num_constraints': num_constraints,
            'scenario_total_scenarios': self._num_scenarios,
        }

    def get_dependencies(self) -> List[str]:
        """Get module dependencies."""
        return []

    def reset(self):
        """Reset the module state."""
        self._cached_constraints = []
        self._current_timestep = 0
        if self._scenario_mpc is not None:
            self._scenario_mpc.reset()
        LOG_DEBUG("ScenarioConstraint: Reset complete")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get module statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_cached_constraints': len(self._cached_constraints),
            'current_timestep': self._current_timestep,
            'initialized': self._initialized,
        }

        if self._scenario_mpc is not None:
            mpc_stats = self._scenario_mpc.get_statistics()
            stats.update(mpc_stats)

        return stats

    def get_visualizer(self):
        """Return visualizer for this module."""
        return ScenarioConstraintVisualizer(self)


class ScenarioConstraintVisualizer:
    """Visualizer for scenario constraints."""

    def __init__(self, module: ScenarioConstraint):
        self.module = module

    def visualize(self, state, data, stage_idx=0):
        """
        Visualize scenario constraints.

        Args:
            state: Current state
            data: Data object
            stage_idx: Stage to visualize
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            from matplotlib.lines import Line2D

            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot ego position
            if state is not None:
                ego_x = state.get('x')
                ego_y = state.get('y')
                if ego_x is not None and ego_y is not None:
                    ax.plot(ego_x, ego_y, 'bo', markersize=10, label='Ego')
                    ego_circle = Circle(
                        (ego_x, ego_y),
                        self.module._ego_radius,
                        fill=False,
                        color='blue',
                        linestyle='--'
                    )
                    ax.add_patch(ego_circle)

            # Plot obstacles
            if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
                for i, obs in enumerate(data.dynamic_obstacles):
                    ox, oy = obs.position[0], obs.position[1]
                    radius = obs.radius if hasattr(obs, 'radius') else self.module._obstacle_radius
                    ax.plot(ox, oy, 'ro', markersize=8)
                    obs_circle = Circle(
                        (ox, oy),
                        radius,
                        fill=False,
                        color='red',
                        linestyle='--'
                    )
                    ax.add_patch(obs_circle)

            # Plot constraint lines
            constraints = self.module._cached_constraints
            stage_constraints = [c for c in constraints if c.get('k', 0) == stage_idx]

            for c in stage_constraints[:10]:  # Limit to first 10
                a1, a2, b = c['a1'], c['a2'], c['b']
                # Plot halfspace boundary line
                if abs(a2) > 1e-6:
                    x_vals = np.linspace(-5, 15, 100)
                    y_vals = (b - a1 * x_vals) / a2
                    ax.plot(x_vals, y_vals, 'g-', alpha=0.3, linewidth=0.5)

            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title(f'Scenario Constraints (Stage {stage_idx})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            LOG_WARN(f"ScenarioConstraintVisualizer: Error: {e}")
