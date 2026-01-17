"""
Type adapters for bridging scenario_mpc types with the existing planning types.

This module provides conversion functions and adapter classes that allow the
new scenario-based MPC components to work seamlessly with the existing codebase.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

# Import existing types
from planning.types import (
    State,
    DynamicObstacle,
    StaticObstacle,
    PredictionStep as LegacyPredictionStep,
    Prediction,
    PredictionType,
    Data,
    Trajectory,
)

# Import scenario_mpc types
from scenario_mpc.types import (
    EgoState,
    EgoInput,
    ObstacleState,
    ModeModel,
    ModeHistory,
    PredictionStep as ScenarioPredictionStep,
    ObstacleTrajectory,
    Scenario,
    TrajectoryMoments,
    CollisionConstraint,
    MPCResult,
    WeightType,
)


# =============================================================================
# State Conversion Functions
# =============================================================================

def state_to_ego_state(state: State) -> EgoState:
    """
    Convert an existing State object to an EgoState.

    Args:
        state: Existing State object with x, y, psi, v

    Returns:
        EgoState with equivalent values
    """
    return EgoState(
        x=float(state.get('x')),
        y=float(state.get('y')),
        theta=float(state.get('psi')),  # psi is theta in scenario_mpc
        v=float(state.get('v'))
    )


def ego_state_to_state(ego_state: EgoState, model_type=None) -> State:
    """
    Convert an EgoState to an existing State object.

    Args:
        ego_state: EgoState from scenario_mpc
        model_type: Optional dynamics model type for the State

    Returns:
        State object with equivalent values
    """
    state = State(model_type)
    state.set('x', ego_state.x)
    state.set('y', ego_state.y)
    state.set('psi', ego_state.theta)
    state.set('v', ego_state.v)
    return state


def ego_input_to_control_dict(ego_input: EgoInput) -> Dict[str, float]:
    """
    Convert EgoInput to a control dictionary.

    Args:
        ego_input: EgoInput from scenario_mpc

    Returns:
        Dictionary with control keys (a, w)
    """
    return {
        'a': ego_input.a,
        'w': ego_input.delta  # delta is angular velocity w
    }


def control_dict_to_ego_input(control_dict: Dict[str, float]) -> EgoInput:
    """
    Convert a control dictionary to EgoInput.

    Args:
        control_dict: Dictionary with 'a' and 'w' keys

    Returns:
        EgoInput with equivalent values
    """
    return EgoInput(
        a=control_dict.get('a', 0.0),
        delta=control_dict.get('w', 0.0)
    )


# =============================================================================
# Obstacle Conversion Functions
# =============================================================================

def dynamic_obstacle_to_obstacle_state(obstacle: DynamicObstacle) -> ObstacleState:
    """
    Convert a DynamicObstacle to an ObstacleState.

    Args:
        obstacle: Existing DynamicObstacle

    Returns:
        ObstacleState with position and velocity
    """
    # Get velocity from prediction if available
    vx, vy = 0.0, 0.0
    if hasattr(obstacle, 'prediction') and obstacle.prediction is not None:
        if hasattr(obstacle.prediction, 'steps') and len(obstacle.prediction.steps) >= 2:
            # Estimate velocity from first two prediction steps
            step0 = obstacle.prediction.steps[0]
            step1 = obstacle.prediction.steps[1]
            dt = 0.1  # Assume standard timestep
            vx = (step1.position[0] - step0.position[0]) / dt
            vy = (step1.position[1] - step0.position[1]) / dt

    return ObstacleState(
        x=float(obstacle.position[0]),
        y=float(obstacle.position[1]),
        vx=float(vx),
        vy=float(vy)
    )


def obstacle_state_to_dynamic_obstacle(
    obs_state: ObstacleState,
    index: int,
    radius: float = 0.5,
    prediction_type: PredictionType = PredictionType.GAUSSIAN
) -> DynamicObstacle:
    """
    Convert an ObstacleState to a DynamicObstacle.

    Args:
        obs_state: ObstacleState from scenario_mpc
        index: Obstacle index
        radius: Obstacle radius
        prediction_type: Type of prediction (GAUSSIAN, DETERMINISTIC)

    Returns:
        DynamicObstacle with position and velocity
    """
    position = np.array([obs_state.x, obs_state.y])
    angle = np.arctan2(obs_state.vy, obs_state.vx) if (obs_state.vx != 0 or obs_state.vy != 0) else 0.0

    obstacle = DynamicObstacle(
        index=index,
        position=position,
        angle=angle,
        radius=radius
    )

    # Create a basic prediction
    obstacle.prediction = Prediction(prediction_type)
    obstacle.velocity = np.array([obs_state.vx, obs_state.vy])

    return obstacle


# =============================================================================
# Prediction Conversion Functions
# =============================================================================

def legacy_prediction_to_trajectory(
    prediction: Prediction,
    obstacle_id: int,
    mode_id: str = "constant_velocity"
) -> ObstacleTrajectory:
    """
    Convert a legacy Prediction to an ObstacleTrajectory.

    Args:
        prediction: Existing Prediction object
        obstacle_id: Obstacle identifier
        mode_id: Mode identifier for this trajectory

    Returns:
        ObstacleTrajectory with prediction steps
    """
    steps = []
    for k, step in enumerate(prediction.steps):
        # Legacy PredictionStep has position, angle, major_radius, minor_radius
        mean = np.array([step.position[0], step.position[1]])
        # Construct covariance from ellipse parameters
        major = step.major_radius
        minor = step.minor_radius
        angle = step.angle

        # Rotation matrix
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        D = np.array([[major**2, 0], [0, minor**2]])
        covariance = R @ D @ R.T

        steps.append(ScenarioPredictionStep(k=k, mean=mean, covariance=covariance))

    return ObstacleTrajectory(
        obstacle_id=obstacle_id,
        mode_id=mode_id,
        steps=steps
    )


def trajectory_to_legacy_prediction(
    trajectory: ObstacleTrajectory,
    prediction_type: PredictionType = PredictionType.GAUSSIAN
) -> Prediction:
    """
    Convert an ObstacleTrajectory to a legacy Prediction.

    Args:
        trajectory: ObstacleTrajectory from scenario_mpc
        prediction_type: Type of prediction

    Returns:
        Legacy Prediction object
    """
    pred = Prediction(prediction_type)

    for step in trajectory.steps:
        # Extract ellipse parameters from covariance
        cov = step.covariance
        # Eigendecomposition for ellipse
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = np.sqrt(max(eigvals[0], 1e-6))
        minor = np.sqrt(max(eigvals[1], 1e-6))
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        legacy_step = LegacyPredictionStep(
            position=step.mean,
            angle=angle,
            major_radius=major,
            minor_radius=minor
        )
        pred.steps.append(legacy_step)

    return pred


# =============================================================================
# Scenario Conversion Functions
# =============================================================================

def create_scenario_from_obstacles(
    obstacles: Dict[int, DynamicObstacle],
    scenario_id: int = 0,
    mode_id: str = "constant_velocity"
) -> Scenario:
    """
    Create a Scenario from a dictionary of DynamicObstacles.

    Args:
        obstacles: Dictionary mapping obstacle_id to DynamicObstacle
        scenario_id: Scenario identifier
        mode_id: Mode identifier for all obstacles

    Returns:
        Scenario with obstacle trajectories
    """
    trajectories = {}
    for obs_id, obstacle in obstacles.items():
        if hasattr(obstacle, 'prediction') and obstacle.prediction is not None:
            traj = legacy_prediction_to_trajectory(
                obstacle.prediction,
                obs_id,
                mode_id
            )
            trajectories[obs_id] = traj

    return Scenario(scenario_id=scenario_id, trajectories=trajectories)


def scenario_to_obstacle_dict(
    scenario: Scenario,
    obstacle_radii: Dict[int, float] = None
) -> Dict[int, DynamicObstacle]:
    """
    Convert a Scenario to a dictionary of DynamicObstacles.

    Args:
        scenario: Scenario from scenario_mpc
        obstacle_radii: Dictionary mapping obstacle_id to radius

    Returns:
        Dictionary mapping obstacle_id to DynamicObstacle
    """
    obstacles = {}
    obstacle_radii = obstacle_radii or {}

    for obs_id, traj in scenario.trajectories.items():
        # Get initial position from first step
        if traj.steps:
            position = traj.steps[0].mean
            radius = obstacle_radii.get(obs_id, 0.5)

            obstacle = DynamicObstacle(
                index=obs_id,
                position=position,
                angle=0.0,
                radius=radius
            )

            # Convert trajectory to prediction
            obstacle.prediction = trajectory_to_legacy_prediction(traj)
            obstacles[obs_id] = obstacle

    return obstacles


# =============================================================================
# Constraint Conversion Functions
# =============================================================================

def collision_constraint_to_dict(constraint: CollisionConstraint) -> Dict:
    """
    Convert a CollisionConstraint to a dictionary format used by the solver.

    Args:
        constraint: CollisionConstraint from scenario_mpc

    Returns:
        Dictionary with a1, a2, b, k, obstacle_id, scenario_id
    """
    return {
        'a1': float(constraint.a[0]),
        'a2': float(constraint.a[1]),
        'b': float(constraint.b),
        'k': constraint.k,
        'obstacle_id': constraint.obstacle_id,
        'scenario_id': constraint.scenario_id,
        'type': 'linear_halfspace'
    }


def dict_to_collision_constraint(d: Dict) -> CollisionConstraint:
    """
    Convert a constraint dictionary to a CollisionConstraint.

    Args:
        d: Dictionary with constraint parameters

    Returns:
        CollisionConstraint
    """
    return CollisionConstraint(
        k=d.get('k', 0),
        obstacle_id=d.get('obstacle_id', 0),
        scenario_id=d.get('scenario_id', 0),
        a=np.array([d.get('a1', 1.0), d.get('a2', 0.0)]),
        b=d.get('b', 0.0)
    )


# =============================================================================
# MPCResult Conversion Functions
# =============================================================================

def mpc_result_to_trajectory(result: MPCResult, model_type=None) -> Trajectory:
    """
    Convert an MPCResult to a Trajectory.

    Args:
        result: MPCResult from scenario_mpc
        model_type: Optional dynamics model type

    Returns:
        Trajectory with states from the MPC result
    """
    traj = Trajectory()
    for ego_state in result.ego_trajectory:
        state = ego_state_to_state(ego_state, model_type)
        traj.add_state(state)
    return traj


def mpc_result_to_planner_output(result: MPCResult, model_type=None):
    """
    Convert an MPCResult to a PlannerOutput-compatible structure.

    Args:
        result: MPCResult from scenario_mpc
        model_type: Optional dynamics model type

    Returns:
        Dictionary with success, control, trajectory
    """
    control = None
    if result.first_input is not None:
        control = ego_input_to_control_dict(result.first_input)

    return {
        'success': result.success,
        'control': control,
        'trajectory': mpc_result_to_trajectory(result, model_type),
        'cost': result.cost,
        'solve_time': result.solve_time,
        'active_scenarios': result.active_scenarios
    }


# =============================================================================
# Data Wrapper for Scenario MPC
# =============================================================================

class ScenarioDataAdapter:
    """
    Adapter class that wraps the existing Data object and provides
    scenario_mpc compatible interfaces.
    """

    def __init__(self, data: Data):
        """
        Initialize adapter with an existing Data object.

        Args:
            data: Existing Data object from the planning system
        """
        self._data = data
        self._mode_histories: Dict[int, ModeHistory] = {}
        self._available_modes: Dict[str, ModeModel] = {}

    @property
    def data(self) -> Data:
        """Access the underlying Data object."""
        return self._data

    def get_ego_state(self) -> Optional[EgoState]:
        """Get current ego state from data."""
        if hasattr(self._data, 'state') and self._data.state is not None:
            return state_to_ego_state(self._data.state)
        return None

    def set_ego_state(self, ego_state: EgoState):
        """Set ego state in data."""
        if hasattr(self._data, 'state') and self._data.state is not None:
            self._data.state.set('x', ego_state.x)
            self._data.state.set('y', ego_state.y)
            self._data.state.set('psi', ego_state.theta)
            self._data.state.set('v', ego_state.v)

    def get_obstacles(self) -> Dict[int, ObstacleState]:
        """Get obstacles as ObstacleState dictionary."""
        obstacles = {}
        if hasattr(self._data, 'dynamic_obstacles') and self._data.dynamic_obstacles:
            for i, obs in enumerate(self._data.dynamic_obstacles):
                obs_id = obs.index if hasattr(obs, 'index') else i
                obstacles[obs_id] = dynamic_obstacle_to_obstacle_state(obs)
        return obstacles

    def get_goal(self) -> Optional[np.ndarray]:
        """Get goal position."""
        if hasattr(self._data, 'goal') and self._data.goal is not None:
            return np.array(self._data.goal[:2])
        if hasattr(self._data, 'parameters') and self._data.parameters:
            goal_x = self._data.parameters.get('goal_x')
            goal_y = self._data.parameters.get('goal_y')
            if goal_x is not None and goal_y is not None:
                return np.array([goal_x, goal_y])
        return None

    def get_horizon(self) -> int:
        """Get planning horizon."""
        if hasattr(self._data, 'horizon') and self._data.horizon is not None:
            return int(self._data.horizon)
        return 20  # Default

    def get_timestep(self) -> float:
        """Get timestep."""
        if hasattr(self._data, 'timestep') and self._data.timestep is not None:
            return float(self._data.timestep)
        return 0.1  # Default

    def set_available_modes(self, modes: Dict[str, ModeModel]):
        """Set available modes for obstacle prediction."""
        self._available_modes = modes

    def get_mode_history(self, obstacle_id: int) -> Optional[ModeHistory]:
        """Get mode history for an obstacle."""
        return self._mode_histories.get(obstacle_id)

    def initialize_mode_history(self, obstacle_id: int):
        """Initialize mode history for an obstacle."""
        if self._available_modes:
            self._mode_histories[obstacle_id] = ModeHistory(
                obstacle_id=obstacle_id,
                available_modes=self._available_modes
            )

    def update_mode_observation(self, obstacle_id: int, mode_id: str, timestep: int):
        """Record a mode observation for an obstacle."""
        if obstacle_id not in self._mode_histories:
            self.initialize_mode_history(obstacle_id)

        history = self._mode_histories.get(obstacle_id)
        if history is not None:
            try:
                history.record_observation(timestep, mode_id)
            except ValueError:
                pass  # Unknown mode, ignore
