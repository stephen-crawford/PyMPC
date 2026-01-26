"""
MPC Controller for RC Car.

This module wraps the PyMPC framework to provide real-time trajectory
optimization for the RC car. It handles state updates from Vicon and
outputs throttle/steering commands.
"""

import numpy as np
import time
import threading
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging

# PyMPC imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from planning.planner import Planner
from planning.types import State, Data, DynamicObstacle, PredictionType, PredictionStep, ReferencePath, generate_reference_path, Problem
from rc_car_server.vicon_interface import VehicleState

logger = logging.getLogger(__name__)


@dataclass
class CarCommand:
    """Command to send to the RC car."""
    throttle: float  # -1.0 (reverse) to 1.0 (forward)
    steering: float  # 0 (left) to 180 (right), 90 is center
    timestamp: float
    valid: bool = True

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'throttle': self.throttle,
            'steering': self.steering
        }


class RCCarMPCController:
    """
    MPC Controller for RC Car using PyMPC framework.

    This controller:
    1. Receives state updates from Vicon
    2. Runs MPC optimization to compute optimal trajectory
    3. Converts MPC outputs (acceleration, angular velocity) to car commands
    4. Handles timing, safety limits, and fallback behaviors
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        reference_path: Optional[ReferencePath] = None
    ):
        """
        Initialize MPC controller.

        Args:
            config_path: Path to YAML configuration file
            reference_path: Reference path to follow (optional, can be set later)
        """
        self.config = self._load_config(config_path)

        # MPC parameters
        self.horizon = self.config.get('horizon', 10)
        self.timestep = self.config.get('timestep', 0.1)

        # Vehicle parameters
        self.wheelbase = self.config.get('wheelbase', 0.26)  # meters
        self.max_velocity = self.config.get('max_velocity', 3.0)  # m/s
        self.max_acceleration = self.config.get('max_acceleration', 2.0)  # m/s^2
        self.max_steering_angle = self.config.get('max_steering_angle', 30.0)  # degrees
        self.max_angular_velocity = self.config.get('max_angular_velocity', 0.8)  # rad/s

        # Command conversion parameters
        self.throttle_gain = self.config.get('throttle_gain', 0.3)  # acceleration to throttle
        self.steering_center = self.config.get('steering_center', 90.0)  # degrees
        self.steering_gain = self.config.get('steering_gain', 45.0)  # rad to degrees

        # Safety parameters
        self.command_timeout = self.config.get('command_timeout', 0.5)  # seconds
        self.min_update_interval = self.config.get('min_update_interval', 0.05)  # seconds

        # State
        self.current_state: Optional[VehicleState] = None
        self.last_command: Optional[CarCommand] = None
        self.last_solve_time = 0.0
        self._lock = threading.Lock()

        # Reference path
        self.reference_path = reference_path

        # Obstacles
        self.obstacles: List[DynamicObstacle] = []

        # Initialize planner
        self.planner: Optional[Planner] = None
        self.solver = None  # Will be set to planner.solver during initialize()
        self._initialized = False

        logger.info(f"RCCarMPCController initialized with horizon={self.horizon}, dt={self.timestep}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            'horizon': 10,
            'timestep': 0.1,
            'wheelbase': 0.26,
            'max_velocity': 3.0,
            'max_acceleration': 2.0,
            'max_steering_angle': 30.0,
            'max_angular_velocity': 0.8,
            'throttle_gain': 0.3,
            'steering_center': 90.0,
            'steering_gain': 45.0,
            'command_timeout': 0.5,
            'min_update_interval': 0.05,
            'robot_radius': 0.15,
            'dynamics_model': 'unicycle',
            'objective': 'contouring',
            'constraints': ['safe_horizon', 'contouring']
        }

        if config_path is None:
            return default_config

        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            # Merge with defaults
            default_config.update(loaded_config)
            return default_config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return default_config

    def initialize(self) -> bool:
        """
        Initialize the MPC planner and solver.

        Must be called after setting reference path.

        Returns:
            True if initialization successful
        """
        try:
            if self.reference_path is None:
                logger.error("Reference path must be set before initialization")
                return False

            # Load base config from CONFIG.yml
            from utils.utils import read_config_file
            base_config = read_config_file()

            # Override with RC car specific settings
            base_config['planner'] = base_config.get('planner', {})
            base_config['planner']['horizon'] = self.horizon
            base_config['planner']['timestep'] = self.timestep

            base_config['robot'] = base_config.get('robot', {})
            base_config['robot']['radius'] = self.config.get('robot_radius', 0.15)

            base_config['dynamics_model'] = self.config.get('dynamics_model', 'unicycle')
            base_config['objective'] = self.config.get('objective', 'contouring')
            base_config['constraints'] = self.config.get('constraints', ['contouring'])

            base_config['contouring_objective'] = base_config.get('contouring_objective', {})
            base_config['contouring_objective']['lag_weight'] = 10.0
            base_config['contouring_objective']['contour_weight'] = 5.0
            base_config['contouring_objective']['velocity_weight'] = 1.0

            base_config['safe_horizon_constraints'] = base_config.get('safe_horizon_constraints', {})
            base_config['safe_horizon_constraints']['enable_adaptive_mode_sampling'] = False
            base_config['safe_horizon_constraints']['num_scenarios'] = 50
            base_config['safe_horizon_constraints']['epsilon_p'] = 0.15
            base_config['safe_horizon_constraints']['beta'] = 0.1
            base_config['safe_horizon_constraints']['max_constraints_per_disc'] = 5

            # Create dynamics model
            dynamics_model_type = self.config.get('dynamics_model', 'unicycle')
            if dynamics_model_type == 'unicycle':
                from planning.dynamic_models import ContouringSecondOrderUnicycleModel
                dynamics_model = ContouringSecondOrderUnicycleModel()
            elif dynamics_model_type == 'bicycle':
                from planning.dynamic_models import CurvatureAwareSecondOrderBicycleModel
                dynamics_model = CurvatureAwareSecondOrderBicycleModel()
            else:
                from planning.dynamic_models import ContouringSecondOrderUnicycleModel
                dynamics_model = ContouringSecondOrderUnicycleModel()

            self.dynamics_model = dynamics_model

            # Create objective module
            from modules.objectives.contouring_objective import ContouringObjective
            from modules.objectives.control_effort_objective import ControlEffortObjective
            from modules.objectives.control_jerk_objective import ControlJerkObjective
            from modules.objectives.path_reference_velocity_objective import PathReferenceVelocityObjective

            objective_module = ContouringObjective()
            objective_module.config = base_config
            objective_module.settings = base_config

            control_effort = ControlEffortObjective()
            control_effort.config = base_config
            control_effort.settings = base_config

            control_jerk = ControlJerkObjective()
            control_jerk.config = base_config
            control_jerk.settings = base_config

            path_velocity = PathReferenceVelocityObjective()
            path_velocity.config = base_config
            path_velocity.settings = base_config

            # Create constraint modules
            constraint_modules = []
            constraints_config = self.config.get('constraints', ['contouring'])
            for constraint_type in constraints_config:
                if constraint_type == 'contouring':
                    from modules.constraints.contouring_constraints import ContouringConstraints
                    module = ContouringConstraints()
                elif constraint_type == 'safe_horizon':
                    from modules.constraints.safe_horizon_constraint import SafeHorizonConstraint
                    module = SafeHorizonConstraint(settings=base_config)
                else:
                    continue
                module.config = base_config
                module.settings = base_config
                constraint_modules.append(module)

            # Build Problem for Planner
            problem = Problem()
            problem.model_type = dynamics_model
            problem.modules = constraint_modules + [control_effort, control_jerk, path_velocity, objective_module]
            # Add horizon and timestep methods to problem
            problem.get_horizon = lambda: self.horizon
            problem.get_timestep = lambda: self.timestep

            # Create data with reference path
            data = Data()
            data.reference_path = self.reference_path
            data.dynamic_obstacles = self.obstacles if self.obstacles else []
            data.horizon = self.horizon
            data.timestep = self.timestep
            data.dynamics_model = dynamics_model
            problem.data = data

            # Create planner
            self.planner = Planner(problem, config=base_config)
            self.solver = self.planner.solver
            self.solver.horizon = self.horizon
            self.solver.timestep = self.timestep

            # Set solver on all modules
            for module in self.planner.solver.module_manager.get_modules():
                if hasattr(module, 'solver'):
                    module.solver = self.planner.solver

            # Define parameters
            self.planner.solver.define_parameters()

            self._initialized = True
            logger.info("MPC controller initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MPC controller: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_reference_path(self, path: ReferencePath):
        """Set the reference path to follow."""
        self.reference_path = path
        if self.planner is not None:
            self.planner.set_reference_path(path)

    def set_reference_path_from_waypoints(self, waypoints: List[Tuple[float, float]]):
        """
        Create and set reference path from waypoints.

        Args:
            waypoints: List of (x, y) waypoints in meters
        """
        # Use generate_reference_path to create path from waypoints
        if len(waypoints) >= 2:
            start = waypoints[0]
            goal = waypoints[-1]
            path = generate_reference_path(start, goal, path_type="curved", num_points=len(waypoints))
            # Override with actual waypoints
            path.x = np.array([w[0] for w in waypoints])
            path.y = np.array([w[1] for w in waypoints])
            self.set_reference_path(path)
        else:
            logger.warning("Need at least 2 waypoints to create reference path")

    def update_state(self, state: VehicleState):
        """
        Update the current vehicle state from Vicon.

        Args:
            state: Current vehicle state
        """
        with self._lock:
            self.current_state = state

    def update_obstacles(self, obstacles: List[Dict[str, Any]]):
        """
        Update obstacle information.

        Args:
            obstacles: List of obstacle dictionaries with keys:
                - x, y: position
                - vx, vy: velocity
                - radius: obstacle radius
        """
        self.obstacles = []
        for i, obs in enumerate(obstacles):
            obstacle = DynamicObstacle(
                id=i,
                position=np.array([obs['x'], obs['y']]),
                radius=obs.get('radius', 0.2)
            )
            obstacle.velocity = np.array([obs.get('vx', 0.0), obs.get('vy', 0.0)])

            # Create simple prediction (constant velocity)
            obstacle.prediction.type = PredictionType.GAUSSIAN
            obstacle.prediction.steps = []
            for k in range(self.horizon + 1):
                t = k * self.timestep
                pred_x = obs['x'] + obs.get('vx', 0.0) * t
                pred_y = obs['y'] + obs.get('vy', 0.0) * t
                step = PredictionStep(
                    position=np.array([pred_x, pred_y]),
                    angle=0.0,
                    major_radius=0.3 + 0.1 * k,  # Growing uncertainty
                    minor_radius=0.3 + 0.1 * k
                )
                obstacle.prediction.steps.append(step)

            self.obstacles.append(obstacle)

    def compute_command(self) -> CarCommand:
        """
        Compute the next control command using MPC.

        Returns:
            CarCommand with throttle and steering values
        """
        current_time = time.time()

        # Check if we should compute new command
        if current_time - self.last_solve_time < self.min_update_interval:
            if self.last_command is not None:
                return self.last_command
            return self._get_safe_command()

        with self._lock:
            state = self.current_state

        # Check state validity
        if state is None or not state.valid:
            logger.warning("Invalid state, returning safe command")
            return self._get_safe_command()

        # Check if initialized
        if not self._initialized or self.planner is None:
            logger.warning("Controller not initialized, returning safe command")
            return self._get_safe_command()

        try:
            # Convert Vicon state to MPC state
            mpc_state = State(model_type=self.planner.dynamics_model)
            mpc_state.set('x', state.x)
            mpc_state.set('y', state.y)
            mpc_state.set('psi', state.yaw)
            mpc_state.set('v', np.sqrt(state.vx**2 + state.vy**2))

            # Create data object
            data = Data()
            data.dynamic_obstacles = self.obstacles
            data.reference_path = self.reference_path

            # Solve MPC
            solve_start = time.time()
            result = self.planner.solve_mpc(mpc_state, data)
            solve_time = time.time() - solve_start

            if result.success and result.trajectory is not None:
                # Extract first control input
                controls = result.trajectory.get_controls()
                if controls and len(controls) > 0:
                    u0 = controls[0]
                    acceleration = u0.get('a', 0.0)
                    angular_velocity = u0.get('w', 0.0)

                    # Convert to car commands
                    command = self._convert_to_car_command(
                        acceleration, angular_velocity, state
                    )

                    self.last_command = command
                    self.last_solve_time = current_time

                    logger.debug(f"MPC solved in {solve_time*1000:.1f}ms: "
                               f"a={acceleration:.2f}, w={angular_velocity:.2f} -> "
                               f"throttle={command.throttle:.2f}, steering={command.steering:.1f}")

                    return command

            # MPC failed, use fallback
            logger.warning(f"MPC solve failed, using fallback")
            return self._get_fallback_command(state)

        except Exception as e:
            logger.error(f"MPC computation error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_safe_command()

    def _convert_to_car_command(
        self,
        acceleration: float,
        angular_velocity: float,
        state: VehicleState
    ) -> CarCommand:
        """
        Convert MPC outputs to car commands.

        Args:
            acceleration: Desired acceleration [m/s^2]
            angular_velocity: Desired angular velocity [rad/s]
            state: Current vehicle state

        Returns:
            CarCommand with throttle and steering
        """
        # Convert acceleration to throttle
        # Throttle range: -1.0 (reverse) to 1.0 (forward)
        throttle = np.clip(acceleration * self.throttle_gain, -1.0, 1.0)

        # Add feedforward based on current velocity to maintain speed
        current_v = np.sqrt(state.vx**2 + state.vy**2)
        if current_v > 0.1:
            # Small feedforward to overcome friction
            throttle += 0.1 * np.sign(throttle) if abs(throttle) > 0.01 else 0.0
        throttle = np.clip(throttle, -1.0, 1.0)

        # Convert angular velocity to steering angle
        # For a unicycle/bicycle model: omega = v * tan(delta) / L
        # So: delta = atan(omega * L / v)
        if current_v > 0.1:
            steering_rad = np.arctan(angular_velocity * self.wheelbase / current_v)
        else:
            # At low speed, use angular velocity directly
            steering_rad = angular_velocity * 0.5

        # Clamp steering
        max_steer_rad = np.radians(self.max_steering_angle)
        steering_rad = np.clip(steering_rad, -max_steer_rad, max_steer_rad)

        # Convert to servo angle (0-180, 90 is center)
        steering_deg = self.steering_center - np.degrees(steering_rad) * self.steering_gain / self.max_steering_angle
        steering_deg = np.clip(steering_deg, 0.0, 180.0)

        return CarCommand(
            throttle=throttle,
            steering=steering_deg,
            timestamp=time.time(),
            valid=True
        )

    def _get_safe_command(self) -> CarCommand:
        """Get a safe (stopped) command."""
        return CarCommand(
            throttle=0.0,
            steering=self.steering_center,
            timestamp=time.time(),
            valid=True
        )

    def _get_fallback_command(self, state: VehicleState) -> CarCommand:
        """
        Get a fallback command when MPC fails.

        Uses simple proportional control to slow down safely.
        """
        current_v = np.sqrt(state.vx**2 + state.vy**2)

        # Decelerate if moving
        if current_v > 0.1:
            throttle = -0.3  # Gentle braking
        else:
            throttle = 0.0

        return CarCommand(
            throttle=throttle,
            steering=self.steering_center,
            timestamp=time.time(),
            valid=True
        )

    def stop(self):
        """Emergency stop - send zero throttle command."""
        self.last_command = self._get_safe_command()
        return self.last_command

    def is_ready(self) -> bool:
        """Check if controller is ready to compute commands."""
        return (
            self._initialized and
            self.planner is not None and
            self.reference_path is not None
        )

    def get_predicted_trajectory(self) -> Optional[List[Tuple[float, float]]]:
        """
        Get the last computed predicted trajectory.

        Returns:
            List of (x, y) positions or None
        """
        if self.planner is None or not hasattr(self.planner, 'last_trajectory'):
            return None

        try:
            traj = self.planner.last_trajectory
            if traj is None:
                return None

            states = traj.get_states()
            return [(s.get('x'), s.get('y')) for s in states]
        except:
            return None
