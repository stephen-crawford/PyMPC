"""
Standardized Integration Test Framework for PyMPC

This framework provides a standardized way to run integration tests with:
- Reference path
- Objective module
- Constraint modules
- Vehicle dynamics model
- Obstacle dynamics models
- Configuration from CONFIG.yml

Each test outputs a timestamped folder with:
- Test script copy
- CSV state history
- Log file
- GIF animation
"""
import os
import sys
import time
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
# yaml not used
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from solver.src.casadi_solver import CasADiSolver
from planning.src.types import Data, DynamicObstacle
from planning.src.planner import Planner
from planning.src.types import generate_reference_path, calculate_path_normals, Bound
from planning.src.data_prep import define_robot_area, ensure_obstacle_size, propagate_obstacles
from utils.utils import read_config_file
from .obstacle_manager import ObstacleManager, ObstacleConfig, create_unicycle_obstacle, create_bicycle_obstacle, create_point_mass_obstacle


@dataclass
class TestConfig:
    """Configuration for integration test."""
    reference_path: np.ndarray
    objective_module: str
    constraint_modules: List[str]
    vehicle_dynamics: str
    num_obstacles: int
    obstacle_dynamics: List[str]
    test_name: str
    duration: float = 10.0
    timestep: float = 0.1
    # Optional explicit obstacle configurations to avoid randomness in tests
    obstacle_configs: Optional[List[ObstacleConfig]] = None


@dataclass
class TestResult:
    """Result of integration test."""
    success: bool
    vehicle_states: List[np.ndarray]
    obstacle_states: List[List[np.ndarray]]
    computation_times: List[float]
    constraint_violations: List[bool]
    output_folder: str


class IntegrationTestFramework:
    """Standardized framework for PyMPC integration tests."""
    
    def __init__(self, config_file: str = "config/CONFIG.yml"):
        """Initialize the test framework."""
        self.config_file = config_file
        # Project-wide config loader ignores passed path and reads standard CONFIG
        self.config = read_config_file()
        self.output_base_dir = "test_outputs"
        
        # Create output directory
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # Initialize solver
        self.solver = CasADiSolver()
        self.solver.horizon = self.config.get("horizon", 10)
        self.solver.timestep = self.config.get("timestep", 0.1)
        
    def create_test_folder(self, test_config: TestConfig) -> str:
        """Create timestamped test folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{test_config.objective_module}_{'_'.join(test_config.constraint_modules)}_{test_config.vehicle_dynamics}"
        folder_path = os.path.join(self.output_base_dir, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
        
    def setup_logging(self, output_folder: str) -> logging.Logger:
        """Setup logging for the test."""
        log_file = os.path.join(output_folder, "test.log")
        
        logger = logging.getLogger("integration_test")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def create_vehicle_dynamics(self, dynamics_type: str):
        """Create vehicle dynamics model."""
        if dynamics_type == "bicycle":
            from planning.src.dynamic_models import SecondOrderBicycleModel
            return SecondOrderBicycleModel()
        elif dynamics_type == "unicycle":
            from planning.src.dynamic_models import SecondOrderUnicycleModel
            return SecondOrderUnicycleModel()
        elif dynamics_type == "point_mass":
            from .obstacle_manager import PointMassModel
            return PointMassModel()
        else:
            raise ValueError(f"Unknown vehicle dynamics type: {dynamics_type}")
            
    def create_objective_module(self, objective_type: str):
        """Create objective module."""
        if objective_type == "contouring":
            from planner_modules.src.objectives.contouring_objective import ContouringObjective
            return ContouringObjective(self.solver)
        elif objective_type == "goal":
            from planner_modules.src.objectives.goal_objective import GoalObjective
            return GoalObjective(self.solver)
        elif objective_type == "path_reference_velocity":
            from planner_modules.src.objectives.path_reference_velocity_objective import PathReferenceVelocityObjective
            return PathReferenceVelocityObjective(self.solver)
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
            
    def create_constraint_modules(self, constraint_types: List[str]):
        """Create constraint modules."""
        constraints = []
        
        for constraint_type in constraint_types:
            if constraint_type == "safe_horizon":
                from planner_modules.src.constraints.safe_horizon_constraint import SafeHorizonConstraint
                constraints.append(SafeHorizonConstraint(self.solver))
            elif constraint_type == "contouring":
                from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
                constraints.append(ContouringConstraints(self.solver))
            elif constraint_type == "gaussian":
                from planner_modules.src.constraints.gaussian_constraints import GaussianConstraints
                constraints.append(GaussianConstraints(self.solver))
            elif constraint_type == "linear":
                from planner_modules.src.constraints.linearized_constraints import LinearizedConstraints
                constraints.append(LinearizedConstraints(self.solver))
            elif constraint_type == "ellipsoid":
                from planner_modules.src.constraints.ellipsoid_constraints import EllipsoidConstraints
                constraints.append(EllipsoidConstraints(self.solver))
            elif constraint_type == "decomp":
                from planner_modules.src.constraints.decomp_constraints import DecompConstraints
                constraints.append(DecompConstraints(self.solver))
            elif constraint_type == "guidance":
                from planner_modules.src.constraints.guidance_constraints import GuidanceConstraints
                constraints.append(GuidanceConstraints(self.solver))
            elif constraint_type == "scenario":
                from planner_modules.src.constraints.scenario_constraints import ScenarioConstraints
                constraints.append(ScenarioConstraints(self.solver))
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")
                
        return constraints
        
    def create_obstacles(self, num_obstacles: int, dynamics_types: List[str],
                         obstacle_configs: Optional[List[ObstacleConfig]] = None) -> List[DynamicObstacle]:
        """Create obstacles with specified dynamics using obstacle manager.
        If obstacle_configs is provided, it will be used directly for deterministic setups.
        """
        obstacle_manager = ObstacleManager(self.config)
        
        # If not provided, create random obstacle configurations
        if obstacle_configs is None:
            obstacle_configs = []
            for i in range(num_obstacles):
                dynamics_type = dynamics_types[i % len(dynamics_types)]
                
                # Random initial position
                x = np.random.uniform(0.0, 20.0)
                y = np.random.uniform(-5.0, 5.0)
                
                # Random initial velocity
                speed = np.random.uniform(0.5, 2.0)
                angle = np.random.uniform(0, 2 * np.pi)
                velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
                
                if dynamics_type == "unicycle":
                    config = create_unicycle_obstacle(i, np.array([x, y]), velocity)
                elif dynamics_type == "bicycle":
                    config = create_bicycle_obstacle(i, np.array([x, y]), velocity)
                elif dynamics_type == "point_mass":
                    config = create_point_mass_obstacle(i, np.array([x, y]), velocity)
                else:
                    raise ValueError(f"Unknown dynamics type: {dynamics_type}")
                    
                obstacle_configs.append(config)
            
        # Create obstacles
        obstacles = obstacle_manager.create_obstacles_from_config(obstacle_configs)
        
        # Store obstacle manager for state updates
        self.obstacle_manager = obstacle_manager
        
        return obstacles
        
    def run_test(self, test_config: TestConfig) -> TestResult:
        """Run the integration test."""
        logger = logging.getLogger("integration_test")
        logger.info(f"Starting integration test: {test_config.test_name}")
        
        # Create output folder
        output_folder = self.create_test_folder(test_config)
        logger.info(f"Output folder: {output_folder}")
        
        # Setup logging
        logger = self.setup_logging(output_folder)
        
        # Copy test script
        self.copy_test_script(output_folder)
        
        try:
            # Create modules
            vehicle_dynamics = self.create_vehicle_dynamics(test_config.vehicle_dynamics)
            objective_module = self.create_objective_module(test_config.objective_module)
            constraint_modules = self.create_constraint_modules(test_config.constraint_modules)
            
            # Configure solver with dynamics and modules
            # IMPORTANT: Add constraints BEFORE objectives to satisfy dependencies
            self.solver.set_dynamics_model(vehicle_dynamics)
            # Add constraints first (some objectives depend on constraints)
            for c in constraint_modules:
                self.solver.module_manager.add_module(c)
            # Then add objectives (which may have dependencies on constraints)
            self.solver.module_manager.add_module(objective_module)

            # Create planner using real solver
            planner = Planner(self.solver, model_type=vehicle_dynamics)

            # Define parameters now that modules are registered
            self.solver.define_parameters()

            # Create obstacles
            obstacles = self.create_obstacles(
                test_config.num_obstacles,
                test_config.obstacle_dynamics,
                test_config.obstacle_configs
            )
            
            # Initialize data
            data = Data()
            data.dynamic_obstacles = obstacles

            # Build reference path and boundaries
            try:
                # Handle reference_path: can be numpy array or already a ReferencePath object
                if test_config.reference_path is not None:
                    if isinstance(test_config.reference_path, np.ndarray):
                        # Convert numpy array to ReferencePath object
                        from planning.src.types import ReferencePath
                        from scipy.interpolate import CubicSpline
                        ref_path = ReferencePath()
                        # Keep as numpy arrays for arithmetic operations
                        x_arr = test_config.reference_path[:, 0]
                        y_arr = test_config.reference_path[:, 1]
                        z_arr = np.zeros(len(x_arr))
                        
                        # Compute arc length
                        s = np.zeros(len(x_arr))
                        for i in range(1, len(x_arr)):
                            dx = x_arr[i] - x_arr[i - 1]
                            dy = y_arr[i] - y_arr[i - 1]
                            s[i] = s[i - 1] + np.sqrt(dx**2 + dy**2)
                        
                        # Store as lists (ReferencePath expects lists)
                        ref_path.x = x_arr.tolist()
                        ref_path.y = y_arr.tolist()
                        ref_path.z = z_arr.tolist()
                        ref_path.s = s.tolist()
                        
                        # Build splines
                        ref_path.x_spline = CubicSpline(s, ref_path.x)
                        ref_path.y_spline = CubicSpline(s, ref_path.y)
                        ref_path.z_spline = CubicSpline(s, ref_path.z)
                        ref_path.length = float(s[-1])
                        
                        data.reference_path = ref_path
                        start_pt = [ref_path.x[0], ref_path.y[0], 0.0]
                        goal_pt = [ref_path.x[-1], ref_path.y[-1], 0.0]
                    else:
                        # Already a ReferencePath object
                        data.reference_path = test_config.reference_path
                        start_pt = [data.reference_path.x[0], data.reference_path.y[0], 0.0]
                        goal_pt = [data.reference_path.x[-1], data.reference_path.y[-1], 0.0]
                else:
                    start_pt = [0.0, 0.0, 0.0]
                    goal_pt = [20.0, 0.0, 0.0]
                    ref_path = generate_reference_path(start_pt, goal_pt, path_type="straight")
                    data.reference_path = ref_path

                # Set start/goal and goal flag for GoalObjective
                data.start = np.array(start_pt[:2])
                data.goal = np.array(goal_pt[:2])
                data.goal_received = True

                # Compute road boundaries using path normals
                normals = calculate_path_normals(ref_path)
                road_width = float(self.config.get("road", {}).get("width", 7.0))
                half_width = road_width / 2.0
                left_x, left_y, right_x, right_y = [], [], [], []
                for i in range(len(ref_path.x)):
                    nx, ny = normals[i]
                    left_x.append(ref_path.x[i] + nx * half_width)
                    left_y.append(ref_path.y[i] + ny * half_width)
                    right_x.append(ref_path.x[i] - nx * half_width)
                    right_y.append(ref_path.y[i] - ny * half_width)
                data.left_boundary_x = left_x
                data.left_boundary_y = left_y
                data.right_boundary_x = right_x
                data.right_boundary_y = right_y
                data.left_bound = Bound(left_x, left_y, ref_path.s)
                data.right_bound = Bound(right_x, right_y, ref_path.s)
            except Exception:
                # Minimal fallback
                data.goal = np.array([20.0, 0.0])
                data.goal_received = True

            # Robot area discs - ensure we have enough discs for all constraint modules
            num_discs = int(self.config.get("num_discs", 1))
            robot_radius = float(self.config.get("robot", {}).get("radius", 0.5))
            vehicle_length = getattr(vehicle_dynamics, "length", 2.0 * robot_radius)
            vehicle_width = getattr(vehicle_dynamics, "width", 2.0 * robot_radius)
            
            # Ensure we have at least num_discs discs
            try:
                data.robot_area = define_robot_area(vehicle_length, vehicle_width, max(1, num_discs))
                # Validate we have enough discs
                if len(data.robot_area) < num_discs:
                    logger.warning(f"Only {len(data.robot_area)} discs created, but {num_discs} required. Adding more.")
                    from planning.src.types import Disc
                    while len(data.robot_area) < num_discs:
                        # Add discs evenly spaced
                        offset = -vehicle_length / 2 + (len(data.robot_area) * vehicle_length / (num_discs + 1))
                        data.robot_area.append(Disc(offset, robot_radius))
            except Exception as e:
                logger.warning(f"Error creating robot_area: {e}")
                # Fallback single disc
                from planning.src.types import Disc
                data.robot_area = [Disc(0.0, robot_radius)]
                # Add more discs if needed
                while len(data.robot_area) < num_discs:
                    offset = -vehicle_length / 2 + (len(data.robot_area) * vehicle_length / (num_discs + 1))
                    data.robot_area.append(Disc(offset, robot_radius))

            data.planning_start_time = time.time()

            # Shape obstacles to meet module expectations
            try:
                # First propagate obstacles to ensure prediction steps exist
                # Compute obstacle speeds for propagate_obstacles
                obstacle_speeds = []
                for obs in data.dynamic_obstacles:
                    if hasattr(obs, 'prediction') and obs.prediction and len(obs.prediction.steps) > 0:
                        # Use existing prediction steps to estimate speed
                        if len(obs.prediction.steps) > 1:
                            p0 = obs.prediction.steps[0].position
                            p1 = obs.prediction.steps[1].position
                            speed = np.linalg.norm(p1 - p0) / self.solver.timestep
                        else:
                            speed = 1.0  # Default speed
                    else:
                        # Estimate speed from obstacle angle and velocity
                        if hasattr(obs, 'velocity') and obs.velocity is not None:
                            speed = np.linalg.norm(obs.velocity)
                        else:
                            speed = 1.0  # Default speed
                    obstacle_speeds.append(speed)
                
                # Use average speed for propagate_obstacles if obstacles don't have paths
                avg_speed = np.mean(obstacle_speeds) if obstacle_speeds else 1.0
                
                # propagate_obstacles needs obstacles to have predictions with paths, or it will create constant velocity predictions
                propagate_obstacles(data, dt=self.solver.timestep, horizon=self.solver.horizon, speed=avg_speed)
                # Then ensure obstacle list is sized to max_obstacles (needs prediction.steps to exist)
                ensure_obstacle_size(data.dynamic_obstacles, planner.state)
                # Propagate again after sizing to ensure all obstacles have proper predictions
                propagate_obstacles(data, dt=self.solver.timestep, horizon=self.solver.horizon, speed=avg_speed)
            except Exception as e:
                logger.warning(f"Error shaping obstacles: {e}")
                # Ensure at least robot_area is set correctly
                if not hasattr(data, 'robot_area') or len(data.robot_area) == 0:
                    from planning.src.types import Disc
                    data.robot_area = [Disc(0.0, robot_radius)]

            # Preload parameter manager storage to correct size
            try:
                pm = self.solver.parameter_manager
                if pm.parameter_values is None:
                    import numpy as _np
                    pm.parameter_values = _np.zeros(pm.parameter_count, dtype=float)
            except Exception:
                pass

            # Notify modules about data (critical for contouring modules to receive reference path)
            try:
                planner.on_data_received(data)
            except Exception as e:
                logger.warning(f"Error in on_data_received: {e}")
            
            # Initialize planner once before loop (mirrors original scripts)
            try:
                planner.initialize(data)
            except Exception:
                pass
            
            # Initialize state tracking
            vehicle_states = []
            obstacle_states = [[] for _ in range(test_config.num_obstacles)]
            computation_times = []
            constraint_violations = []
            
            # Initial state - include spline variable if model has it
            vehicle_state = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, theta, v
            # Initialize spline to 0 if model requires it
            if 'spline' in vehicle_dynamics.get_all_vars():
                # Add spline and other variables if needed
                if len(vehicle_state) < len(vehicle_dynamics.dependent_vars):
                    # Append missing state variables with default values
                    for var in vehicle_dynamics.dependent_vars[len(vehicle_state):]:
                        if var == 'spline':
                            vehicle_state = np.append(vehicle_state, 0.0)  # Start at beginning of path
                        elif var == 'delta':
                            vehicle_state = np.append(vehicle_state, 0.0)  # No steering initially
                        elif var == 'slack':
                            vehicle_state = np.append(vehicle_state, 0.0)  # No slack initially
                        else:
                            vehicle_state = np.append(vehicle_state, 0.0)
            vehicle_states.append(vehicle_state.copy())
            
            # Run simulation
            num_steps = int(test_config.duration / test_config.timestep)
            goal_reached = False
            goal_reached_step = None
            
            for step in range(num_steps):
                logger.info(f"Step {step}/{num_steps}")
                
                start_time = time.time()
                
                # Check if goal is reached (check before solving to avoid unnecessary computation)
                if not goal_reached and hasattr(planner, 'is_objective_reached'):
                    try:
                        goal_reached = planner.is_objective_reached(data)
                        if goal_reached:
                            goal_reached_step = step
                            logger.info(f"Goal reached at step {step}!")
                    except Exception as e:
                        logger.debug(f"Could not check goal status: {e}")
                
                # Prepare iteration for modules that support it
                for module in [objective_module] + constraint_modules:
                    if hasattr(module, 'prepare_iteration'):
                        module.prepare_iteration(vehicle_state, data)
                
                # Sync planner state - include all state variables
                planner.state.set('x', float(vehicle_state[0]))
                planner.state.set('y', float(vehicle_state[1]))
                planner.state.set('psi', float(vehicle_state[2]))
                planner.state.set('v', float(vehicle_state[3]))
                # Set spline and other state variables if present in model
                if 'spline' in vehicle_dynamics.get_all_vars() and len(vehicle_state) > 4:
                    spline_idx = vehicle_dynamics.dependent_vars.index('spline') if 'spline' in vehicle_dynamics.dependent_vars else None
                    if spline_idx is not None and len(vehicle_state) > spline_idx:
                        planner.state.set('spline', float(vehicle_state[spline_idx]))
                if 'delta' in vehicle_dynamics.get_all_vars() and len(vehicle_state) > 5:
                    delta_idx = vehicle_dynamics.dependent_vars.index('delta') if 'delta' in vehicle_dynamics.dependent_vars else None
                    if delta_idx is not None and len(vehicle_state) > delta_idx:
                        planner.state.set('delta', float(vehicle_state[delta_idx]))
                
                # Update modules with current state (critical for contouring to work)
                try:
                    for module in [objective_module] + constraint_modules:
                        if hasattr(module, 'update'):
                            module.update(planner.state, data)
                except Exception as e:
                    logger.debug(f"Error updating modules: {e}")

                # Ensure robot_area is still valid
                if not hasattr(data, 'robot_area') or len(data.robot_area) == 0:
                    logger.warning(f"robot_area is empty at step {step}, recreating...")
                    from planning.src.types import Disc
                    data.robot_area = [Disc(0.0, robot_radius)]
                    while len(data.robot_area) < num_discs:
                        offset = -vehicle_length / 2 + (len(data.robot_area) * vehicle_length / (num_discs + 1))
                        data.robot_area.append(Disc(offset, robot_radius))

                # Ensure obstacles have valid predictions before solve
                if data.dynamic_obstacles:
                    for obs in data.dynamic_obstacles:
                        if not hasattr(obs, 'prediction') or obs.prediction is None:
                            logger.warning(f"Obstacle {obs.index} has no prediction, creating default...")
                            from planning.src.types import Prediction, PredictionType
                            obs.prediction = Prediction(PredictionType.GAUSSIAN)
                            obs.prediction.steps = []
                        elif not hasattr(obs.prediction, 'steps') or len(obs.prediction.steps) == 0:
                            logger.warning(f"Obstacle {obs.index} has no prediction steps, propagating...")
                            # Quick propagation to create steps
                            try:
                                speed_est = 1.0
                                if hasattr(obs, 'angle'):
                                    velocity = np.array([np.cos(obs.angle), np.sin(obs.angle)]) * speed_est
                                else:
                                    velocity = np.array([1.0, 0.0])
                                from planning.src.data_prep import get_constant_velocity_prediction
                                obs.prediction = get_constant_velocity_prediction(
                                    obs.position, velocity, self.solver.timestep, self.solver.horizon + 1
                                )
                            except Exception as e:
                                logger.warning(f"Failed to create prediction steps: {e}")

                # Initialize and solve via planner
                self.solver.initialize(data)
                # Ensure parameter_values is properly initialized after define_parameters
                pm = self.solver.parameter_manager
                if pm.parameter_values is None or len(pm.parameter_values) != pm.parameter_count:
                    import numpy as _np
                    pm.parameter_values = _np.zeros(pm.parameter_count, dtype=float)
                self.solver.initialize_rollout(planner.state)

                try:
                    planner_output = planner.solve_mpc(data)
                    # Prefer advancing with model integration like example scripts
                    try:
                        from planning.src.dynamic_models import numeric_rk4
                        # Attempt to get next controls from latest trajectory
                        traj = self.solver.get_reference_trajectory()
                        if traj is not None and len(traj.get_states()) >= 2:
                            next_state = traj.get_states()[1]
                            # Assemble z vector [u, x]
                            a_next = next_state.get('a') if next_state.has('a') else 0.0
                            w_next = next_state.get('w') if next_state.has('w') else 0.0
                            # Derive delta/slack if present
                            u_vec = [a_next, w_next]
                            if 'slack' in getattr(vehicle_dynamics, 'inputs', []):
                                u_vec.append(0.0)
                            x_vec = [planner.state.get('x'), planner.state.get('y')]
                            if 'psi' in getattr(vehicle_dynamics, 'dependent_vars', []):
                                x_vec.append(planner.state.get('psi'))
                            if 'v' in getattr(vehicle_dynamics, 'dependent_vars', []):
                                x_vec.append(planner.state.get('v'))
                            if 'delta' in getattr(vehicle_dynamics, 'dependent_vars', []):
                                x_vec.append(planner.state.get('delta'))
                            if 'spline' in getattr(vehicle_dynamics, 'dependent_vars', []):
                                x_vec.append(planner.state.get('spline'))
                            import casadi as _cs
                            z_k = _cs.vertcat(*u_vec, *_cs.vertcat(*x_vec)) if isinstance(u_vec, list) else _cs.vertcat(u_vec, x_vec)
                            vehicle_dynamics.load(z_k)
                            next_state_symbolic = vehicle_dynamics.discrete_dynamics(z_k, self.solver.parameter_manager, self.solver.timestep)
                            next_state_num = numeric_rk4(next_state_symbolic, vehicle_dynamics, self.solver.parameter_manager, self.solver.timestep)
                            # Map back to vehicle_state (x,y,psi,v where available)
                            x_next = float(next_state_num[0])
                            y_next = float(next_state_num[1])
                            psi_next = float(next_state_num[2]) if vehicle_dynamics.state_dimension >= 3 else vehicle_state[2]
                            v_next = float(next_state_num[3]) if vehicle_dynamics.state_dimension >= 4 else vehicle_state[3]
                            # Build new state array preserving all variables
                            new_state = np.array([x_next, y_next, psi_next, v_next])
                            # Preserve additional state variables (delta, spline, etc.)
                            if len(vehicle_state) > 4:
                                for i in range(4, len(vehicle_state)):
                                    if i < len(next_state_num):
                                        new_state = np.append(new_state, float(next_state_num[i]))
                                    else:
                                        new_state = np.append(new_state, vehicle_state[i])
                            vehicle_state = new_state
                        else:
                            # minimal fallback integration
                            dt = test_config.timestep
                            vehicle_state[0] += vehicle_state[3] * np.cos(vehicle_state[2]) * dt
                            vehicle_state[1] += vehicle_state[3] * np.sin(vehicle_state[2]) * dt
                    except Exception:
                        # fallback to reference trajectory mapping as before
                        traj = self.solver.get_reference_trajectory()
                        if traj is not None and len(traj.get_states()) >= 2:
                            next_state = traj.get_states()[1]
                            x_next = next_state.get('x')
                            y_next = next_state.get('y')
                            psi_next = next_state.get('psi') if next_state.has('psi') else vehicle_state[2]
                            v_next = next_state.get('v') if next_state.has('v') else vehicle_state[3]
                            # Build new state array preserving all variables
                            new_state = np.array([float(x_next), float(y_next), float(psi_next), float(v_next)])
                            # Preserve additional state variables (delta, spline, etc.)
                            if len(vehicle_state) > 4:
                                for i in range(4, len(vehicle_state)):
                                    var_name = vehicle_dynamics.dependent_vars[i] if i < len(vehicle_dynamics.dependent_vars) else None
                                    if var_name and next_state.has(var_name):
                                        new_state = np.append(new_state, float(next_state.get(var_name)))
                                    else:
                                        new_state = np.append(new_state, vehicle_state[i])
                            vehicle_state = new_state
                        else:
                            dt = test_config.timestep
                            vehicle_state[0] += vehicle_state[3] * np.cos(vehicle_state[2]) * dt
                            vehicle_state[1] += vehicle_state[3] * np.sin(vehicle_state[2]) * dt
 
                    # Update obstacle states using obstacle manager
                    if hasattr(self, 'obstacle_manager'):
                        self.obstacle_manager.update_obstacle_states(test_config.timestep)
                        
                        # Get updated obstacle states
                        for i, obstacle in enumerate(obstacles):
                            if i < len(obstacle_states):
                                obstacle_state = self.obstacle_manager.get_obstacle_at_time(i, len(obstacle_states[i]))
                                if obstacle_state is not None:
                                    obstacle_states[i].append(obstacle_state[:2].copy())  # Only position
                                else:
                                    # Fallback to prediction steps
                                    if step < len(obstacle.prediction.steps):
                                        obstacle_state = obstacle.prediction.steps[step].position
                                        obstacle_states[i].append(obstacle_state.copy())
                                    else:
                                        # Extrapolate
                                        last_state = obstacle_states[i][-1] if obstacle_states[i] else obstacle.position
                                        obstacle_states[i].append(last_state.copy())
                    else:
                        # Fallback to original method
                        for i, obstacle in enumerate(obstacles):
                            if i < len(obstacle_states):
                                if step < len(obstacle.prediction.steps):
                                    obstacle_state = obstacle.prediction.steps[step].position
                                    obstacle_states[i].append(obstacle_state.copy())
                                else:
                                    # Extrapolate
                                    last_state = obstacle_states[i][-1] if obstacle_states[i] else obstacle.position
                                    obstacle_states[i].append(last_state.copy())
                    
                    vehicle_states.append(vehicle_state.copy())
                    # Collision check: mark violation if vehicle overlaps any obstacle
                    robot_radius = self.config.get("robot", {}).get("radius", 0.5)
                    any_violation = False
                    for i, obs_states in enumerate(obstacle_states):
                        if obs_states:
                            obs_pos = obs_states[-1]
                            dx = vehicle_state[0] - obs_pos[0]
                            dy = vehicle_state[1] - obs_pos[1]
                            dist = np.hypot(dx, dy)
                            # Use obstacle radius from config default if not available
                            obs_radius = self.config.get("obstacle_radius", 0.35)
                            if dist < (robot_radius + obs_radius):
                                any_violation = True
                                break
                    constraint_violations.append(any_violation)
                    
                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    logger.error(f"MPC solve failed at step {step}: {e}")
                    logger.error(f"Traceback: {tb_str}")
                    # Log state of critical data structures
                    logger.error(f"robot_area length: {len(data.robot_area) if hasattr(data, 'robot_area') and data.robot_area else 0}")
                    logger.error(f"dynamic_obstacles count: {len(data.dynamic_obstacles) if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles else 0}")
                    if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
                        for i, obs in enumerate(data.dynamic_obstacles):
                            steps_len = len(obs.prediction.steps) if hasattr(obs, 'prediction') and obs.prediction and hasattr(obs.prediction, 'steps') else 0
                            logger.error(f"  Obstacle {i}: prediction steps = {steps_len}")
                    constraint_violations.append(True)
                    
                computation_time = time.time() - start_time
                computation_times.append(computation_time)
                
                logger.info(f"Step {step} completed in {computation_time:.3f}s")

                # Reset solver each iteration to avoid accumulation
                try:
                    self.solver.reset()
                except Exception:
                    pass
                
                # If goal reached, add a few more frames to show final state, then break
                if goal_reached:
                    # Add 2-3 more frames to show the vehicle at the goal
                    for _ in range(3):
                        vehicle_states.append(vehicle_state.copy())
                        # Keep obstacles stationary at final position
                        for i, obs_states in enumerate(obstacle_states):
                            if obs_states:
                                obs_states.append(obs_states[-1].copy())
                        constraint_violations.append(False)
                        computation_times.append(0.0)
                    logger.info(f"Goal reached at step {goal_reached_step}, added {3} extra frames for visualization")
                    break
                    
            # Save results
            self.save_state_history(output_folder, vehicle_states, obstacle_states)
            # Store data object for goal plotting in animation
            self.last_data = data
            self.create_animation(output_folder, vehicle_states, obstacle_states, test_config, goal_reached_step)
            
            logger.info("Test completed successfully")
            
            return TestResult(
                success=True,
                vehicle_states=vehicle_states,
                obstacle_states=obstacle_states,
                computation_times=computation_times,
                constraint_violations=constraint_violations,
                output_folder=output_folder
            )
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return TestResult(
                success=False,
                vehicle_states=[],
                obstacle_states=[],
                computation_times=[],
                constraint_violations=[],
                output_folder=output_folder
            )
            
    def copy_test_script(self, output_folder: str):
        """Copy the test script to output folder."""
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back
        caller_file = caller_frame.f_globals['__file__']
        
        script_name = os.path.basename(caller_file)
        dest_path = os.path.join(output_folder, script_name)
        shutil.copy2(caller_file, dest_path)
        
    def save_state_history(self, output_folder: str, vehicle_states: List[np.ndarray], 
                          obstacle_states: List[List[np.ndarray]]):
        """Save state history to CSV files."""
        # Save vehicle states
        vehicle_file = os.path.join(output_folder, "vehicle_states.csv")
        with open(vehicle_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'x', 'y', 'theta', 'velocity'])
            
            for i, state in enumerate(vehicle_states):
                writer.writerow([i * self.solver.timestep] + state.tolist())
                
        # Save obstacle states with detailed information from obstacle manager
        if hasattr(self, 'obstacle_manager'):
            obstacle_info = self.obstacle_manager.get_obstacle_info()
            
            # Save obstacle summary
            summary_file = os.path.join(output_folder, "obstacle_summary.csv")
            with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['obstacle_id', 'dynamics_type', 'radius', 'initial_x', 'initial_y', 'initial_angle'])
                
                for obs_info in obstacle_info['obstacle_details']:
                    writer.writerow([
                        obs_info['id'],
                        obs_info['dynamics_type'],
                        obs_info['radius'],
                        obs_info['position'][0],
                        obs_info['position'][1],
                        obs_info['angle']
                    ])
            
            # Save detailed obstacle states
            all_obstacle_states = self.obstacle_manager.get_all_obstacle_states()
            for i, obs_states in enumerate(all_obstacle_states):
                obstacle_file = os.path.join(output_folder, f"obstacle_{i}_detailed_states.csv")
                with open(obstacle_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Determine state variables based on dynamics type
                    dynamics_type = obstacle_info['obstacle_details'][i]['dynamics_type']
                    if 'Unicycle' in dynamics_type:
                        writer.writerow(['time', 'x', 'y', 'psi', 'v'])
                    elif 'Bicycle' in dynamics_type:
                        writer.writerow(['time', 'x', 'y', 'psi', 'v', 'delta', 'spline'])
                    elif 'PointMass' in dynamics_type:
                        writer.writerow(['time', 'x', 'y', 'vx', 'vy'])
                    else:
                        writer.writerow(['time', 'x', 'y'])
                    
                    for j, state in enumerate(obs_states):
                        writer.writerow([j * self.solver.timestep] + state.tolist())
        else:
            # Fallback to simple obstacle states
            for i, obs_states in enumerate(obstacle_states):
                obstacle_file = os.path.join(output_folder, f"obstacle_{i}_states.csv")
                with open(obstacle_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['time', 'x', 'y'])
                    
                    for j, state in enumerate(obs_states):
                        writer.writerow([j * self.solver.timestep] + state.tolist())
                    
    def create_animation(self, output_folder: str, vehicle_states: List[np.ndarray], 
                        obstacle_states: List[List[np.ndarray]], test_config: TestConfig, 
                        goal_reached_step: Optional[int] = None):
        """Create GIF animation of the test showing full trajectory until goal is reached."""
        if not vehicle_states:
            logging.getLogger("integration_test").warning("No vehicle states to animate")
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine plot bounds from actual trajectory data
        all_x = [state[0] for state in vehicle_states]
        all_y = [state[1] for state in vehicle_states]
        
        # Add obstacle positions for bounds calculation
        for obs_states in obstacle_states:
            if obs_states:
                all_x.extend([obs[0] for obs in obs_states])
                all_y.extend([obs[1] for obs in obs_states])
        
        # Add reference path points for bounds
        if test_config.reference_path is not None:
            all_x.extend(test_config.reference_path[:, 0])
            all_y.extend(test_config.reference_path[:, 1])
        
        # Set up plot with dynamic bounds
        margin = 2.0
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Add goal reached info to title
        title = f'Integration Test: {test_config.test_name}'
        if goal_reached_step is not None:
            title += f' (Goal reached at step {goal_reached_step})'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot reference path
        if test_config.reference_path is not None:
            ax.plot(test_config.reference_path[:, 0], test_config.reference_path[:, 1], 
                   'k--', linewidth=2, label='Reference Path', alpha=0.7)
        
        # Plot goal location if available
        try:
            # Try to get goal from data object (stored during test setup)
            if hasattr(self, 'last_data') and hasattr(self.last_data, 'goal'):
                goal = self.last_data.goal
                if goal is not None and len(goal) >= 2:
                    ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal', zorder=10)
        except Exception:
            pass
        
        # Initialize plot elements
        vehicle_plot, = ax.plot([], [], 'bo', markersize=8, label='Vehicle', zorder=5)
        vehicle_trail, = ax.plot([], [], 'b-', linewidth=1, alpha=0.3, label='Vehicle Trail')
        obstacle_plots = []
        obstacle_trails = []
        for i in range(len(obstacle_states)):
            plot, = ax.plot([], [], 'ro', markersize=6, label=f'Obstacle {i}', zorder=4)
            obstacle_plots.append(plot)
            trail, = ax.plot([], [], 'r-', linewidth=1, alpha=0.2)
            obstacle_trails.append(trail)
            
        # Add vehicle radius circle
        vehicle_circle = plt.Circle((0, 0), self.config.get("robot", {}).get("radius", 0.5), 
                                  color='blue', alpha=0.2, zorder=3)
        ax.add_patch(vehicle_circle)
        
        # Add obstacle radius circles
        obstacle_circles = []
        for i in range(len(obstacle_states)):
            circle = plt.Circle((0, 0), self.config.get("obstacle_radius", 0.35), 
                              color='red', alpha=0.2, zorder=2)
            obstacle_circles.append(circle)
            ax.add_patch(circle)
        
        ax.legend(loc='upper right')
        
        # Track trails
        vehicle_trail_x = []
        vehicle_trail_y = []
        obstacle_trail_x = [[] for _ in range(len(obstacle_states))]
        obstacle_trail_y = [[] for _ in range(len(obstacle_states))]
        
        def animate(frame):
            if frame < len(vehicle_states):
                # Update vehicle
                vehicle_state = vehicle_states[frame]
                vehicle_plot.set_data([vehicle_state[0]], [vehicle_state[1]])
                vehicle_circle.center = (vehicle_state[0], vehicle_state[1])
                
                # Update vehicle trail
                vehicle_trail_x.append(vehicle_state[0])
                vehicle_trail_y.append(vehicle_state[1])
                vehicle_trail.set_data(vehicle_trail_x, vehicle_trail_y)
                
                # Update obstacles
                for i, obs_states in enumerate(obstacle_states):
                    if frame < len(obs_states):
                        obs_state = obs_states[frame]
                        obstacle_plots[i].set_data([obs_state[0]], [obs_state[1]])
                        obstacle_circles[i].center = (obs_state[0], obs_state[1])
                        
                        # Update obstacle trail
                        obstacle_trail_x[i].append(obs_state[0])
                        obstacle_trail_y[i].append(obs_state[1])
                        obstacle_trails[i].set_data(obstacle_trail_x[i], obstacle_trail_y[i])
                        
            return [vehicle_plot, vehicle_trail] + obstacle_plots + obstacle_trails + [vehicle_circle] + obstacle_circles
            
        # Create animation - use all frames to show complete trajectory
        total_frames = len(vehicle_states)
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=100, blit=True, repeat=True)
        
        # Save as GIF with appropriate fps (10 fps = 100ms per frame)
        gif_path = os.path.join(output_folder, "animation.gif")
        anim.save(gif_path, writer='pillow', fps=10)
        
        logging.getLogger("integration_test").info(f"Saved animation with {total_frames} frames to {gif_path}")
        plt.close(fig)


def create_reference_path(path_type: str = "straight", length: float = 20.0) -> np.ndarray:
    """Create reference path for testing."""
    if path_type == "straight":
        return np.array([[0.0, 0.0], [length, 0.0]])
    elif path_type == "curve":
        t = np.linspace(0, length, 100)
        x = t
        y = 2.0 * np.sin(0.2 * t)
        return np.column_stack([x, y])
    elif path_type == "s_curve":
        t = np.linspace(0, length, 100)
        x = t
        y = 3.0 * np.sin(0.3 * t) * np.cos(0.1 * t)
        return np.column_stack([x, y])
    else:
        raise ValueError(f"Unknown path type: {path_type}")


# Example usage functions
def run_safe_horizon_test():
    """Example: Safe Horizon constraint test."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("straight", 20.0),
        objective_module="contouring",
        constraint_modules=["safe_horizon", "contouring"],
        vehicle_dynamics="bicycle",
        num_obstacles=3,
        obstacle_dynamics=["unicycle", "unicycle", "unicycle"],
        test_name="Safe Horizon Integration Test",
        duration=10.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


def run_gaussian_constraints_test():
    """Example: Gaussian constraints test."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("curve", 15.0),
        objective_module="goal",
        constraint_modules=["gaussian", "contouring"],
        vehicle_dynamics="unicycle",
        num_obstacles=2,
        obstacle_dynamics=["unicycle", "bicycle"],
        test_name="Gaussian Constraints Integration Test",
        duration=8.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


def run_ellipsoid_constraints_test():
    """Example: Ellipsoid constraints test."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("s_curve", 18.0),
        objective_module="contouring",
        constraint_modules=["ellipsoid", "contouring"],
        vehicle_dynamics="bicycle",
        num_obstacles=3,
        obstacle_dynamics=["unicycle", "unicycle", "unicycle"],
        test_name="Ellipsoid Constraints Integration Test",
        duration=12.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


def run_decomp_constraints_test():
    """Example: Decomposition constraints test."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("curve", 20.0),
        objective_module="contouring",
        constraint_modules=["decomp", "contouring"],
        vehicle_dynamics="bicycle",
        num_obstacles=4,
        obstacle_dynamics=["unicycle", "bicycle", "point_mass", "unicycle"],
        test_name="Decomposition Constraints Integration Test",
        duration=10.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


def run_guidance_constraints_test():
    """Example: Guidance constraints test."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("straight", 22.0),
        objective_module="goal",
        constraint_modules=["guidance", "contouring"],
        vehicle_dynamics="unicycle",
        num_obstacles=2,
        obstacle_dynamics=["unicycle", "bicycle"],
        test_name="Guidance Constraints Integration Test",
        duration=8.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


def run_scenario_constraints_test():
    """Example: Scenario constraints test."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("s_curve", 25.0),
        objective_module="contouring",
        constraint_modules=["scenario", "contouring"],
        vehicle_dynamics="bicycle",
        num_obstacles=3,
        obstacle_dynamics=["unicycle", "unicycle", "unicycle"],
        test_name="Scenario Constraints Integration Test",
        duration=15.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


def run_multi_objective_test():
    """Example: Multiple objectives test."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("curve", 20.0),
        objective_module="path_reference_velocity",
        constraint_modules=["gaussian", "contouring"],
        vehicle_dynamics="bicycle",
        num_obstacles=3,
        obstacle_dynamics=["unicycle", "unicycle", "unicycle"],
        test_name="Path Reference Velocity Integration Test",
        duration=10.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


def run_comprehensive_test():
    """Example: Comprehensive test with multiple constraint types."""
    framework = IntegrationTestFramework()
    
    test_config = TestConfig(
        reference_path=create_reference_path("s_curve", 30.0),
        objective_module="contouring",
        constraint_modules=["safe_horizon", "gaussian", "ellipsoid", "contouring"],
        vehicle_dynamics="bicycle",
        num_obstacles=5,
        obstacle_dynamics=["unicycle", "bicycle", "point_mass", "unicycle", "bicycle"],
        test_name="Comprehensive Multi-Constraint Integration Test",
        duration=20.0,
        timestep=0.1
    )
    
    result = framework.run_test(test_config)
    return result


if __name__ == "__main__":
    # Run example tests
    print("Running Safe Horizon Integration Test...")
    result1 = run_safe_horizon_test()
    print(f"Test 1 completed: {result1.success}")
    
    print("Running Gaussian Constraints Integration Test...")
    result2 = run_gaussian_constraints_test()
    print(f"Test 2 completed: {result2.success}")
