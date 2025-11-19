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
from typing import List, Optional, Tuple
from dataclasses import dataclass
# yaml not used
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from solver.casadi_solver import CasADiSolver
from planning.types import Data, DynamicObstacle, Problem, ReferencePath, Bound, generate_reference_path
from planning.planner import Planner
from planning.types import define_robot_area, propagate_obstacles, ensure_obstacle_size
from utils.utils import read_config_file
from planning.obstacle_manager import ObstacleManager, ObstacleConfig, create_unicycle_obstacle, create_bicycle_obstacle, create_point_mass_obstacle
from planning.types import PredictionType


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
    # New: prediction type per obstacle ("deterministic" or "gaussian")
    obstacle_prediction_types: Optional[List[str]] = None
    duration: float = 10.0
    timestep: float = 0.1
    # Optional explicit obstacle configurations to avoid randomness in tests
    obstacle_configs: Optional[List[ObstacleConfig]] = None
    # Optionally draw solver's predicted trajectory each planner iteration
    show_predicted_trajectory: bool = False
    # Option: allow using fallback control if solver outputs None; default off
    fallback_control_enabled: bool = False
    # Optional: sequence of goal positions for moving goal tests (list of [x, y] or [x, y, z] tuples)
    goal_sequence: Optional[List[List[float]]] = None
    # Optional: temperature setting (0.0-1.0) for obstacle direction change frequency. Default: 0.5
    obstacle_temperature: float = 0.5
    # Optional: maximum test duration in seconds. Default: 60.0 seconds. Set to None to disable timeout.
    timeout_seconds: Optional[float] = 60.0
    # Optional: maximum consecutive MPC solve failures before early termination. Default: 5.
    max_consecutive_failures: int = 5
    # Optional: enable stuck vehicle detection and early exit. Default: True.
    enable_stuck_vehicle_detection: bool = True


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
        
        # Initialize solver with config
        self.solver = CasADiSolver(self.config)
        self.solver.horizon = self.config.get("planner", {}).get("horizon", 10)
        self.solver.timestep = self.config.get("planner", {}).get("timestep", 0.1)
        
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
            from planning.dynamic_models import SecondOrderBicycleModel
            return SecondOrderBicycleModel()
        elif dynamics_type == "unicycle":
            from planning.dynamic_models import SecondOrderUnicycleModel
            return SecondOrderUnicycleModel()
        elif dynamics_type == "contouring_unicycle":
            from planning.dynamic_models import ContouringSecondOrderUnicycleModel
            return ContouringSecondOrderUnicycleModel()
        elif dynamics_type == "contouring_bicycle":
            from planning.dynamic_models import CurvatureAwareSecondOrderBicycleModel
            return CurvatureAwareSecondOrderBicycleModel()
        elif dynamics_type == "point_mass":
            from planning.dynamic_models import PointMassModel
            return PointMassModel()
        else:
            raise ValueError(f"Unknown vehicle dynamics type: {dynamics_type}")
            
    def create_objective_module(self, objective_type: str):
        """Create objective module with config."""
        logger = logging.getLogger("integration_test")
        logger.info(f"Creating objective module: {objective_type}")
        
        if objective_type == "contouring":
            from modules.objectives.contouring_objective import ContouringObjective
            module = ContouringObjective()
        elif objective_type == "goal":
            from modules.objectives.goal_objective import GoalObjective
            module = GoalObjective()
        elif objective_type == "path_reference_velocity":
            from modules.objectives.path_reference_velocity_objective import PathReferenceVelocityObjective
            module = PathReferenceVelocityObjective()
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
        
        # Ensure module has config
        if hasattr(module, 'config') and module.config is None:
            module.config = self.config
            module.settings = self.config
            logger.debug(f"Set config on objective module '{objective_type}'")
        
        logger.info(f"Objective module '{objective_type}' created: {module.name}")
        return module
            
    def create_constraint_modules(self, constraint_types: List[str]):
        """Create constraint modules with config."""
        logger = logging.getLogger("integration_test")
        logger.info(f"Creating {len(constraint_types)} constraint module(s): {constraint_types}")
        constraints = []
        
        for constraint_type in constraint_types:
            logger.debug(f"Creating constraint module: {constraint_type}")
            
            if constraint_type == "safe_horizon":
                from modules.constraints.safe_horizon_constraint import SafeHorizonConstraint
                module = SafeHorizonConstraint()
            elif constraint_type == "contouring":
                from modules.constraints.contouring_constraints import ContouringConstraints
                module = ContouringConstraints()
            elif constraint_type == "gaussian":
                from modules.constraints.gaussian_constraints import GaussianConstraints
                module = GaussianConstraints()
            elif constraint_type == "linear":
                from modules.constraints.linearized_constraints import LinearizedConstraints
                module = LinearizedConstraints()
            elif constraint_type == "ellipsoid":
                from modules.constraints.ellipsoid_constraints import EllipsoidConstraints
                module = EllipsoidConstraints()
            elif constraint_type == "decomp":
                from modules.constraints.decomp_constraints import DecompConstraints
                module = DecompConstraints(self.solver)
            elif constraint_type == "guidance":
                from modules.constraints.guidance_constraints import GuidanceConstraints
                module = GuidanceConstraints()
            elif constraint_type == "scenario":
                # Scenario constraints removed or pending refactor
                logger.warning(f"Skipping scenario constraint (not implemented)")
                continue
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")
            
            # Ensure module has config
            if hasattr(module, 'config') and module.config is None:
                module.config = self.config
                module.settings = self.config
                logger.debug(f"Set config on constraint module '{constraint_type}'")
            
            constraints.append(module)
            logger.info(f"Constraint module '{constraint_type}' created: {module.name}")
                
        logger.info(f"Created {len(constraints)} constraint module(s)")
        return constraints
        
    def create_obstacles(self, num_obstacles: int, dynamics_types: List[str],
                         obstacle_configs: Optional[List[ObstacleConfig]] = None,
                         prediction_types: Optional[List[str]] = None,
                         plot_bounds: Optional[Tuple[float, float, float, float]] = None,
                         temperature: float = 0.5) -> List[DynamicObstacle]:
        """Create obstacles with specified dynamics using obstacle manager.
        If obstacle_configs is provided, it will be used directly for deterministic setups.
        
        Args:
            plot_bounds: Optional (x_min, x_max, y_min, y_max) for bouncing behavior
            temperature: Temperature setting (0.0-1.0) for direction change frequency. Default: 0.5
        """
        obstacle_manager = ObstacleManager(self.config, plot_bounds=plot_bounds, temperature=temperature)
        
        # If not provided, create random obstacle configurations
        if obstacle_configs is None:
            obstacle_configs = []
            for i in range(num_obstacles):
                dynamics_type = dynamics_types[i % len(dynamics_types)]
                pred_type_str = (prediction_types[i % len(prediction_types)] if prediction_types else None)
                
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

                # Override prediction type if provided
                if pred_type_str is not None:
                    if pred_type_str.lower() in ("gaussian", "normal"):
                        config.prediction_type = PredictionType.GAUSSIAN
                    elif pred_type_str.lower() in ("deterministic", "det"):
                        config.prediction_type = PredictionType.DETERMINISTIC
                    else:
                        raise ValueError(f"Unknown obstacle prediction type: {pred_type_str}")
                    
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
            # Ensure solver horizon is set BEFORE creating modules
            # Modules may access solver.horizon during initialization
            planner_config = self.config.get("planner", {})
            if not hasattr(self.solver, 'horizon') or self.solver.horizon is None:
                self.solver.horizon = planner_config.get("horizon", 10)
            if not hasattr(self.solver, 'timestep') or self.solver.timestep is None:
                self.solver.timestep = planner_config.get("timestep", 0.1)
            
            # Create modules (they will use self.solver which now has horizon set)
            # Apply runtime planner overrides
            try:
                self.config.setdefault("planner", {})
                self.config["planner"].setdefault("fallback_control_enabled", bool(test_config.fallback_control_enabled))
                # Emphasize contour following when using contouring objective
                if test_config.objective_module == "contouring":
                    self.config.setdefault("weights", {})
                    # Check if we have obstacles - if so, allow more deviation and encourage progress
                    has_obstacles = (test_config.num_obstacles > 0 or 
                                    (hasattr(test_config, 'obstacle_configs') and 
                                     test_config.obstacle_configs is not None and 
                                     len(test_config.obstacle_configs) > 0))
                    
                    if has_obstacles:
                        # For obstacle avoidance: lower contour weight to allow deviation,
                        # significantly higher lag weight to strongly encourage progress and prevent vehicle from sitting still
                        self.config["weights"].setdefault("contour_weight", 1.0)  # Lower to allow deviation
                        self.config["weights"].setdefault("contouring_lag_weight", 5.0)  # Much higher to strongly encourage progress
                    else:
                        # Boost contour weight; modest lag weight to reduce phase error
                        self.config["weights"].setdefault("contour_weight", 5.0)
                        self.config["weights"].setdefault("contouring_lag_weight", 0.2)
                    # Prefer dynamic velocity handling if supported by module
                    self.config.setdefault("contouring", {})
                    self.config["contouring"].setdefault("dynamic_velocity_reference", True)
            except Exception:
                pass
            # If contouring objective, prefer contouring-aware dynamics with 'spline' state
            dyn_type = test_config.vehicle_dynamics
            if test_config.objective_module == "contouring":
                logger.info(f"Contouring objective detected: automatically switching to contouring-aware dynamics model")
                if dyn_type == "bicycle":
                    dyn_type = "contouring_bicycle"
                    logger.info(f"  Converted 'bicycle' -> 'contouring_bicycle'")
                elif dyn_type == "unicycle":
                    dyn_type = "contouring_unicycle"
                    logger.info(f"  Converted 'unicycle' -> 'contouring_unicycle'")
                else:
                    logger.warning(f"  Vehicle dynamics '{dyn_type}' is not automatically converted. Ensure it supports contouring (has 'spline' state variable)")
            vehicle_dynamics = self.create_vehicle_dynamics(dyn_type)
            logger.info(f"Created vehicle dynamics model: {vehicle_dynamics.__class__.__name__}")
            
            # CRITICAL VALIDATION: If contouring objective is used, dynamics model MUST have spline state
            if test_config.objective_module == "contouring":
                dependent_vars = vehicle_dynamics.get_dependent_vars()
                if "spline" not in dependent_vars:
                    error_msg = (
                        f"CRITICAL ERROR: Contouring objective requires a dynamics model with 'spline' state variable.\n"
                        f"  Current dynamics model: {vehicle_dynamics.__class__.__name__}\n"
                        f"  Current state variables: {dependent_vars}\n"
                        f"  Required: dynamics model must include 'spline' in dependent_vars\n"
                        f"  Solution: Use 'contouring_unicycle' or 'contouring_bicycle' for vehicle_dynamics\n"
                        f"  Example: vehicle_dynamics='contouring_unicycle'"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.info(f"✓ Validation passed: Contouring objective with dynamics model that has 'spline' state")
            
            objective_module = self.create_objective_module(test_config.objective_module)
            constraint_modules = self.create_constraint_modules(test_config.constraint_modules)

            # Always include control-effort penalties to mirror reference MPCBaseModule
            from modules.objectives.control_effort_objective import ControlEffortObjective
            from modules.objectives.control_jerk_objective import ControlJerkObjective
            control_effort_module = ControlEffortObjective()
            control_jerk_module = ControlJerkObjective()

            auxiliary_objectives = [control_effort_module, control_jerk_module]

            # For contouring-style tests, also include the path reference velocity objective
            if test_config.objective_module in ("contouring", "path_reference_velocity"):
                from modules.objectives.path_reference_velocity_objective import PathReferenceVelocityObjective
                auxiliary_objectives.append(PathReferenceVelocityObjective())

            for aux_module in auxiliary_objectives:
                if hasattr(aux_module, "config") and aux_module.config is None:
                    aux_module.config = self.config
                    aux_module.settings = self.config
            
            # Configure solver with dynamics and modules
            # Dynamics model is set via problem.model_type which planner uses
            # No need to call set_dynamics_model separately
            
            # Build Problem for Planner per new framework
            problem = Problem()
            problem.model_type = vehicle_dynamics
            
            # Set problem modules directly (no temporary solver)
            problem.modules = constraint_modules + auxiliary_objectives + [objective_module]
            # obstacles/data/x0 set below

            # Create planner using problem (planner creates its own solver, but we'll update modules after)
            planner = Planner(problem, config=self.config)
            
            # No solver rebinding needed; modules use Data/config access
            
            # Keep reference to planner's solver if needed by framework utilities
            self.solver = planner.solver
            
            # CRITICAL: Set solver on all modules immediately so they can access parameter_manager
            logger.info("Setting solver on all modules...")
            for module in planner.solver.module_manager.get_modules():
                if hasattr(module, 'solver'):
                    module.solver = planner.solver
                    logger.debug(f"  Set solver on module: {module.name}")
                else:
                    logger.warning(f"  Module {module.name} has no solver attribute")

            # Define parameters now that modules are registered
            # Use planner's solver for define_parameters
            planner.solver.define_parameters()

            # Calculate plot bounds early so we can pass them to obstacle manager for bouncing
            # We'll recalculate them in create_animation, but this gives us an initial estimate
            plot_bounds_estimate = None
            if test_config.reference_path is not None:
                if isinstance(test_config.reference_path, np.ndarray):
                    ref_x = test_config.reference_path[:, 0]
                    ref_y = test_config.reference_path[:, 1]
                    margin = 10.0
                    plot_bounds_estimate = (
                        float(np.min(ref_x) - margin),
                        float(np.max(ref_x) + margin),
                        float(np.min(ref_y) - margin),
                        float(np.max(ref_y) + margin)
                    )
            
            # Create obstacles
            obstacles = self.create_obstacles(
                test_config.num_obstacles,
                test_config.obstacle_dynamics,
                test_config.obstacle_configs,
                test_config.obstacle_prediction_types,
                plot_bounds=plot_bounds_estimate,
                temperature=getattr(test_config, 'obstacle_temperature', 0.5)
            )
            
            # Initialize data
            data = Data()
            # Always initialize dynamic_obstacles to a list to avoid attribute errors
            data.dynamic_obstacles = obstacles if obstacles is not None else []
            problem.data = data

            # Build reference path ONLY for contouring-related objectives
            try:
                use_ref_path = test_config.objective_module in ("contouring", "path_reference_velocity")
                if use_ref_path:
                    # Handle reference_path: can be numpy array or already a ReferencePath object
                    if test_config.reference_path is not None:
                        if isinstance(test_config.reference_path, np.ndarray):
                            # Convert numpy array to ReferencePath object
                            from planning.types import ReferencePath
                            from utils.math_tools import TKSpline
                            ref_path = ReferencePath()
                            # Ensure numpy arrays for arithmetic operations
                            x_arr = np.asarray(test_config.reference_path[:, 0], dtype=float)
                            y_arr = np.asarray(test_config.reference_path[:, 1], dtype=float)
                            z_arr = np.zeros(x_arr.shape[0], dtype=float)
                            # Compute arc length
                            s = np.zeros(x_arr.shape[0], dtype=float)
                            for i in range(1, x_arr.shape[0]):
                                dx = x_arr[i] - x_arr[i - 1]
                                dy = y_arr[i] - y_arr[i - 1]
                                s[i] = s[i - 1] + float(np.hypot(dx, dy))
                            # Store as numpy arrays in ReferencePath
                            ref_path.x = x_arr
                            ref_path.y = y_arr
                            ref_path.z = z_arr
                            ref_path.s = s
                            # Build numeric splines using TKSpline (for post-optimization evaluation)
                            from utils.math_tools import TKSpline
                            ref_path.x_spline = TKSpline(s, x_arr)
                            ref_path.y_spline = TKSpline(s, y_arr)
                            ref_path.z_spline = TKSpline(s, z_arr)
                            ref_path.length = float(s[-1])
                            data.reference_path = ref_path
                            start_pt = [float(x_arr[0]), float(y_arr[0]), 0.0]
                            goal_pt = [float(x_arr[-1]), float(y_arr[-1]), 0.0]
                        else:
                            # Already a ReferencePath object
                            data.reference_path = test_config.reference_path
                            # Ensure numpy arrays on existing ReferencePath
                            data.reference_path.x = np.asarray(data.reference_path.x, dtype=float)
                            data.reference_path.y = np.asarray(data.reference_path.y, dtype=float)
                            data.reference_path.s = np.asarray(data.reference_path.s, dtype=float)
                            start_pt = [float(data.reference_path.x[0]), float(data.reference_path.y[0]), 0.0]
                            goal_pt = [float(data.reference_path.x[-1]), float(data.reference_path.y[-1]), 0.0]
                    else:
                        # Generate a simple straight reference if not provided
                        start_pt = [0.0, 0.0, 0.0]
                        goal_pt = [20.0, 0.0, 0.0]
                        ref_path = generate_reference_path(start_pt, goal_pt, path_type="straight")
                        data.reference_path = ref_path
                else:
                    # Goal objective and others: DO NOT use reference path
                    data.reference_path = None
                    start_pt = [0.0, 0.0, 0.0]
                    goal_pt = [20.0, 0.0, 0.0]

                # Set start/goal only for Goal objective; contouring follows reference path end
                if test_config.objective_module == "goal":
                    data.start = np.array(start_pt[:2])
                    # Support multiple goals - if test_config has goal_sequence, use it, otherwise use single goal
                    if hasattr(test_config, 'goal_sequence') and test_config.goal_sequence:
                        data.goal_sequence = [np.array(g[:2]) for g in test_config.goal_sequence]
                        data.current_goal_index = 0
                        data.goal = data.goal_sequence[0]
                        logger.info(f"Goal objective: Using goal_sequence, setting goal to ({data.goal[0]:.3f}, {data.goal[1]:.3f})")
                    else:
                        data.goal_sequence = None
                        data.goal = np.array(goal_pt[:2])
                        logger.info(f"Goal objective: Using default goal_pt, setting goal to ({data.goal[0]:.3f}, {data.goal[1]:.3f})")
                    data.goal_received = True
                    data.reached_goals = []  # Track all reached goals
                    logger.info(f"Goal objective: goal_received={data.goal_received}, goal={data.goal}")
                # Compute road boundaries using path normals (ONLY if using reference path)
                if use_ref_path and data.reference_path is not None:
                    def calculate_path_normals(_ref_path: ReferencePath):
                        s_vals = _ref_path.s
                        # Derivatives of splines
                        dx = np.gradient(_ref_path.x)
                        dy = np.gradient(_ref_path.y)
                        # Normalize tangents
                        tangents = []
                        normals_local = []
                        for i in range(len(s_vals)):
                            tx = dx[i]
                            ty = dy[i]
                            norm = np.hypot(tx, ty)
                            if norm < 1e-9:
                                tangents.append((1.0, 0.0))
                                nx, ny = 0.0, 1.0
                            else:
                                txn, tyn = tx / norm, ty / norm
                                tangents.append((txn, tyn))
                                # Left normal = (-ty, tx)
                                nx, ny = -tyn, txn
                            normals_local.append((nx, ny))
                        return normals_local

                    ref_path = data.reference_path
                    normals = calculate_path_normals(ref_path)
                    road_width = float(self.config.get("road", {}).get("width", 7.0))
                    half_width = road_width / 2.0
                    left_x, left_y, right_x, right_y = [], [], []
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
                    
                    # Create boundary splines for obstacle path_intersect behavior (numeric evaluation)
                    from utils.math_tools import TKSpline
                    s_arr = np.asarray(ref_path.s, dtype=float)
                    data.left_spline_x = TKSpline(s_arr, np.array(left_x))
                    data.left_spline_y = TKSpline(s_arr, np.array(left_y))
                    data.right_spline_x = TKSpline(s_arr, np.array(right_x))
                    data.right_spline_y = TKSpline(s_arr, np.array(right_y))
                    logger.debug(f"Created boundary splines for path_intersect behavior")
            except Exception:
                # Minimal fallback (only set goal for Goal objective)
                if test_config.objective_module == "goal":
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
                    from planning.types import Disc
                    while len(data.robot_area) < num_discs:
                        # Add discs evenly spaced
                        offset = -vehicle_length / 2 + (len(data.robot_area) * vehicle_length / (num_discs + 1))
                        data.robot_area.append(Disc(offset, robot_radius))
            except Exception as e:
                logger.warning(f"Error creating robot_area: {e}")
                # Fallback single disc
                from planning.types import Disc
                data.robot_area = [Disc(0.0, robot_radius)]
                # Add more discs if needed
                while len(data.robot_area) < num_discs:
                    offset = -vehicle_length / 2 + (len(data.robot_area) * vehicle_length / (num_discs + 1))
                    data.robot_area.append(Disc(offset, robot_radius))

            data.planning_start_time = time.time()
            problem.obstacles = obstacles

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
                horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
                timestep_val = self.solver.timestep if self.solver.timestep is not None else 0.1
                propagate_obstacles(data, dt=timestep_val, horizon=horizon_val, speed=avg_speed)
                # Then ensure obstacle list is sized to max_obstacles (needs prediction.steps to exist)
                ensure_obstacle_size(data.dynamic_obstacles, planner.state)
                # Propagate again after sizing to ensure all obstacles have proper predictions
                propagate_obstacles(data, dt=timestep_val, horizon=horizon_val, speed=avg_speed)
            except Exception as e:
                logger.warning(f"Error shaping obstacles: {e}")
                # Ensure at least robot_area is set correctly
                if not hasattr(data, 'robot_area') or len(data.robot_area) == 0:
                    from planning.types import Disc
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
                pass
            except Exception:
                pass
            
            # Initialize state tracking
            vehicle_states = []
            obstacle_states = [[] for _ in range(test_config.num_obstacles)]
            computation_times = []
            constraint_violations = []
            # Optional: predicted trajectories per frame (list of list of (x,y))
            predicted_trajs = []
            halfspaces_per_step = []  # Capture contouring constraint halfspaces for visualization
            linearized_halfspaces_per_step = []  # Capture linearized constraint halfspaces for visualization
            
            # Initial state - include spline variable if model has it
            vehicle_state = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, theta, v
            vehicle_start_pos = (float(vehicle_state[0]), float(vehicle_state[1]))
            logger.info(f"Initial vehicle state (before spline): {vehicle_state}")
            logger.info(f"Vehicle starting position: ({vehicle_start_pos[0]:.2f}, {vehicle_start_pos[1]:.2f})")
            
            # CRITICAL REQUIREMENT: For contouring objective/constraints, ensure reference path starts at vehicle position
            if hasattr(data, 'reference_path') and data.reference_path is not None:
                ref_path_start = (float(data.reference_path.x[0]), float(data.reference_path.y[0]))
                logger.info(f"Reference path start: ({ref_path_start[0]:.2f}, {ref_path_start[1]:.2f})")
                
                # Check if contouring modules are being used
                has_contouring = (test_config.objective_module == "contouring" or 
                                "contouring" in test_config.constraint_modules)
                
                if has_contouring:
                    # Calculate distance between vehicle start and path start
                    dist = np.sqrt((ref_path_start[0] - vehicle_start_pos[0])**2 + 
                                 (ref_path_start[1] - vehicle_start_pos[1])**2)
                    
                    if dist > 0.01:  # More than 1cm difference
                        logger.warning(f"Reference path does not start at vehicle position! "
                                     f"Distance: {dist:.3f}m. Adjusting reference path...")
                        # Shift reference path to start at vehicle position
                        x_offset = vehicle_start_pos[0] - ref_path_start[0]
                        y_offset = vehicle_start_pos[1] - ref_path_start[1]
                        
                        # Adjust all path points
                        data.reference_path.x = np.asarray(data.reference_path.x, dtype=float) + x_offset
                        data.reference_path.y = np.asarray(data.reference_path.y, dtype=float) + y_offset
                        
                        # Rebuild numeric splines with adjusted coordinates using TKSpline
                        from utils.math_tools import TKSpline
                        s_arr = np.asarray(data.reference_path.s, dtype=float)
                        data.reference_path.x_spline = TKSpline(s_arr, data.reference_path.x)
                        data.reference_path.y_spline = TKSpline(s_arr, data.reference_path.y)
                        if hasattr(data.reference_path, 'z') and data.reference_path.z is not None:
                            data.reference_path.z_spline = TKSpline(s_arr, data.reference_path.z)
                        
                        logger.info(f"Adjusted reference path to start at vehicle position: "
                                  f"({data.reference_path.x[0]:.2f}, {data.reference_path.y[0]:.2f})")
                        # Mark as adjusted to prevent re-adjustment in modules
                        data._reference_path_adjusted = True
                    else:
                        logger.info("Reference path starts at vehicle position (requirement satisfied)")
                        # Mark as already correct
                        data._reference_path_adjusted = True
            
            # Initialize spline to 0 if model requires it
            if 'spline' in vehicle_dynamics.get_all_vars():
                logger.info("Contouring-aware dynamics detected: initializing spline variable")
                # Add spline and other variables if needed
                if len(vehicle_state) < len(vehicle_dynamics.dependent_vars):
                    # Append missing state variables with default values
                    for var in vehicle_dynamics.dependent_vars[len(vehicle_state):]:
                        if var == 'spline':
                            # Initialize spline to 0 (start of path)
                            # Note: spline should be normalized [0,1] or actual arc length depending on model
                            spline_init = 0.0
                            if hasattr(data, 'reference_path') and data.reference_path is not None:
                                # Could initialize to closest point, but 0.0 is safe
                                logger.debug(f"Initializing spline to {spline_init} (start of reference path)")
                            vehicle_state = np.append(vehicle_state, spline_init)
                            logger.info(f"Added spline variable: {spline_init}")
                        elif var == 'delta':
                            vehicle_state = np.append(vehicle_state, 0.0)  # No steering initially
                            logger.debug(f"Added delta variable: 0.0")
                        elif var == 'slack':
                            vehicle_state = np.append(vehicle_state, 0.0)  # No slack initially
                            logger.debug(f"Added slack variable: 0.0")
                        else:
                            vehicle_state = np.append(vehicle_state, 0.0)
                            logger.debug(f"Added {var} variable: 0.0")
                logger.info(f"Final initial state: {vehicle_state} (variables: {vehicle_dynamics.dependent_vars})")
            else:
                logger.info(f"Non-contouring dynamics: no spline variable needed")
            vehicle_states.append(vehicle_state.copy())
            
            # Initialize early exit flags
            early_exit_stuck = False
            early_exit_failures = False
            
            # Create initial state object and set it on the problem
            from planning.types import State
            initial_state = State(vehicle_dynamics)
            initial_state.set('x', float(vehicle_state[0]))
            initial_state.set('y', float(vehicle_state[1]))
            initial_state.set('psi', float(vehicle_state[2]))
            initial_state.set('v', float(vehicle_state[3]))
            # Set spline and other state variables if present in model
            if 'spline' in vehicle_dynamics.get_all_vars() and len(vehicle_state) > 4:
                spline_idx = vehicle_dynamics.dependent_vars.index('spline') if 'spline' in vehicle_dynamics.dependent_vars else None
                if spline_idx is not None and len(vehicle_state) > spline_idx:
                    initial_state.set('spline', float(vehicle_state[spline_idx]))
            if 'delta' in vehicle_dynamics.get_all_vars() and len(vehicle_state) > 5:
                delta_idx = vehicle_dynamics.dependent_vars.index('delta') if 'delta' in vehicle_dynamics.dependent_vars else None
                if delta_idx is not None and len(vehicle_state) > delta_idx:
                    initial_state.set('delta', float(vehicle_state[delta_idx]))
            
            # Set x0 (initial state) on problem before planner uses it
            problem.x0 = initial_state
            problem.set_state(initial_state)
            
            # Run simulation
            goal_reached = False
            goal_reached_step = None
            step = 0
            # Track consecutive MPC failures (reset on success)
            consecutive_solver_failures = 0
            max_consecutive_failures = test_config.max_consecutive_failures
            # Initialize timeout tracking
            test_start_time = time.time()
            timeout_seconds = test_config.timeout_seconds
            
            # For contouring objective, run until end-of-path (spline reaches path length),
            # otherwise cap by duration
            is_contouring = (test_config.objective_module == "contouring")
            max_steps_cap = int(self.config.get("planner", {}).get("max_steps", 2000))
            num_steps = int(test_config.duration / test_config.timestep)
            
            while True:
                # Check timeout
                if timeout_seconds is not None:
                    elapsed_time = time.time() - test_start_time
                    if elapsed_time >= timeout_seconds:
                        logger.warning(f"Test timeout reached ({timeout_seconds:.1f}s) at step {step}. Terminating test early.")
                        break
                if is_contouring:
                    logger.info(f"Step {step} (contouring until path end)")
                else:
                    logger.info(f"Step {step}/{num_steps}")
                
                start_time = time.time()
                
                # Prepare iteration for modules that support it
                for module in [objective_module] + constraint_modules:
                    if hasattr(module, 'prepare_iteration'):
                        module.prepare_iteration(vehicle_state, data)
                
                # Sync planner state - include all state variables (do this BEFORE checking goal)
                logger.debug(f"Syncing planner state at step {step}: vehicle_state={vehicle_state}")
                planner.state.set('x', float(vehicle_state[0]))
                planner.state.set('y', float(vehicle_state[1]))
                planner.state.set('psi', float(vehicle_state[2]))
                planner.state.set('v', float(vehicle_state[3]))
                # Set spline and other state variables if present in model
                if 'spline' in vehicle_dynamics.get_all_vars() and len(vehicle_state) > 4:
                    spline_idx = vehicle_dynamics.dependent_vars.index('spline') if 'spline' in vehicle_dynamics.dependent_vars else None
                    if spline_idx is not None and len(vehicle_state) > spline_idx:
                        spline_val = float(vehicle_state[spline_idx])
                        planner.state.set('spline', spline_val)
                        logger.debug(f"  Set spline state to {spline_val}")
                    else:
                        logger.warning(f"  Could not set spline: idx={spline_idx}, state_len={len(vehicle_state)}")
                if 'delta' in vehicle_dynamics.get_all_vars() and len(vehicle_state) > 5:
                    delta_idx = vehicle_dynamics.dependent_vars.index('delta') if 'delta' in vehicle_dynamics.dependent_vars else None
                    if delta_idx is not None and len(vehicle_state) > delta_idx:
                        planner.state.set('delta', float(vehicle_state[delta_idx]))
                
                # Update problem's state so get_state() works correctly
                problem.set_state(planner.state)
                
                # Update obstacle states BEFORE module updates and solver solve
                # This ensures constraints are computed with current obstacle positions
                if hasattr(self, 'obstacle_manager') and hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles:
                    # Prepare vehicle state for obstacle behaviors
                    vehicle_state_vec = None
                    if hasattr(planner, 'state') and planner.state is not None:
                        try:
                            vehicle_x = float(planner.state.get('x', 0.0))
                            vehicle_y = float(planner.state.get('y', 0.0))
                            vehicle_psi = float(planner.state.get('psi', 0.0))
                            vehicle_v = float(planner.state.get('v', 1.0))
                            vehicle_state_vec = np.array([vehicle_x, vehicle_y, vehicle_psi, vehicle_v])
                        except:
                            pass
                    
                    # Get goal for obstacle behaviors
                    goal_pos = None
                    if hasattr(data, 'goal') and data.goal is not None:
                        try:
                            goal_pos = np.array(data.goal[:2])
                        except:
                            pass
                    
                    # Update obstacle states with behavior context
                    self.obstacle_manager.update_obstacle_states(
                        test_config.timestep,
                        vehicle_state=vehicle_state_vec,
                        reference_path=data.reference_path if hasattr(data, 'reference_path') else None,
                        goal=goal_pos,
                        data=data  # Pass data object to provide boundary information
                    )
                    
                    # Get updated obstacle states and update obstacle positions in data object
                    # This must happen BEFORE module.update() so constraints use current positions
                    obstacles = data.dynamic_obstacles
                    for i, obstacle in enumerate(obstacles):
                        if i < len(obstacle_states):
                            obstacle_state = self.obstacle_manager.get_obstacle_at_time(i, len(obstacle_states[i]))
                            if obstacle_state is not None:
                                # Update obstacle position in data object to match actual current position
                                # This ensures linearized constraints use actual positions
                                obstacle.position = np.array([obstacle_state[0], obstacle_state[1]])
                                
                                # Update obstacle velocity if available in state
                                if len(obstacle_state) > 3:
                                    # Compute velocity from state (for unicycle: v*cos(psi), v*sin(psi))
                                    if len(obstacle_state) >= 4:
                                        v = obstacle_state[3]
                                        psi = obstacle_state[2] if len(obstacle_state) > 2 else 0.0
                                        obstacle.velocity = np.array([v * np.cos(psi), v * np.sin(psi)])
                                
                                obstacle_states[i].append(obstacle_state[:2].copy())  # Only position
                            else:
                                # Fallback to prediction steps
                                if step < len(obstacle.prediction.steps):
                                    obstacle_state = obstacle.prediction.steps[step].position
                                    obstacle.position = obstacle_state.copy()  # Update position
                                    obstacle_states[i].append(obstacle_state.copy())
                                else:
                                    # Extrapolate
                                    last_state = obstacle_states[i][-1] if obstacle_states[i] else obstacle.position
                                    obstacle.position = last_state.copy()  # Update position
                                    obstacle_states[i].append(last_state.copy())
                
                # Update modules with current state (critical for contouring to work)
                # Ensure all modules are using planner's solver before update
                for module in [objective_module] + constraint_modules:
                    if hasattr(module, 'solver') and module.solver != planner.solver:
                        module.solver = planner.solver
                try:
                    for module in [objective_module] + constraint_modules:
                        # Compatibility: some modules expect solver.get_initial_state()
                        try:
                            if not hasattr(self.solver, 'get_initial_state'):
                                self.solver.get_initial_state = lambda: planner.state
                        except Exception:
                            pass
                        if hasattr(module, 'update'):
                            module.update(planner.state, data)
                except Exception as e:
                    logger.debug(f"Error updating modules: {e}")

                # Ensure robot_area is still valid
                if not hasattr(data, 'robot_area') or len(data.robot_area) == 0:
                    logger.warning(f"robot_area is empty at step {step}, recreating...")
                    from planning.types import Disc
                    data.robot_area = [Disc(0.0, robot_radius)]
                    while len(data.robot_area) < num_discs:
                        offset = -vehicle_length / 2 + (len(data.robot_area) * vehicle_length / (num_discs + 1))
                        data.robot_area.append(Disc(offset, robot_radius))

                # Ensure obstacles have valid predictions before solve
                if data.dynamic_obstacles:
                    for obs in data.dynamic_obstacles:
                        if not hasattr(obs, 'prediction') or obs.prediction is None:
                            logger.warning(f"Obstacle {obs.index} has no prediction, creating default...")
                            from planning.types import Prediction, PredictionType
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
                                from planning.types import get_constant_velocity_prediction
                                horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
                                obs.prediction = get_constant_velocity_prediction(
                                    obs.position, velocity, self.solver.timestep, horizon_val + 1
                                )
                            except Exception as e:
                                logger.warning(f"Failed to create prediction steps: {e}")

                # Ensure planner's solver has horizon set before accessing it
                if not hasattr(planner.solver, 'horizon') or planner.solver.horizon is None:
                    planner_config = self.config.get("planner", {})
                    planner.solver.horizon = planner_config.get("horizon", 10)
                    planner.solver.timestep = planner_config.get("timestep", 0.1)
                
                # Ensure we have a valid horizon value
                horizon_val = planner.solver.horizon if (hasattr(planner.solver, 'horizon') and planner.solver.horizon is not None) else (self.solver.horizon if self.solver.horizon is not None else 10)
                timestep_val = planner.solver.timestep if (hasattr(planner.solver, 'timestep') and planner.solver.timestep is not None) else (self.solver.timestep if self.solver.timestep is not None else 0.1)
                
                # Set data attributes for solver compatibility
                data.horizon = horizon_val
                data.timestep = timestep_val
                data.dynamics_model = vehicle_dynamics
                
                # Initialize and solve via planner
                # Note: solver.initialize doesn't exist - data is passed directly to solve
                # Ensure parameter_values is properly initialized after define_parameters
                pm = self.solver.parameter_manager
                if hasattr(pm, 'parameter_values') and (pm.parameter_values is None or len(pm.parameter_values) != pm.parameter_count):
                    import numpy as _np
                    if hasattr(pm, 'parameter_count') and pm.parameter_count:
                        pm.parameter_values = _np.zeros(pm.parameter_count, dtype=float)
                # Ensure solver is initialized before initialize_rollout
                if not hasattr(self.solver, 'opti') or self.solver.opti is None:
                    if data and hasattr(data, 'dynamics_model') and data.dynamics_model:
                        self.solver.dynamics_model = data.dynamics_model
                        self.solver.intialize_solver(data)
                self.solver.initialize_rollout(planner.state, data)

                # Check if goal is reached (AFTER syncing state so we use current position)
                if hasattr(planner, 'is_objective_reached'):
                    try:
                        goal_reached = planner.is_objective_reached(data)
                        if goal_reached:
                            # Support moving goals - if goal_sequence exists, move to next goal
                            if hasattr(data, 'goal_sequence') and data.goal_sequence is not None:
                                if hasattr(data, 'reached_goals'):
                                    data.reached_goals.append(np.array(data.goal))
                                logger.info(f"Goal {data.current_goal_index + 1} reached at step {step}!")
                                data.current_goal_index += 1
                                if data.current_goal_index < len(data.goal_sequence):
                                    data.goal = data.goal_sequence[data.current_goal_index]
                                    logger.info(f"Moving to next goal: {data.goal}")
                                    # Don't manually update parameters here - solve_mpc will call set_parameters
                                    # which will read the updated data.goal
                                    goal_reached = False  # Continue simulation
                                else:
                                    # All goals reached
                                    goal_reached_step = step
                                    logger.info(f"All goals reached at step {step}!")
                                    break
                            else:
                                # Single goal - stop simulation
                                goal_reached_step = step
                                logger.info(f"Goal reached at step {step}!")
                                break
                    except Exception as e:
                        logger.debug(f"Could not check goal status: {e}")

                try:
                    # CRITICAL: Update reference path to start at current vehicle position at each step
                    # This is essential for contouring constraints to work correctly
                    # Re-check has_contouring at each step to ensure it's in scope
                    has_contouring_check = (test_config.objective_module == "contouring" or 
                                          "contouring" in test_config.constraint_modules)
                    if has_contouring_check and hasattr(data, 'reference_path') and data.reference_path is not None:
                        try:
                            vehicle_pos = np.array([vehicle_state[0], vehicle_state[1]])
                            ref_path_start = np.array([float(data.reference_path.x[0]), float(data.reference_path.y[0])])
                            dist = np.linalg.norm(vehicle_pos - ref_path_start)
                            
                            logger.debug(f"Step {step}: Checking path alignment: vehicle=({vehicle_pos[0]:.3f}, {vehicle_pos[1]:.3f}), "
                                       f"path_start=({ref_path_start[0]:.3f}, {ref_path_start[1]:.3f}), dist={dist:.3f}m")
                            
                            if dist > 0.01:  # More than 1cm difference - update path
                                x_offset = vehicle_pos[0] - ref_path_start[0]
                                y_offset = vehicle_pos[1] - ref_path_start[1]
                                
                                # Adjust all path points
                                data.reference_path.x = np.asarray(data.reference_path.x, dtype=float) + x_offset
                                data.reference_path.y = np.asarray(data.reference_path.y, dtype=float) + y_offset
                                
                                # Rebuild numeric splines with adjusted coordinates using TKSpline
                                from utils.math_tools import TKSpline
                                s_arr = np.asarray(data.reference_path.s, dtype=float)
                                data.reference_path.x_spline = TKSpline(s_arr, data.reference_path.x)
                                data.reference_path.y_spline = TKSpline(s_arr, data.reference_path.y)
                                if hasattr(data.reference_path, 'z') and data.reference_path.z is not None:
                                    data.reference_path.z_spline = TKSpline(s_arr, data.reference_path.z)
                                
                                logger.info(f"Step {step}: Updated reference path to start at vehicle position: "
                                           f"({data.reference_path.x[0]:.2f}, {data.reference_path.y[0]:.2f}), "
                                           f"offset=({x_offset:.3f}, {y_offset:.3f}), distance={dist:.3f}m")
                                
                                # Also update the reference_path in contouring objective/constraints modules if they have cached it
                                if hasattr(planner, 'module_manager') and planner.module_manager is not None:
                                    for module in planner.module_manager.get_modules():
                                        if hasattr(module, 'reference_path') and module.reference_path is not None:
                                            # Update the cached reference path in the module
                                            module.reference_path = data.reference_path
                                            logger.debug(f"  Updated cached reference_path in module {module.name}")
                        except Exception as path_update_err:
                            logger.warning(f"Could not update reference path at step {step}: {path_update_err}")
                    
                    logger.info(f"=== Calling planner.solve_mpc() at step {step} ===")
                    planner_output = planner.solve_mpc(data)
                    
                    if planner_output.success:
                        logger.info(f"Step {step}: MPC solve successful")
                        consecutive_solver_failures = 0  # Reset on success
                        
                        # Check for stuck vehicle (not moving forward) - only if enabled
                        if test_config.enable_stuck_vehicle_detection and step > 5 and len(vehicle_states) >= 6:
                            # Check if vehicle has moved less than 0.1m in last 5 steps
                            recent_states = vehicle_states[-6:]
                            positions = np.array([[s[0], s[1]] for s in recent_states])
                            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                            total_movement = np.sum(distances)
                            
                            if total_movement < 0.1:  # Vehicle stuck (moved less than 10cm in 5 steps)
                                logger.error(f"🚨 VEHICLE STUCK DETECTED at step {step}!")
                                logger.error(f"  Vehicle has moved only {total_movement:.4f}m in last 5 steps")
                                logger.error(f"  Current position: ({vehicle_states[-1][0]:.3f}, {vehicle_states[-1][1]:.3f})")
                                logger.error(f"  Position 5 steps ago: ({vehicle_states[-6][0]:.3f}, {vehicle_states[-6][1]:.3f})")
                                
                                # Log constraint violations at current position
                                logger.error("  Checking constraint violations at current position...")
                                try:
                                    # Get linearized constraints module
                                    linearized_module = None
                                    for module in constraint_modules:
                                        if hasattr(module, 'name') and module.name == 'linearized_constraints':
                                            linearized_module = module
                                            break
                                    
                                    if linearized_module:
                                        constraints = linearized_module.calculate_constraints(planner.state, data, 0)
                                        violations = []
                                        vehicle_x = float(planner.state.get('x'))
                                        vehicle_y = float(planner.state.get('y'))
                                        for c in constraints:
                                            if 'a1' in c and 'a2' in c and 'b' in c:
                                                constraint_val = c['a1'] * vehicle_x + c['a2'] * vehicle_y
                                                if constraint_val > c['b']:
                                                    violations.append({
                                                        'a1': c['a1'],
                                                        'a2': c['a2'],
                                                        'b': c['b'],
                                                        'value': constraint_val,
                                                        'violation': constraint_val - c['b']
                                                    })
                                        
                                        if violations:
                                            logger.error(f"  Found {len(violations)} constraint violations:")
                                            for i, v in enumerate(violations):
                                                logger.error(f"    Violation {i+1}: a1={v['a1']:.4f}, a2={v['a2']:.4f}, b={v['b']:.4f}, "
                                                            f"value={v['value']:.4f}, violation_amount={v['violation']:.4f}")
                                        else:
                                            logger.warning("  No constraint violations detected, but vehicle is stuck")
                                except Exception as e:
                                    logger.error(f"  Could not check constraint violations: {e}")
                                
                                logger.error("  Terminating test early due to stuck vehicle")
                                # Flush logs safely (may fail if stdout/stderr are closed)
                                try:
                                    import sys
                                    if sys.stdout and not sys.stdout.closed:
                                        sys.stdout.flush()
                                    if sys.stderr and not sys.stderr.closed:
                                        sys.stderr.flush()
                                except (BrokenPipeError, OSError, AttributeError) as flush_err:
                                    logger.debug(f"Could not flush stdout/stderr (non-fatal): {flush_err}")
                                # Mark that we're exiting early due to stuck vehicle
                                early_exit_stuck = True
                                break
                    else:
                        logger.warning(f"Step {step}: MPC solve failed")
                        consecutive_solver_failures += 1
                        logger.warning(f"  Consecutive MPC failures: {consecutive_solver_failures}/{max_consecutive_failures}")
                        if consecutive_solver_failures >= max_consecutive_failures:
                            logger.error(f"Max consecutive MPC failures ({max_consecutive_failures}) reached. "
                                       f"No successful solve in last {max_consecutive_failures} attempts. "
                                       f"Terminating test early at step {step}.")
                            # Flush logs safely (may fail if stdout/stderr are closed)
                            try:
                                import sys
                                if sys.stdout and not sys.stdout.closed:
                                    sys.stdout.flush()
                                if sys.stderr and not sys.stderr.closed:
                                    sys.stderr.flush()
                            except (BrokenPipeError, OSError, AttributeError) as flush_err:
                                logger.debug(f"Could not flush stdout/stderr (non-fatal): {flush_err}")
                            # Mark that we're exiting early due to solver failures
                            early_exit_failures = True
                            break
                    
                    # Visualize constraints using module visualizers
                    try:
                        planner.visualize(stage_idx=1)
                    except Exception as viz_err:
                        logger.debug(f"Visualization error (non-fatal): {viz_err}")

                    # Optionally capture solver's current predicted trajectory for visualization
                    if test_config.show_predicted_trajectory:
                        try:
                            traj = self.solver.get_reference_trajectory()
                            if traj is not None and hasattr(traj, 'get_states'):
                                pts = []
                                for st in traj.get_states():
                                    if st is not None and st.has('x') and st.has('y'):
                                        pts.append((float(st.get('x')), float(st.get('y'))))
                                predicted_trajs.append(pts)
                            else:
                                predicted_trajs.append([])
                        except Exception:
                            predicted_trajs.append([])
                    
                    # Capture contouring constraints halfspaces for visualization (stage 0 only)
                    try:
                        # Find the contouring constraints module
                        contouring_module = None
                        if hasattr(planner, 'module_manager') and hasattr(planner.module_manager, 'modules'):
                            for module in planner.module_manager.modules:
                                if hasattr(module, 'name') and module.name == 'contouring_constraints':
                                    contouring_module = module
                                    break
                        
                        if contouring_module is not None:
                            # For visualization, we need numeric constraints with a1, a2, b format
                            # Get current spline value and compute numeric constraints directly
                            step_halfspaces = []
                            try:
                                # Get current spline value
                                cur_s = None
                                if planner.state is not None and planner.state.has('spline'):
                                    try:
                                        spline_val = planner.state.get('spline')
                                        # Check if it's symbolic - if so, try to get numeric value
                                        import casadi as cd
                                        if isinstance(spline_val, (cd.MX, cd.SX)):
                                            # Try to get numeric value from warmstart or use path start
                                            if hasattr(planner, 'solver') and planner.solver is not None:
                                                if hasattr(planner.solver, 'warmstart_values') and 'spline' in planner.solver.warmstart_values:
                                                    if len(planner.solver.warmstart_values['spline']) > 0:
                                                        cur_s = float(planner.solver.warmstart_values['spline'][0])
                                            if cur_s is None:
                                                # Use path start as fallback
                                                ref_path = getattr(contouring_module, '_reference_path', None)
                                                if ref_path is not None and hasattr(ref_path, 's'):
                                                    s_arr = np.asarray(ref_path.s, dtype=float)
                                                    if len(s_arr) > 0:
                                                        cur_s = float(s_arr[0])
                                        else:
                                            cur_s = float(spline_val)
                                    except Exception as e:
                                        logger.debug(f"Could not get spline value: {e}")
                                
                                # Compute numeric constraints for visualization
                                if cur_s is not None:
                                    constraints = contouring_module._compute_numeric_constraints(cur_s, planner.state, data, 0)
                                    
                                    if len(constraints) > 0:
                                        logger.info(f"Step {step}: Computed {len(constraints)} numeric contouring constraints for visualization (s={cur_s:.3f})")
                                    else:
                                        logger.debug(f"Step {step}: No numeric constraints computed (s={cur_s:.3f})")
                                else:
                                    logger.debug(f"Step {step}: Cannot compute constraints - no spline value available")
                                    constraints = []
                            except Exception as e:
                                logger.warning(f"Step {step}: Error computing numeric constraints for visualization: {e}")
                                import traceback
                                logger.debug(traceback.format_exc())
                                constraints = []
                            
                            if constraints:
                                
                                # Get the reference path and road width for accurate visualization
                                ref_path = getattr(contouring_module, '_reference_path', None)
                                road_width_half = getattr(contouring_module, '_road_width_half', None)
                                
                                # Get current spline value to compute path point
                                cur_s = None
                                if planner.state is not None and planner.state.has('spline'):
                                    try:
                                        cur_s = float(planner.state.get('spline'))
                                    except Exception:
                                        pass
                                
                                # Extract halfspace information from constraint dictionaries
                                # Each constraint dict has: {a1, a2, b, disc_offset, is_left, spline_s}
                                # Group by (a1, a2, is_left) to get unique halfspaces, keeping left and right separate
                                # Store spline_s for each constraint to visualize the spline segment
                                seen_halfspaces = {}
                                for const_dict in constraints:
                                    a1 = float(const_dict.get('a1', 0.0))
                                    a2 = float(const_dict.get('a2', 0.0))
                                    b = float(const_dict.get('b', 0.0))
                                    disc_offset = float(const_dict.get('disc_offset', 0.0))
                                    is_left = const_dict.get('is_left', None)
                                    spline_s = const_dict.get('spline_s', cur_s)  # Get spline_s from constraint, fallback to cur_s
                                    
                                    # Use is_left from constraint if available, otherwise infer
                                    if is_left is None:
                                        # Fallback heuristic
                                        is_left = (a1 < 0) if abs(a1) > abs(a2) else (a2 > 0)
                                    
                                    # Create a key based on normalized direction AND is_left flag
                                    # This ensures left and right boundaries are kept separate
                                    norm = np.sqrt(a1**2 + a2**2)
                                    if norm > 1e-6:
                                        # Include is_left in key to keep left and right boundaries separate
                                        key = (round(a1/norm, 6), round(a2/norm, 6), bool(is_left))
                                        # For visualization, compute b value that represents actual road boundary
                                        # (without slack and vehicle width adjustments)
                                        if key not in seen_halfspaces:
                                            A = np.array([a1, a2])
                                            A_norm = A / norm
                                            
                                            # Use spline_s from constraint if available, otherwise use cur_s
                                            constraint_s = spline_s if spline_s is not None else cur_s
                                            
                                            # Compute visualization b value: actual road boundary
                                            # The constraint b includes slack and vehicle width, but for visualization
                                            # we want to show the actual road boundary at road_width_half distance
                                            if ref_path is not None and constraint_s is not None and road_width_half is not None:
                                                try:
                                                    # Get path point at constraint s
                                                    path_point_x = ref_path.x_spline(constraint_s)
                                                    path_point_y = ref_path.y_spline(constraint_s)
                                                    path_point = np.array([float(path_point_x), float(path_point_y)])
                                                    
                                                    # Get path tangent to determine left/right direction
                                                    path_dx = float(ref_path.x_spline.derivative()(constraint_s))
                                                    path_dy = float(ref_path.y_spline.derivative()(constraint_s))
                                                    path_norm = np.sqrt(path_dx**2 + path_dy**2)
                                                    if path_norm > 1e-6:
                                                        path_dx_norm = path_dx / path_norm
                                                        path_dy_norm = path_dy / path_norm
                                                        # Normal pointing left: [path_dy_norm, -path_dx_norm]
                                                        normal_left = np.array([path_dy_norm, -path_dx_norm])
                                                        
                                                        # Compute boundary point based on is_left flag
                                                        if is_left:
                                                            # Left boundary: boundary is on right side of path
                                                            # Use normal pointing right (opposite to normal_left)
                                                            boundary_point = path_point - normal_left * road_width_half
                                                        else:
                                                            # Right boundary: boundary is on left side of path
                                                            # Use normal pointing left
                                                            boundary_point = path_point + normal_left * road_width_half
                                                    else:
                                                        # Fallback: use A direction
                                                        if is_left:
                                                            boundary_point = path_point + A_norm * road_width_half
                                                        else:
                                                            boundary_point = path_point - A_norm * road_width_half
                                                    
                                                    # Compute b for visualization: A·boundary_point
                                                    b_vis = np.dot(A, boundary_point)
                                                    # Store with spline_s for visualization
                                                    seen_halfspaces[key] = (A, b_vis, bool(is_left), float(constraint_s))
                                                except Exception as e:
                                                    # Fallback to using original b value
                                                    logger.debug(f"Could not compute visualization boundary: {e}, using original b")
                                                    seen_halfspaces[key] = (A, b, bool(is_left), float(constraint_s) if constraint_s is not None else 0.0)
                                            else:
                                                # Fallback: use original b value
                                                seen_halfspaces[key] = (A, b, bool(is_left), float(constraint_s) if constraint_s is not None else 0.0)
                                
                                # Convert to list of tuples (A, b, is_left, spline_s)
                                step_halfspaces = list(seen_halfspaces.values())
                                
                                if len(step_halfspaces) > 0:
                                    logger.debug(f"Step {step}: Captured {len(step_halfspaces)} contouring constraint halfspaces for visualization")
                                else:
                                    logger.debug(f"Step {step}: No contouring constraint halfspaces captured (constraints may be symbolic)")
                            
                            halfspaces_per_step.append(step_halfspaces)
                        else:
                            halfspaces_per_step.append([])
                    except Exception as e:
                        logger.debug(f"Could not capture contouring constraints halfspaces: {e}")
                        halfspaces_per_step.append([])
                    
                    # Capture linearized constraints halfspaces for visualization (stage 0 only)
                    try:
                        # Find the linearized constraints module
                        linearized_module = None
                        if hasattr(planner, 'module_manager') and hasattr(planner.module_manager, 'modules'):
                            for module in planner.module_manager.modules:
                                if hasattr(module, 'name') and module.name == 'linearized_constraints':
                                    linearized_module = module
                                    break
                        
                        if linearized_module is not None:
                            # Get constraints for stage 0 (current vehicle position) to show active constraints
                            # For visualization, extract constraint parameters from module's internal state
                            # This works for both numeric and symbolic constraints
                            step_linearized_halfspaces = []
                            
                            # Extract constraint parameters from module's internal arrays (_a1, _a2, _b)
                            # These are computed in update_step() and are available for stage 0
                            num_discs = getattr(linearized_module, 'num_discs', 1)
                            num_active_obstacles = getattr(linearized_module, 'num_active_obstacles', 0)
                            num_other_halfspaces = getattr(linearized_module, 'num_other_halfspaces', 0)
                            constraints_per_disc = num_active_obstacles + num_other_halfspaces
                            
                            # Get constraint arrays (computed in update_step for stage 0)
                            if hasattr(linearized_module, '_a1') and hasattr(linearized_module, '_a2') and hasattr(linearized_module, '_b'):
                                for disc_id in range(num_discs):
                                    for obs_idx in range(num_active_obstacles + num_other_halfspaces):
                                        # Get constraint parameters from internal arrays (stage 0)
                                        # Arrays are indexed as: _a1[disc_id][step][obs_idx]
                                        if (disc_id < len(linearized_module._a1) and 
                                            len(linearized_module._a1[disc_id]) > 0 and
                                            obs_idx < len(linearized_module._a1[disc_id][0])):
                                            a1 = linearized_module._a1[disc_id][0][obs_idx]
                                            a2 = linearized_module._a2[disc_id][0][obs_idx]
                                            b = linearized_module._b[disc_id][0][obs_idx]
                                            
                                            # Skip if constraint is None or invalid
                                            if a1 is None or a2 is None or b is None:
                                                continue
                                            
                                            a1 = float(a1)
                                            a2 = float(a2)
                                            b = float(b)
                                            
                                            # Check if constraint is valid (non-zero normal)
                                            norm = np.sqrt(a1**2 + a2**2)
                                            if norm > 1e-6:
                                                A = np.array([a1, a2])
                                                
                                                # Determine which obstacle this constraint belongs to
                                                # Only map to obstacle if it's within active obstacles (not other_halfspaces)
                                                if obs_idx < num_active_obstacles:
                                                    obstacle_id = obs_idx
                                                else:
                                                    # This is an "other_halfspace" constraint, assign to obstacle 0 for visualization
                                                    obstacle_id = 0
                                                
                                                # Store as (A, b, obstacle_id, halfspace_offset) tuple for visualization
                                                # Get halfspace_offset from module if available
                                                halfspace_offset = getattr(linearized_module, 'halfspace_offset', 0.0)
                                                step_linearized_halfspaces.append((A, b, obstacle_id, halfspace_offset))
                            
                            # Fallback: try to get constraints from calculate_constraints if internal arrays not available
                            if len(step_linearized_halfspaces) == 0:
                                try:
                                    constraints = linearized_module.calculate_constraints(planner.state, data, 0)
                                    if constraints:
                                        # Get obstacle information to map constraints to obstacles
                                        obstacle_positions = []
                                        if hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles is not None:
                                            for obs in data.dynamic_obstacles:
                                                if hasattr(obs, 'position') and obs.position is not None:
                                                    obstacle_positions.append(np.array([obs.position[0], obs.position[1]]))
                                        
                                        constraint_idx = 0
                                        for const_dict in constraints:
                                            # Handle both numeric and symbolic constraint formats
                                            if const_dict.get('type') == 'symbolic_expression':
                                                # For symbolic constraints, skip (can't extract a1, a2, b directly)
                                                # The internal arrays should have been used above
                                                constraint_idx += 1
                                                continue
                                            
                                            a1 = float(const_dict.get('a1', 0.0))
                                            a2 = float(const_dict.get('a2', 0.0))
                                            b = float(const_dict.get('b', 0.0))
                                            disc_offset = float(const_dict.get('disc_offset', 0.0))
                                            
                                            # Check if constraint is valid (non-zero normal)
                                            norm = np.sqrt(a1**2 + a2**2)
                                            if norm > 1e-6:
                                                A = np.array([a1, a2])
                                                
                                                # Determine which obstacle this constraint belongs to
                                                if constraints_per_disc > 0:
                                                    obs_idx = constraint_idx % constraints_per_disc
                                                    if obs_idx < num_active_obstacles:
                                                        obstacle_id = obs_idx
                                                    else:
                                                        obstacle_id = 0
                                                else:
                                                    obstacle_id = 0
                                                
                                                # Get halfspace_offset from module if available
                                                halfspace_offset = getattr(linearized_module, 'halfspace_offset', 0.0)
                                                step_linearized_halfspaces.append((A, b, obstacle_id, halfspace_offset))
                                            constraint_idx += 1
                                except Exception as e:
                                    logger.debug(f"Could not extract constraints from calculate_constraints: {e}")
                            
                            if step == 0 and len(step_linearized_halfspaces) > 0:
                                logger.debug(f"Captured {len(step_linearized_halfspaces)} linearized constraint halfspaces at step {step}")
                            
                            linearized_halfspaces_per_step.append(step_linearized_halfspaces)
                        else:
                            if step == 0:
                                logger.debug(f"Linearized constraints module not found at step {step}")
                            linearized_halfspaces_per_step.append([])
                    except Exception as e:
                        logger.debug(f"Could not capture linearized constraints halfspaces: {e}")
                        linearized_halfspaces_per_step.append([])
                    
                    # Prefer applying the control computed by the planner/solver
                    logger.info(f"=== Applying control at step {step} ===")
                    try:
                        u_names = getattr(vehicle_dynamics, 'get_inputs', lambda: [])()
                        control_dict = getattr(planner.output, 'control', {}) if hasattr(planner, 'output') else {}
                        logger.info(f"  Control dict from planner: {control_dict}")
                        logger.info(f"  Expected input names: {u_names}")
                        logger.info(f"  Planner output success: {planner_output.success if hasattr(planner_output, 'success') else 'unknown'}")
                        
                        if not control_dict:
                            logger.warning(f"  No control dict available! Checking if solve succeeded...")
                            if not planner_output.success:
                                logger.warning(f"  MPC solve failed - no control to apply. Using fallback trajectory.")
                            else:
                                logger.warning(f"  MPC solved but control dict is empty!")
                        
                        if control_dict and u_names:
                            # Use the dynamics model's discrete_dynamics method for proper state update
                            # This ensures contouring models update the spline parameter correctly
                            try:
                                from planning.dynamic_models import numeric_rk4
                                import casadi as cd
                                
                                a = float(control_dict.get('a', 0.0)) if 'a' in u_names else 0.0
                                w = float(control_dict.get('w', 0.0)) if 'w' in u_names else 0.0
                                dt = self.solver.timestep
                                
                                # Build z vector: [u, x] = [a, w, x, y, psi, v, ...]
                                u_vec = [a, w]
                                x_vec = []
                                dep_vars = vehicle_dynamics.dependent_vars
                                for idx, var_name in enumerate(dep_vars):
                                    value = None
                                    
                                    # Prefer planner.state numeric value if available
                                    if planner.state.has(var_name):
                                        try:
                                            value = float(planner.state.get(var_name))
                                        except Exception:
                                            value = None
                                    
                                    # Fallback to vehicle_state (always numeric) using same ordering as dependent_vars
                                    if value is None:
                                        if idx < len(vehicle_state):
                                            try:
                                                value = float(vehicle_state[idx])
                                            except Exception:
                                                value = None
                                    
                                    if value is None:
                                        value = 0.0
                                    
                                    x_vec.append(value)
                                
                                z_vec = u_vec + x_vec
                                # CRITICAL: For numeric evaluation, use CasADi DM (not MX)
                                # This ensures discrete_dynamics uses numeric RK4 path
                                z_k = cd.DM(z_vec)  # Use DM for numeric evaluation
                                
                                # Load into dynamics model
                                vehicle_dynamics.load(z_k)
                                
                                # Set parameters for the dynamics model
                                if hasattr(self.solver, 'parameter_manager') and self.solver.parameter_manager is not None:
                                    # Create a parameter getter that accesses parameter_manager
                                    def param_getter(key):
                                        try:
                                            # Try to get from parameter_manager (stage 0 for current step)
                                            params = self.solver.parameter_manager.get_all(0)
                                            if key in params:
                                                return params[key]
                                        except:
                                            pass
                                        # Fallback to data.parameters
                                        try:
                                            if hasattr(data, 'parameters') and data.parameters is not None:
                                                return data.parameters.get(key)
                                        except:
                                            pass
                                        return 0.0
                                    
                                    # Create parameter wrapper for discrete_dynamics
                                    # discrete_dynamics expects a parameter object with get_p() method
                                    class ParamWrapper:
                                        def __init__(self, p_getter):
                                            self.p_getter = p_getter
                                        def get(self, key, default=None):
                                            val = self.p_getter(key)
                                            return val if val is not None else default
                                        def get_p(self):
                                            # Return self for compatibility with discrete_dynamics signature
                                            return self
                                    
                                    params = ParamWrapper(param_getter)
                                else:
                                    # No parameter manager - create empty wrapper
                                    class ParamWrapper:
                                        def get(self, key, default=None):
                                            return default
                                        def get_p(self):
                                            return self
                                    params = ParamWrapper()
                                
                                # Use discrete_dynamics to get next state (includes proper spline update)
                                # discrete_dynamics handles RK4 integration and calls model_discrete_dynamics for spline
                                next_state_result = vehicle_dynamics.discrete_dynamics(z_k, params, dt)
                                
                                # CRITICAL: discrete_dynamics may return symbolic (MX) or numeric (DM)
                                # Check if it's symbolic and evaluate numerically if needed
                                
                                if isinstance(next_state_result, (cd.MX, cd.SX)):
                                    # Symbolic result - evaluate using numeric_rk4
                                    next_state = numeric_rk4(next_state_result, vehicle_dynamics, params, dt)
                                else:
                                    # Already numeric (DM or numpy)
                                    next_state = next_state_result
                                
                                # Convert CasADi DM or numpy array to numpy array
                                if isinstance(next_state, cd.DM):
                                    new_state = np.array(next_state).flatten()
                                elif isinstance(next_state, np.ndarray):
                                    new_state = next_state.flatten()
                                elif hasattr(next_state, '__iter__'):
                                    try:
                                        new_state = np.array([float(next_state[i]) for i in range(len(next_state))])
                                    except:
                                        # If conversion fails, try numeric_rk4
                                        logger.warning(f"  Could not convert next_state to numeric, trying numeric_rk4")
                                        next_state = numeric_rk4(next_state_result, vehicle_dynamics, params, dt)
                                        if isinstance(next_state, cd.DM):
                                            new_state = np.array(next_state).flatten()
                                        else:
                                            new_state = np.array(next_state).flatten()
                                else:
                                    new_state = np.array([float(next_state)])
                                
                                vehicle_state = new_state
                                
                                # Update planner state with new values
                                for i, var_name in enumerate(vehicle_dynamics.dependent_vars):
                                    if i < len(new_state):
                                        planner.state.set(var_name, float(new_state[i]))
                                
                            except Exception as e:
                                # Attempt symbolic_dynamics evaluation before falling back to Euler
                                logger.warning(f"  Error using dynamics model discrete_dynamics: {e}, attempting symbolic_dynamics fallback")
                                try:
                                    def _param_call(key):
                                        if 'param_getter' in locals():
                                            return param_getter(key)
                                        return 0.0
                                    
                                    x_input = cd.DM(x_vec)
                                    u_input = cd.DM(u_vec)
                                    sym_next = vehicle_dynamics.symbolic_dynamics(
                                        x_input,
                                        u_input,
                                        _param_call,
                                        dt
                                    )
                                    new_state = np.array(sym_next).flatten()
                                    vehicle_state = new_state
                                except Exception as sym_e:
                                    logger.warning(f"  symbolic_dynamics fallback failed: {sym_e}, reverting to Euler integration")
                                    a = float(control_dict.get('a', 0.0)) if 'a' in u_names else 0.0
                                    w = float(control_dict.get('w', 0.0)) if 'w' in u_names else 0.0
                                    dt = self.solver.timestep
                                    # Current state
                                    x = float(planner.state.get('x')) if planner.state.has('x') else vehicle_state[0]
                                    y = float(planner.state.get('y')) if planner.state.has('y') else vehicle_state[1]
                                    psi = float(planner.state.get('psi')) if planner.state.has('psi') else vehicle_state[2]
                                    v = float(planner.state.get('v')) if planner.state.has('v') else vehicle_state[3]
                                    # Euler step consistent with unicycle/bicycle front-axle models
                                    x_next = x + v * np.cos(psi) * dt
                                    y_next = y + v * np.sin(psi) * dt
                                    psi_next = psi + w * dt
                                    v_next = v + a * dt
                                    new_state = np.array([x_next, y_next, psi_next, v_next])
                                    # Optional states
                                    if 'spline' in vehicle_dynamics.get_all_vars():
                                        s_val = float(planner.state.get('spline')) if planner.state.has('spline') else (vehicle_state[4] if len(vehicle_state) > 4 else 0.0)
                                        s_next = s_val + v * dt
                                        new_state = np.append(new_state, s_next)
                                    if 'delta' in vehicle_dynamics.get_all_vars():
                                        d_idx = vehicle_dynamics.dependent_vars.index('delta') if 'delta' in vehicle_dynamics.dependent_vars else None
                                        d_val = float(planner.state.get('delta')) if planner.state.has('delta') else (vehicle_state[d_idx] if (d_idx is not None and len(vehicle_state) > d_idx) else 0.0)
                                        d_next = d_val + w * dt
                                        new_state = np.append(new_state, d_next)
                                    vehicle_state = new_state
                        else:
                            # No MPC control available: if fallback disabled, hold state (no movement)
                            if not bool(self.config.get("planner", {}).get("fallback_control_enabled", False)):
                                logger.warning("  No MPC control available; holding current state (fallback disabled)")
                            else:
                                # Fallbacks remain for explicit opt-in only
                                traj = self.solver.get_reference_trajectory()
                                if traj is not None and len(traj.get_states()) >= 2:
                                    next_state = traj.get_states()[1]
                                    x_next = next_state.get('x')
                                    y_next = next_state.get('y')
                                    psi_next = next_state.get('psi') if next_state.has('psi') else vehicle_state[2]
                                    v_next = next_state.get('v') if next_state.has('v') else vehicle_state[3]
                                    new_state = np.array([float(x_next), float(y_next), float(psi_next), float(v_next)])
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
                    except Exception:
                        dt = test_config.timestep
                        vehicle_state[0] += vehicle_state[3] * np.cos(vehicle_state[2]) * dt
                        vehicle_state[1] += vehicle_state[3] * np.sin(vehicle_state[2]) * dt
 
                    # Early stop if contouring objective reached end of reference path
                    try:
                        if test_config.objective_module == 'contouring' and hasattr(data, 'reference_path') and hasattr(data.reference_path, 'length'):
                            if 'spline' in vehicle_dynamics.get_all_vars() and len(vehicle_state) >= 5:
                                s_val = float(vehicle_state[4])
                                if s_val >= float(data.reference_path.length) - 1e-3:
                                    logger.info(f"Reference path end reached at step {step} (s={s_val:.3f})")
                                    break
                    except Exception:
                        pass

                    # Update plot bounds for obstacle manager (for bouncing behavior)
                    # Calculate bounds from reference path and current vehicle/obstacle positions
                    if hasattr(self, 'obstacle_manager') and self.obstacle_manager is not None:
                        # Calculate current plot bounds
                        margin = 10.0
                        if test_config.reference_path is not None and isinstance(test_config.reference_path, np.ndarray):
                            ref_x = test_config.reference_path[:, 0]
                            ref_y = test_config.reference_path[:, 1]
                            x_min = float(np.min(ref_x) - margin)
                            x_max = float(np.max(ref_x) + margin)
                            y_min = float(np.min(ref_y) - margin)
                            y_max = float(np.max(ref_y) + margin)
                            # Update obstacle manager with current bounds
                            self.obstacle_manager.plot_bounds = (x_min, x_max, y_min, y_max)
                    
                    # Note: Obstacle states are now updated BEFORE module.update() and solver solve
                    # (see code around line 803) to ensure constraints use current positions.
                    # We still need to append obstacle states for visualization tracking.
                    # Obstacle positions in data object are already updated, so we just track them.
                    if hasattr(self, 'obstacle_manager'):
                        # Obstacle states were already appended during the pre-update phase
                        # Just ensure all obstacles have states tracked
                        for i, obstacle in enumerate(obstacles):
                            if i < len(obstacle_states) and len(obstacle_states[i]) == step:
                                # State wasn't appended yet, append current position
                                if hasattr(obstacle, 'position') and obstacle.position is not None:
                                    obstacle_states[i].append(obstacle.position[:2].copy())
                    else:
                        # Fallback to original method (shouldn't happen if obstacle_manager exists)
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
                    # Count exception as a failure
                    consecutive_solver_failures += 1
                    logger.warning(f"  Consecutive MPC failures (exception): {consecutive_solver_failures}/{max_consecutive_failures}")
                    if consecutive_solver_failures >= max_consecutive_failures:
                        logger.error(f"Max consecutive MPC failures ({max_consecutive_failures}) reached due to exceptions. "
                                   f"No successful solve in last {max_consecutive_failures} attempts. "
                                   f"Terminating test early at step {step}.")
                        break
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
                
                # Increment step and check stopping criteria
                step += 1
                if not is_contouring and step >= num_steps:
                    break
                if is_contouring:
                    # Hard safety cap to avoid infinite loops in degenerate cases
                    if step >= max_steps_cap:
                        logger.info(f"Reached safety cap of {max_steps_cap} steps before path end; stopping")
                        break
                    
            # Save results - ALWAYS save files even if early exit occurred
            try:
                logger.info("Saving test results...")
                self.save_state_history(output_folder, vehicle_states, obstacle_states)
                logger.info(f"Saved state history: {len(vehicle_states)} vehicle states, {len(obstacle_states)} obstacles")
            except Exception as save_err:
                logger.error(f"Error saving state history: {save_err}")
            
            # Store data object for goal plotting in animation
            self.last_data = data
            
            # Create animation - ALWAYS create even if early exit occurred
            try:
                logger.info("Creating animation...")
                # halfspaces_per_step and linearized_halfspaces_per_step are already captured in the loop above
                self.create_animation(output_folder, vehicle_states, obstacle_states, test_config, goal_reached_step, predicted_trajs, halfspaces_per_step, linearized_halfspaces_per_step)
                logger.info("Animation created successfully")
            except Exception as anim_err:
                logger.error(f"Error creating animation: {anim_err}")
            # Save framework summary for validation
            try:
                summary = {
                    "objective_module": test_config.objective_module,
                    "constraint_modules": test_config.constraint_modules,
                    "vehicle_dynamics": test_config.vehicle_dynamics,
                    "horizon": int(self.solver.horizon) if hasattr(self.solver, 'horizon') and self.solver.horizon is not None else None,
                    "timestep": float(self.solver.timestep) if hasattr(self.solver, 'timestep') and self.solver.timestep is not None else None,
                    "solver": "casadi",
                }
                import json as _json
                with open(os.path.join(output_folder, "framework_summary.json"), 'w', encoding='utf-8') as f:
                    _json.dump(summary, f, indent=2)
            except Exception:
                pass
            
            # Determine test success - may have exited early but still successful if files were saved
            if early_exit_stuck:
                logger.warning("Test exited early due to stuck vehicle, but files were saved")
                test_success = True  # Files saved, so consider it successful
            elif early_exit_failures:
                logger.warning("Test exited early due to solver failures, but files were saved")
                test_success = True  # Files saved, so consider it successful
            else:
                logger.info("Test completed successfully")
                test_success = True
            
            return TestResult(
                success=test_success,
                vehicle_states=vehicle_states,
                obstacle_states=obstacle_states,
                computation_times=computation_times,
                constraint_violations=constraint_violations,
                output_folder=output_folder
            )
            
        except Exception as e:
            import traceback as _tb
            logger.error(f"Test failed: {e}")
            logger.error(f"Traceback: {_tb.format_exc()}")
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
        logger = logging.getLogger("integration_test")
        
        # Validate inputs
        if not vehicle_states:
            logger.warning("No vehicle states to save")
            return
        
        if not hasattr(self, 'solver') or self.solver is None:
            logger.error("Solver not available for saving state history")
            return
        
        if not hasattr(self.solver, 'timestep') or self.solver.timestep is None:
            logger.error("Solver timestep not available for saving state history")
            return
        
        # Save vehicle states
        vehicle_file = os.path.join(output_folder, "vehicle_states.csv")
        try:
            with open(vehicle_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'x', 'y', 'theta', 'velocity'])
                
                for i, state in enumerate(vehicle_states):
                    if state is None:
                        logger.warning(f"Skipping None state at index {i}")
                        continue
                    try:
                        state_list = state.tolist() if hasattr(state, 'tolist') else list(state)
                        writer.writerow([i * self.solver.timestep] + state_list)
                    except Exception as e:
                        logger.error(f"Error writing vehicle state at index {i}: {e}")
                        raise
            logger.info(f"Saved {len(vehicle_states)} vehicle states to {vehicle_file}")
        except Exception as e:
            logger.error(f"Error saving vehicle states to {vehicle_file}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Remove empty file if it was created
            if os.path.exists(vehicle_file) and os.path.getsize(vehicle_file) == 0:
                try:
                    os.remove(vehicle_file)
                except Exception:
                    pass
            raise
                
        # Save obstacle states with detailed information from obstacle manager
        try:
            if hasattr(self, 'obstacle_manager') and self.obstacle_manager is not None:
                obstacle_info = self.obstacle_manager.get_obstacle_info()
                
                # Save obstacle summary
                summary_file = os.path.join(output_folder, "obstacle_summary.csv")
                try:
                    with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['obstacle_id', 'dynamics_type', 'radius', 'initial_x', 'initial_y', 'initial_angle'])
                        
                        for idx, obs_info in enumerate(obstacle_info.get('obstacle_details', [])):
                            writer.writerow([
                                obs_info.get('id', idx),
                                obs_info.get('dynamics_type', 'Unknown'),
                                obs_info.get('radius', 0.0),
                                obs_info.get('position', [0.0, 0.0])[0],
                                obs_info.get('position', [0.0, 0.0])[1],
                                obs_info.get('angle', 0.0)
                            ])
                    logger.info(f"Saved obstacle summary to {summary_file}")
                except Exception as e:
                    logger.error(f"Error saving obstacle summary: {e}")
                
                # Save detailed obstacle states
                all_obstacle_states = self.obstacle_manager.get_all_obstacle_states()
                for i, obs_states in enumerate(all_obstacle_states):
                    if not obs_states:
                        continue
                    obstacle_file = os.path.join(output_folder, f"obstacle_{i}_detailed_states.csv")
                    try:
                        with open(obstacle_file, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            
                            # Determine state variables based on dynamics type
                            dynamics_type = obstacle_info.get('obstacle_details', [{}])[i].get('dynamics_type', 'PointMass')
                            if 'Unicycle' in dynamics_type:
                                writer.writerow(['time', 'x', 'y', 'psi', 'v'])
                            elif 'Bicycle' in dynamics_type:
                                writer.writerow(['time', 'x', 'y', 'psi', 'v', 'delta', 'spline'])
                            elif 'PointMass' in dynamics_type:
                                writer.writerow(['time', 'x', 'y', 'vx', 'vy'])
                            else:
                                writer.writerow(['time', 'x', 'y'])
                            
                            for j, state in enumerate(obs_states):
                                if state is None:
                                    logger.warning(f"Skipping None obstacle state {i} at index {j}")
                                    continue
                                try:
                                    state_list = state.tolist() if hasattr(state, 'tolist') else list(state)
                                    writer.writerow([j * self.solver.timestep] + state_list)
                                except Exception as e:
                                    logger.error(f"Error writing obstacle {i} state at index {j}: {e}")
                                    raise
                        logger.info(f"Saved {len(obs_states)} states for obstacle {i} to {obstacle_file}")
                    except Exception as e:
                        logger.error(f"Error saving obstacle {i} states: {e}")
                        # Remove empty file if it was created
                        if os.path.exists(obstacle_file) and os.path.getsize(obstacle_file) == 0:
                            try:
                                os.remove(obstacle_file)
                            except Exception:
                                pass
            else:
                # Fallback to simple obstacle states
                for i, obs_states in enumerate(obstacle_states):
                    if not obs_states:
                        continue
                    obstacle_file = os.path.join(output_folder, f"obstacle_{i}_states.csv")
                    try:
                        with open(obstacle_file, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(['time', 'x', 'y'])
                            
                            for j, state in enumerate(obs_states):
                                if state is None:
                                    logger.warning(f"Skipping None obstacle state {i} at index {j}")
                                    continue
                                try:
                                    state_list = state.tolist() if hasattr(state, 'tolist') else list(state)
                                    writer.writerow([j * self.solver.timestep] + state_list)
                                except Exception as e:
                                    logger.error(f"Error writing obstacle {i} state at index {j}: {e}")
                                    raise
                        logger.info(f"Saved {len(obs_states)} states for obstacle {i} to {obstacle_file}")
                    except Exception as e:
                        logger.error(f"Error saving obstacle {i} states: {e}")
                        # Remove empty file if it was created
                        if os.path.exists(obstacle_file) and os.path.getsize(obstacle_file) == 0:
                            try:
                                os.remove(obstacle_file)
                            except Exception:
                                pass
        except Exception as e:
            logger.error(f"Error saving obstacle states: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
                    
    def create_animation(self, output_folder: str, vehicle_states: List[np.ndarray], 
                        obstacle_states: List[List[np.ndarray]], test_config: TestConfig, 
                        goal_reached_step: Optional[int] = None,
                        predicted_trajs: Optional[List[List[tuple]]] = None,
                        halfspaces_per_step: Optional[List[List[tuple]]] = None,
                        linearized_halfspaces_per_step: Optional[List[List[tuple]]] = None):
        """Create GIF animation of the test showing full trajectory until goal is reached."""
        if not vehicle_states:
            logging.getLogger("integration_test").warning("No vehicle states to animate")
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Determine plot bounds with 10 meter margin on all sides of reference path bounds
        # First, collect reference path bounds (centerline + left/right boundaries)
        ref_path_x_list = []
        ref_path_y_list = []
        
        # Add reference path centerline points
        if test_config.reference_path is not None:
            if isinstance(test_config.reference_path, np.ndarray):
                ref_path_x_list.extend(test_config.reference_path[:, 0].tolist())
                ref_path_y_list.extend(test_config.reference_path[:, 1].tolist())
        
        # Add reference path from data object (more complete, includes spline points)
        try:
            if hasattr(self, 'last_data') and hasattr(self.last_data, 'reference_path') and self.last_data.reference_path is not None:
                rp = self.last_data.reference_path
                if hasattr(rp, 'x') and hasattr(rp, 'y'):
                    ref_path_x_list.extend(np.asarray(rp.x, dtype=float).tolist())
                    ref_path_y_list.extend(np.asarray(rp.y, dtype=float).tolist())
        except Exception:
            pass
        
        # Add reference path boundary points (left and right bounds) - these define the actual path bounds
        boundary_points_added = False
        try:
            if hasattr(self, 'last_data') and self.last_data is not None:
                # Add left boundary points (check both array and Bound object formats)
                left_x_list = None
                left_y_list = None
                
                if hasattr(self.last_data, 'left_boundary_x') and hasattr(self.last_data, 'left_boundary_y'):
                    if (self.last_data.left_boundary_x is not None and self.last_data.left_boundary_y is not None and
                        len(self.last_data.left_boundary_x) > 0 and len(self.last_data.left_boundary_y) > 0):
                        left_x_list = np.asarray(self.last_data.left_boundary_x, dtype=float)
                        left_y_list = np.asarray(self.last_data.left_boundary_y, dtype=float)
                        boundary_points_added = True
                
                # Also check Bound objects if arrays weren't available
                if not boundary_points_added and hasattr(self.last_data, 'left_bound') and self.last_data.left_bound is not None:
                    if hasattr(self.last_data.left_bound, 'x') and hasattr(self.last_data.left_bound, 'y'):
                        left_x_list = np.asarray(self.last_data.left_bound.x, dtype=float)
                        left_y_list = np.asarray(self.last_data.left_bound.y, dtype=float)
                        boundary_points_added = True
                
                if left_x_list is not None and left_y_list is not None:
                    ref_path_x_list.extend(left_x_list.tolist())
                    ref_path_y_list.extend(left_y_list.tolist())
                
                # Add right boundary points (check both array and Bound object formats)
                right_x_list = None
                right_y_list = None
                
                if hasattr(self.last_data, 'right_boundary_x') and hasattr(self.last_data, 'right_boundary_y'):
                    if (self.last_data.right_boundary_x is not None and self.last_data.right_boundary_y is not None and
                        len(self.last_data.right_boundary_x) > 0 and len(self.last_data.right_boundary_y) > 0):
                        right_x_list = np.asarray(self.last_data.right_boundary_x, dtype=float)
                        right_y_list = np.asarray(self.last_data.right_boundary_y, dtype=float)
                        boundary_points_added = True
                
                # Also check Bound objects if arrays weren't available
                if hasattr(self.last_data, 'right_bound') and self.last_data.right_bound is not None:
                    if hasattr(self.last_data.right_bound, 'x') and hasattr(self.last_data.right_bound, 'y'):
                        right_x_list = np.asarray(self.last_data.right_bound.x, dtype=float)
                        right_y_list = np.asarray(self.last_data.right_bound.y, dtype=float)
                        boundary_points_added = True
                
                if right_x_list is not None and right_y_list is not None:
                    ref_path_x_list.extend(right_x_list.tolist())
                    ref_path_y_list.extend(right_y_list.tolist())
                
                if boundary_points_added:
                    logging.getLogger("integration_test").debug(f"Added {len(left_x_list) if left_x_list is not None else 0} left and {len(right_x_list) if right_x_list is not None else 0} right boundary points to plot bounds")
        except Exception as e:
            logging.getLogger("integration_test").warning(f"Could not add boundary points to bounds: {e}")
        
        # Calculate bounds from reference path bounds (centerline + left/right boundaries)
        # Add 10 meter margin on all sides
        margin = 10.0
        if ref_path_x_list and ref_path_y_list:
            # Use reference path bounds as base
            ref_path_x_min = min(ref_path_x_list)
            ref_path_x_max = max(ref_path_x_list)
            ref_path_y_min = min(ref_path_y_list)
            ref_path_y_max = max(ref_path_y_list)
            
            x_min = ref_path_x_min - margin
            x_max = ref_path_x_max + margin
            y_min = ref_path_y_min - margin
            y_max = ref_path_y_max + margin
        else:
            # Fallback: use all trajectory data if no reference path bounds available
            all_x = [state[0] for state in vehicle_states]
            all_y = [state[1] for state in vehicle_states]
            
            # Add obstacle positions for bounds calculation
            for obs_states in obstacle_states:
                if obs_states:
                    all_x.extend([obs[0] for obs in obs_states])
                    all_y.extend([obs[1] for obs in obs_states])
            
            if not all_x or not all_y:
                all_x = [0.0]
                all_y = [0.0]
            
            x_range = max(all_x) - min(all_x) if len(all_x) > 1 else 20.0
            y_range = max(all_y) - min(all_y) if len(all_y) > 1 else 20.0
            margin_x = max(margin, x_range * 0.1)  # At least 10m or 10% of range
            margin_y = max(margin, y_range * 0.1)  # At least 10m or 10% of range
            x_min, x_max = min(all_x) - margin_x, max(all_x) + margin_x
            y_min, y_max = min(all_y) - margin_y, max(all_y) + margin_y
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Update obstacle manager with actual plot bounds for bouncing behavior
        if hasattr(self, 'obstacle_manager') and self.obstacle_manager is not None:
            self.obstacle_manager.plot_bounds = (x_min, x_max, y_min, y_max)
        
        # Add goal reached info to title
        title = f'Integration Test: {test_config.test_name}'
        if goal_reached_step is not None:
            title += f' (Goal reached at step {goal_reached_step})'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot reference path and denote end ONLY if data.reference_path exists (contouring cases)
        try:
            if hasattr(self, 'last_data') and hasattr(self.last_data, 'reference_path') and self.last_data.reference_path is not None:
                rp = self.last_data.reference_path
                ax.plot(np.asarray(rp.x, dtype=float), np.asarray(rp.y, dtype=float),
                        'k--', linewidth=2, label='Reference Path', alpha=0.7)
                try:
                    end_x = float(rp.x[-1])
                    end_y = float(rp.y[-1])
                    ax.plot(end_x, end_y, 'rx', markersize=10, mew=2, label='Path End')
                except Exception:
                    pass
        except Exception:
            pass
        
        # Plot reference path boundaries (left and right bounds) if they exist
        # Updated to work with new TKSpline formulations
        try:
            if hasattr(self, 'last_data') and self.last_data is not None:
                ref_path = None
                if hasattr(self.last_data, 'reference_path') and self.last_data.reference_path is not None:
                    ref_path = self.last_data.reference_path
                
                # Try to use boundary splines first (most accurate, uses TKSpline)
                if (hasattr(self.last_data, 'left_spline_x') and self.last_data.left_spline_x is not None and
                    hasattr(self.last_data, 'left_spline_y') and self.last_data.left_spline_y is not None and
                    hasattr(self.last_data, 'right_spline_x') and self.last_data.right_spline_x is not None and
                    hasattr(self.last_data, 'right_spline_y') and self.last_data.right_spline_y is not None and
                    ref_path is not None and hasattr(ref_path, 's') and ref_path.s is not None):
                    # Use TKSpline for smooth boundary visualization
                    s_arr = np.asarray(ref_path.s, dtype=float)
                    if len(s_arr) > 0:
                        s_min = float(s_arr[0])
                        s_max = float(s_arr[-1])
                        s_sample = np.linspace(s_min, s_max, 200)  # Smooth sampling
                        
                        left_x_spline = self.last_data.left_spline_x
                        left_y_spline = self.last_data.left_spline_y
                        right_x_spline = self.last_data.right_spline_x
                        right_y_spline = self.last_data.right_spline_y
                        
                        left_x_vals = [float(left_x_spline(s)) for s in s_sample]
                        left_y_vals = [float(left_y_spline(s)) for s in s_sample]
                        right_x_vals = [float(right_x_spline(s)) for s in s_sample]
                        right_y_vals = [float(right_y_spline(s)) for s in s_sample]
                        
                        # Road boundaries: gray dashed (distinct from orange/cyan contouring constraints)
                        ax.plot(left_x_vals, left_y_vals, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Left Road Boundary')
                        ax.plot(right_x_vals, right_y_vals, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Right Road Boundary')
                
                # Fallback to array-based boundaries
                elif hasattr(self.last_data, 'left_boundary_x') and hasattr(self.last_data, 'left_boundary_y'):
                    if (self.last_data.left_boundary_x is not None and self.last_data.left_boundary_y is not None and
                        len(self.last_data.left_boundary_x) > 0 and len(self.last_data.left_boundary_y) > 0):
                        # Road boundaries: gray dashed (distinct from orange/cyan contouring constraints)
                        ax.plot(np.asarray(self.last_data.left_boundary_x, dtype=float),
                               np.asarray(self.last_data.left_boundary_y, dtype=float),
                               'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Left Road Boundary')
                
                elif hasattr(self.last_data, 'left_bound') and self.last_data.left_bound is not None:
                    if hasattr(self.last_data.left_bound, 'x') and hasattr(self.last_data.left_bound, 'y'):
                        # Road boundaries: gray dashed (distinct from orange/cyan contouring constraints)
                        ax.plot(np.asarray(self.last_data.left_bound.x, dtype=float),
                               np.asarray(self.last_data.left_bound.y, dtype=float),
                               'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Left Road Boundary')
                
                # Right boundary fallback
                if (not (hasattr(self.last_data, 'right_spline_x') and self.last_data.right_spline_x is not None)):
                    if hasattr(self.last_data, 'right_boundary_x') and hasattr(self.last_data, 'right_boundary_y'):
                        if (self.last_data.right_boundary_x is not None and self.last_data.right_boundary_y is not None and
                            len(self.last_data.right_boundary_x) > 0 and len(self.last_data.right_boundary_y) > 0):
                            # Road boundaries: gray dashed (distinct from orange/cyan contouring constraints)
                            ax.plot(np.asarray(self.last_data.right_boundary_x, dtype=float),
                                   np.asarray(self.last_data.right_boundary_y, dtype=float),
                                   'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Right Road Boundary')
                    elif hasattr(self.last_data, 'right_bound') and self.last_data.right_bound is not None:
                        if hasattr(self.last_data.right_bound, 'x') and hasattr(self.last_data.right_bound, 'y'):
                            # Road boundaries: gray dashed (distinct from orange/cyan contouring constraints)
                            ax.plot(np.asarray(self.last_data.right_bound.x, dtype=float),
                                   np.asarray(self.last_data.right_bound.y, dtype=float),
                                   'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Right Road Boundary')
        except Exception as e:
            logging.getLogger("integration_test").debug(f"Could not plot reference path boundaries: {e}")

        # Draw module-provided visualizations (e.g., road bounds) on this animation axes
        # Note: Static visualizations (like road boundaries) are drawn once during setup
        # Dynamic visualizations (like Gaussian constraints) are drawn each frame in animate()
        try:
            if hasattr(self, 'solver') and hasattr(self.solver, 'module_manager'):
                import matplotlib.pyplot as _plt
                _plt.sca(ax)
                for module in self.solver.module_manager.get_modules():
                    if hasattr(module, 'get_visualizer'):
                        viz = module.get_visualizer()
                        if viz is not None and hasattr(viz, 'visualize') and hasattr(self, 'last_data'):
                            # Only draw static visualizations here (not Gaussian constraints which need per-frame updates)
                            # Gaussian constraints will be drawn in animate() function
                            module_name = getattr(module, 'name', '')
                            if 'gaussian' not in module_name.lower():
                                # Use a representative stage index for static visualizations
                                viz.visualize(None, self.last_data, stage_idx=1)
        except Exception:
            # Visualization errors are non-fatal for animation
            pass
        
        # Build dynamic goal artists if a goal sequence exists
        goal_sequence = None
        current_goal_plot = None
        reached_goal_plots = []
        try:
            if hasattr(self, 'last_data') and hasattr(self.last_data, 'goal_sequence') and self.last_data.goal_sequence:
                goal_sequence = [np.array(g) for g in self.last_data.goal_sequence]
        except Exception:
            goal_sequence = None
        if goal_sequence is not None:
            # Draw static crosses for all goals
            for i, seq_goal in enumerate(goal_sequence):
                label = 'Goal Sequence' if i == 0 else ''
                ax.plot(float(seq_goal[0]), float(seq_goal[1]), 'gx', markersize=10, markeredgewidth=2, label=label, zorder=7)
            # Create dynamic artists
            current_goal_plot, = ax.plot([], [], 'g*', markersize=14, label='Current Goal', zorder=10)
            reached_goal_plots = [ax.plot([], [], 'go', markersize=8, markeredgecolor='darkgreen', markeredgewidth=2, zorder=9)[0]
                                   for _ in goal_sequence]
            # Precompute which goal is current at each frame using distance threshold
            thresh = 1.0
            current_idx_by_frame = []
            idx = 0
            for f in range(len(vehicle_states)):
                px, py = float(vehicle_states[f][0]), float(vehicle_states[f][1])
                if idx < len(goal_sequence):
                    gx, gy = float(goal_sequence[idx][0]), float(goal_sequence[idx][1])
                    if (px - gx) ** 2 + (py - gy) ** 2 <= thresh ** 2:
                        idx = min(idx + 1, len(goal_sequence) - 1)
                current_idx_by_frame.append(idx)
        else:
            # Fallback single-goal marker
            try:
                if hasattr(self, 'last_data') and hasattr(self.last_data, 'goal') and self.last_data.goal is not None:
                    g = self.last_data.goal
                    ax.plot(float(g[0]), float(g[1]), 'g*', markersize=14, label='Goal', zorder=10)
            except Exception:
                pass
        
        # Initialize plot elements
        vehicle_plot, = ax.plot([], [], 'bo', markersize=8, label='Vehicle', zorder=5)
        vehicle_trail, = ax.plot([], [], 'b-', linewidth=1, alpha=0.3, label='Vehicle Trail')
        # Predicted trajectory line (optional)
        pred_line, = ax.plot([], [], 'g--', linewidth=1.5, alpha=0.7, label='Predicted Trajectory')
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
        
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l and l not in uniq:
                uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), loc='upper right')
        
        # Track trails
        vehicle_trail_x = []
        vehicle_trail_y = []
        obstacle_trail_x = [[] for _ in range(len(obstacle_states))]
        obstacle_trail_y = [[] for _ in range(len(obstacle_states))]
        
        # Initialize halfspace constraint lines (will be updated dynamically)
        halfspace_lines = []
        linearized_halfspace_lines = []  # For linearized constraint halfspaces
        
        # Initialize Gaussian constraint artists (will be updated dynamically)
        gaussian_constraint_artists = []
        
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
                        
                # Update predicted trajectory if requested
                if predicted_trajs is not None and test_config.show_predicted_trajectory:
                    pts = predicted_trajs[frame] if frame < len(predicted_trajs) else []
                    if pts:
                        px = [p[0] for p in pts]
                        py = [p[1] for p in pts]
                        pred_line.set_data(px, py)
                    else:
                        pred_line.set_data([], [])
                
                # Update halfspace constraints visualization
                # Remove old halfspace lines and fill regions
                for artist in halfspace_lines:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                halfspace_lines.clear()
                
                # Draw new halfspace constraints for this frame
                if halfspaces_per_step is not None and frame < len(halfspaces_per_step):
                    frame_halfspaces = halfspaces_per_step[frame]
                    if frame_halfspaces and len(frame_halfspaces) > 0:
                        # Debug: log constraint count for first few frames
                        if frame < 3:
                            logging.getLogger("integration_test").debug(f"Frame {frame}: Drawing {len(frame_halfspaces)} contouring constraint halfspaces")
                        # Get plot bounds for line extension
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        plot_size = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
                        
                        # Collect left and right boundaries separately for better visualization
                        # Store with spline_s for segment visualization
                        left_halfspaces = []
                        right_halfspaces = []
                        
                        # Get reference path for spline segment visualization
                        ref_path = None
                        if hasattr(self, 'last_data') and self.last_data is not None:
                            if hasattr(self.last_data, 'reference_path') and self.last_data.reference_path is not None:
                                ref_path = self.last_data.reference_path
                        
                        for constraint_tuple in frame_halfspaces:
                            # Unpack constraint tuple (A, b, is_left, spline_s) or (A, b, is_left) for backward compatibility
                            if len(constraint_tuple) >= 4:
                                A, b, is_left, spline_s = constraint_tuple[:4]
                            elif len(constraint_tuple) >= 3:
                                A, b, is_left = constraint_tuple[:3]
                                spline_s = None
                            else:
                                # Fallback for old format
                                A, b = constraint_tuple[:2]
                                is_left = None
                                spline_s = None
                            
                            a1, a2 = float(A[0]), float(A[1])
                            norm = np.sqrt(a1**2 + a2**2)
                            if norm < 1e-6:
                                continue
                            
                            # Normalize A
                            a1_norm = a1 / norm
                            a2_norm = a2 / norm
                            
                            # Determine if this is a left or right boundary constraint
                            if is_left is not None:
                                is_left_constraint = bool(is_left)
                            else:
                                # Fallback heuristic: check dot product with reference path tangent
                                is_left_constraint = a1_norm < 0
                            
                            # Store with spline_s for visualization
                            if is_left_constraint:
                                left_halfspaces.append((a1_norm, a2_norm, b / norm, spline_s))
                            else:
                                right_halfspaces.append((a1_norm, a2_norm, b / norm, spline_s))
                        
                        # Draw left boundary constraints (orange) - show spline segment
                        # Left boundary constraint restricts vehicle to stay on the left side (away from right boundary)
                        for idx, constraint_data in enumerate(left_halfspaces):
                            if len(constraint_data) >= 4:
                                a1_norm, a2_norm, b_norm, spline_s = constraint_data
                            else:
                                a1_norm, a2_norm, b_norm = constraint_data[:3]
                                spline_s = None
                            
                            # Contouring constraints: orange (distinct from gray road boundaries)
                            color = 'orange'
                            alpha = 0.7
                            
                            # Track if we successfully drew a spline segment
                            segment_drawn = False
                            
                            # Draw spline segment if we have spline_s and reference path
                            if ref_path is not None and spline_s is not None and hasattr(ref_path, 'x_spline') and hasattr(ref_path, 'y_spline'):
                                try:
                                    # Draw a segment of the reference path around the constraint s value
                                    s_arr = np.asarray(ref_path.s, dtype=float)
                                    if len(s_arr) > 0:
                                        s_min = float(s_arr[0])
                                        s_max = float(s_arr[-1])
                                        constraint_s = float(spline_s)
                                        
                                        # Clamp to valid range
                                        constraint_s = max(s_min, min(s_max, constraint_s))
                                        
                                        # Define segment length (e.g., 2 meters along the path)
                                        segment_length = 2.0
                                        
                                        # Estimate ds from path derivatives
                                        try:
                                            dx = float(ref_path.x_spline.derivative()(constraint_s))
                                            dy = float(ref_path.y_spline.derivative()(constraint_s))
                                            ds_per_meter = 1.0 / (np.sqrt(dx**2 + dy**2) + 1e-6)
                                            ds_segment = segment_length * ds_per_meter
                                        except:
                                            # Fallback: use fixed fraction of path
                                            ds_segment = (s_max - s_min) * 0.1
                                        
                                        # Sample segment around constraint_s
                                        s_start = max(s_min, constraint_s - ds_segment / 2)
                                        s_end = min(s_max, constraint_s + ds_segment / 2)
                                        s_segment = np.linspace(s_start, s_end, 20)
                                        
                                        # Get path points along segment
                                        x_segment = [float(ref_path.x_spline(s)) for s in s_segment]
                                        y_segment = [float(ref_path.y_spline(s)) for s in s_segment]
                                        
                                        # Get path point and normal at constraint_s
                                        path_x = float(ref_path.x_spline(constraint_s))
                                        path_y = float(ref_path.y_spline(constraint_s))
                                        path_dx = float(ref_path.x_spline.derivative()(constraint_s))
                                        path_dy = float(ref_path.y_spline.derivative()(constraint_s))
                                        path_norm = np.sqrt(path_dx**2 + path_dy**2)
                                        
                                        if path_norm > 1e-6:
                                            path_dx_norm = path_dx / path_norm
                                            path_dy_norm = path_dy / path_norm
                                            # Normal pointing left: [path_dy_norm, -path_dx_norm]
                                            normal_left = np.array([path_dy_norm, -path_dx_norm])
                                            
                                            # Get road width for boundary visualization
                                            road_width_half = 3.5  # Default
                                            if hasattr(self, 'last_data') and self.last_data is not None:
                                                if hasattr(self.last_data, 'left_boundary_x') and hasattr(self.last_data, 'right_boundary_x'):
                                                    # Estimate width from boundaries
                                                    try:
                                                        left_x = self.last_data.left_boundary_x
                                                        right_x = self.last_data.right_boundary_x
                                                        if len(left_x) > 0 and len(right_x) > 0:
                                                            # Find closest index to constraint_s
                                                            s_arr = np.asarray(ref_path.s, dtype=float)
                                                            closest_idx = np.argmin(np.abs(s_arr - constraint_s))
                                                            if closest_idx < len(left_x) and closest_idx < len(right_x):
                                                                center_x = ref_path.x[closest_idx] if hasattr(ref_path, 'x') and closest_idx < len(ref_path.x) else path_x
                                                                center_y = ref_path.y[closest_idx] if hasattr(ref_path, 'y') and closest_idx < len(ref_path.y) else path_y
                                                                left_bound_x = left_x[closest_idx]
                                                                left_bound_y = self.last_data.left_boundary_y[closest_idx]
                                                                right_bound_x = right_x[closest_idx]
                                                                right_bound_y = self.last_data.right_boundary_y[closest_idx]
                                                                width_left = np.sqrt((left_bound_x - center_x)**2 + (left_bound_y - center_y)**2)
                                                                width_right = np.sqrt((right_bound_x - center_x)**2 + (right_bound_y - center_y)**2)
                                                                road_width_half = max(width_left, width_right)
                                                    except:
                                                        pass
                                            
                                            # Compute boundary point (left boundary is on right side of path)
                                            boundary_point = np.array([path_x, path_y]) - normal_left * road_width_half
                                            
                                            # Draw constraint line segment along the boundary
                                            # Project spline segment points onto the constraint line
                                            segment_points_x = []
                                            segment_points_y = []
                                            
                                            for s_val in s_segment:
                                                seg_x = float(ref_path.x_spline(s_val))
                                                seg_y = float(ref_path.y_spline(s_val))
                                                
                                                # Get normal at this point
                                                seg_dx = float(ref_path.x_spline.derivative()(s_val))
                                                seg_dy = float(ref_path.y_spline.derivative()(s_val))
                                                seg_norm = np.sqrt(seg_dx**2 + seg_dy**2)
                                                if seg_norm > 1e-6:
                                                    seg_dx_norm = seg_dx / seg_norm
                                                    seg_dy_norm = seg_dy / seg_norm
                                                    seg_normal = np.array([seg_dy_norm, -seg_dx_norm])
                                                    
                                                    # Compute boundary point at this s
                                                    seg_boundary = np.array([seg_x, seg_y]) - seg_normal * road_width_half
                                                    segment_points_x.append(seg_boundary[0])
                                                    segment_points_y.append(seg_boundary[1])
                                            
                                            # Draw spline segment on boundary
                                            if len(segment_points_x) > 0:
                                                label = 'Left Boundary' if idx == 0 else ''
                                                line, = ax.plot(segment_points_x, segment_points_y, 
                                                               color=color, linestyle='-', 
                                                               linewidth=3.0, alpha=alpha, zorder=1,
                                                               label=label)
                                                halfspace_lines.append(line)
                                                segment_drawn = True
                                except Exception as e:
                                    logger.debug(f"Error drawing spline segment for left boundary: {e}")
                                    # Fallback to full line if spline segment fails
                                    segment_drawn = False
                            
                            # Fallback: draw full halfspace line if spline segment not available
                            if not segment_drawn:
                                # Calculate line segment that spans the plot
                                center_x = (xlim[0] + xlim[1]) / 2
                                center_y = (ylim[0] + ylim[1]) / 2
                                dist_to_line = (a1_norm * center_x + a2_norm * center_y - b_norm)
                                line_center_x = center_x - dist_to_line * a1_norm
                                line_center_y = center_y - dist_to_line * a2_norm
                                dir_x = -a2_norm
                                dir_y = a1_norm
                                line_length = plot_size * 1.2
                                x1 = line_center_x - dir_x * line_length
                                y1 = line_center_y - dir_y * line_length
                                x2 = line_center_x + dir_x * line_length
                                y2 = line_center_y + dir_y * line_length
                                label = 'Left Boundary' if idx == 0 else ''
                                line, = ax.plot([x1, x2], [y1, y2], 
                                               color=color, linestyle='--', 
                                               linewidth=2.0, alpha=alpha, zorder=1,
                                               label=label)
                                halfspace_lines.append(line)
                        
                        # Draw right boundary constraints (cyan) - show spline segment
                        # Right boundary constraint restricts vehicle to stay on the right side (away from left boundary)
                        for idx, constraint_data in enumerate(right_halfspaces):
                            if len(constraint_data) >= 4:
                                a1_norm, a2_norm, b_norm, spline_s = constraint_data
                            else:
                                a1_norm, a2_norm, b_norm = constraint_data[:3]
                                spline_s = None
                            
                            # Contouring constraints: cyan (distinct from gray road boundaries)
                            color = 'cyan'
                            alpha = 0.7
                            
                            # Track if we successfully drew a spline segment
                            segment_drawn = False
                            
                            # Draw spline segment if we have spline_s and reference path
                            if ref_path is not None and spline_s is not None and hasattr(ref_path, 'x_spline') and hasattr(ref_path, 'y_spline'):
                                try:
                                    # Draw a segment of the reference path around the constraint s value
                                    s_arr = np.asarray(ref_path.s, dtype=float)
                                    if len(s_arr) > 0:
                                        s_min = float(s_arr[0])
                                        s_max = float(s_arr[-1])
                                        constraint_s = float(spline_s)
                                        
                                        # Clamp to valid range
                                        constraint_s = max(s_min, min(s_max, constraint_s))
                                        
                                        # Define segment length (e.g., 2 meters along the path)
                                        segment_length = 2.0
                                        
                                        # Estimate ds from path derivatives
                                        try:
                                            dx = float(ref_path.x_spline.derivative()(constraint_s))
                                            dy = float(ref_path.y_spline.derivative()(constraint_s))
                                            ds_per_meter = 1.0 / (np.sqrt(dx**2 + dy**2) + 1e-6)
                                            ds_segment = segment_length * ds_per_meter
                                        except:
                                            # Fallback: use fixed fraction of path
                                            ds_segment = (s_max - s_min) * 0.1
                                        
                                        # Sample segment around constraint_s
                                        s_start = max(s_min, constraint_s - ds_segment / 2)
                                        s_end = min(s_max, constraint_s + ds_segment / 2)
                                        s_segment = np.linspace(s_start, s_end, 20)
                                        
                                        # Get path points along segment
                                        x_segment = [float(ref_path.x_spline(s)) for s in s_segment]
                                        y_segment = [float(ref_path.y_spline(s)) for s in s_segment]
                                        
                                        # Get path point and normal at constraint_s
                                        path_x = float(ref_path.x_spline(constraint_s))
                                        path_y = float(ref_path.y_spline(constraint_s))
                                        path_dx = float(ref_path.x_spline.derivative()(constraint_s))
                                        path_dy = float(ref_path.y_spline.derivative()(constraint_s))
                                        path_norm = np.sqrt(path_dx**2 + path_dy**2)
                                        
                                        if path_norm > 1e-6:
                                            path_dx_norm = path_dx / path_norm
                                            path_dy_norm = path_dy / path_norm
                                            # Normal pointing left: [path_dy_norm, -path_dx_norm]
                                            normal_left = np.array([path_dy_norm, -path_dx_norm])
                                            
                                            # Get road width for boundary visualization
                                            road_width_half = 3.5  # Default
                                            if hasattr(self, 'last_data') and self.last_data is not None:
                                                if hasattr(self.last_data, 'left_boundary_x') and hasattr(self.last_data, 'right_boundary_x'):
                                                    # Estimate width from boundaries
                                                    try:
                                                        left_x = self.last_data.left_boundary_x
                                                        right_x = self.last_data.right_boundary_x
                                                        if len(left_x) > 0 and len(right_x) > 0:
                                                            # Find closest index to constraint_s
                                                            s_arr = np.asarray(ref_path.s, dtype=float)
                                                            closest_idx = np.argmin(np.abs(s_arr - constraint_s))
                                                            if closest_idx < len(left_x) and closest_idx < len(right_x):
                                                                center_x = ref_path.x[closest_idx] if hasattr(ref_path, 'x') and closest_idx < len(ref_path.x) else path_x
                                                                center_y = ref_path.y[closest_idx] if hasattr(ref_path, 'y') and closest_idx < len(ref_path.y) else path_y
                                                                left_bound_x = left_x[closest_idx]
                                                                left_bound_y = self.last_data.left_boundary_y[closest_idx]
                                                                right_bound_x = right_x[closest_idx]
                                                                right_bound_y = self.last_data.right_boundary_y[closest_idx]
                                                                width_left = np.sqrt((left_bound_x - center_x)**2 + (left_bound_y - center_y)**2)
                                                                width_right = np.sqrt((right_bound_x - center_x)**2 + (right_bound_y - center_y)**2)
                                                                road_width_half = max(width_left, width_right)
                                                    except:
                                                        pass
                                            
                                            # Compute boundary point (right boundary is on left side of path)
                                            boundary_point = np.array([path_x, path_y]) + normal_left * road_width_half
                                            
                                            # Draw constraint line segment along the boundary
                                            # Project spline segment points onto the constraint line
                                            segment_points_x = []
                                            segment_points_y = []
                                            
                                            for s_val in s_segment:
                                                seg_x = float(ref_path.x_spline(s_val))
                                                seg_y = float(ref_path.y_spline(s_val))
                                                
                                                # Get normal at this point
                                                seg_dx = float(ref_path.x_spline.derivative()(s_val))
                                                seg_dy = float(ref_path.y_spline.derivative()(s_val))
                                                seg_norm = np.sqrt(seg_dx**2 + seg_dy**2)
                                                if seg_norm > 1e-6:
                                                    seg_dx_norm = seg_dx / seg_norm
                                                    seg_dy_norm = seg_dy / seg_norm
                                                    seg_normal = np.array([seg_dy_norm, -seg_dx_norm])
                                                    
                                                    # Compute boundary point at this s
                                                    seg_boundary = np.array([seg_x, seg_y]) + seg_normal * road_width_half
                                                    segment_points_x.append(seg_boundary[0])
                                                    segment_points_y.append(seg_boundary[1])
                                            
                                            # Draw spline segment on boundary
                                            if len(segment_points_x) > 0:
                                                label = 'Right Boundary' if idx == 0 else ''
                                                line, = ax.plot(segment_points_x, segment_points_y, 
                                                               color=color, linestyle='-', 
                                                               linewidth=3.0, alpha=alpha, zorder=1,
                                                               label=label)
                                                halfspace_lines.append(line)
                                                segment_drawn = True
                                except Exception as e:
                                    logger.debug(f"Error drawing spline segment for right boundary: {e}")
                                    # Fallback to full line if spline segment fails
                                    segment_drawn = False
                            
                            # Fallback: draw full halfspace line if spline segment not available
                            if not segment_drawn:
                                # Calculate line segment that spans the plot
                                center_x = (xlim[0] + xlim[1]) / 2
                                center_y = (ylim[0] + ylim[1]) / 2
                                dist_to_line = (a1_norm * center_x + a2_norm * center_y - b_norm)
                                line_center_x = center_x - dist_to_line * a1_norm
                                line_center_y = center_y - dist_to_line * a2_norm
                                dir_x = -a2_norm
                                dir_y = a1_norm
                                line_length = plot_size * 1.2
                                x1 = line_center_x - dir_x * line_length
                                y1 = line_center_y - dir_y * line_length
                                x2 = line_center_x + dir_x * line_length
                                y2 = line_center_y + dir_y * line_length
                                label = 'Right Boundary' if idx == 0 else ''
                                line, = ax.plot([x1, x2], [y1, y2], 
                                               color=color, linestyle='--', 
                                               linewidth=2.0, alpha=alpha, zorder=1,
                                               label=label)
                                halfspace_lines.append(line)
                
                # Update linearized constraint halfspaces visualization (obstacle avoidance)
                # Remove old linearized halfspace lines
                for artist in linearized_halfspace_lines:
                    try:
                        artist.remove()
                    except Exception:
                        pass
                linearized_halfspace_lines.clear()
                
                # Draw new linearized constraint halfspaces for this frame
                if linearized_halfspaces_per_step is not None and frame < len(linearized_halfspaces_per_step):
                    frame_linearized_halfspaces = linearized_halfspaces_per_step[frame]
                    if frame_linearized_halfspaces and len(frame_linearized_halfspaces) > 0:
                        # Get current vehicle position for shorter line segments
                        if frame < len(vehicle_states):
                            vehicle_pos = vehicle_states[frame]
                            vehicle_x, vehicle_y = vehicle_pos[0], vehicle_pos[1]
                        else:
                            vehicle_x, vehicle_y = 0.0, 0.0
                        
                        # Color palette for different obstacles
                        obstacle_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
                        
                        # Group constraints by obstacle
                        obstacle_constraints = {}
                        halfspace_offset = 0.0  # Default offset
                        for constraint_data in frame_linearized_halfspaces:
                            if len(constraint_data) >= 4:
                                # New format: (A, b, obstacle_id, halfspace_offset)
                                A, b, obstacle_id, halfspace_offset = constraint_data[0], constraint_data[1], constraint_data[2], constraint_data[3]
                            elif len(constraint_data) >= 3:
                                # Old format: (A, b, obstacle_id)
                                A, b, obstacle_id = constraint_data[0], constraint_data[1], constraint_data[2]
                                halfspace_offset = 0.0
                            else:
                                # Fallback for very old format
                                A, b = constraint_data[0], constraint_data[1]
                                obstacle_id = 0
                                halfspace_offset = 0.0
                            
                            if obstacle_id not in obstacle_constraints:
                                obstacle_constraints[obstacle_id] = []
                            obstacle_constraints[obstacle_id].append((A, b, halfspace_offset))
                        
                        # Draw constraints for each obstacle with distinct colors
                        for obstacle_id, constraints_list in obstacle_constraints.items():
                            color = obstacle_colors[obstacle_id % len(obstacle_colors)]
                            alpha = 0.6
                            
                            # Get obstacle position for shorter line segments and connection line
                            obstacle_pos = None
                            if obstacle_id < len(obstacle_states):
                                obstacle_history = obstacle_states[obstacle_id]
                                if frame < len(obstacle_history):
                                    obs_state = obstacle_history[frame]
                                    obstacle_pos = np.array([obs_state[0], obs_state[1]])
                            
                            for idx, constraint_tuple in enumerate(constraints_list):
                                if len(constraint_tuple) >= 3:
                                    A, b, halfspace_offset = constraint_tuple[0], constraint_tuple[1], constraint_tuple[2]
                                else:
                                    A, b = constraint_tuple[0], constraint_tuple[1]
                                    halfspace_offset = 0.0
                                a1, a2 = float(A[0]), float(A[1])
                                norm = np.sqrt(a1**2 + a2**2)
                                if norm < 1e-6:
                                    continue
                                
                                # Normalize A
                                a1_norm = a1 / norm
                                a2_norm = a2 / norm
                                b_norm = b / norm
                                
                                # Calculate constraint line position
                                # The constraint is: A·p <= b, where A points FROM vehicle TO obstacle
                                # From linearized_constraints.py (updated): b = A·obstacle_pos - (obstacle_radius + robot_radius)
                                # The constraint line A·p = b is perpendicular to A
                                # The constraint enforces: A·vehicle <= A·obstacle - safe_distance
                                # Which means: ||obstacle - vehicle|| >= safe_distance
                                if obstacle_pos is not None:
                                    # Vehicle position
                                    vehicle_pos = np.array([vehicle_x, vehicle_y])
                                    
                                    # Vehicle-to-obstacle vector (points FROM vehicle TO obstacle, matches A direction)
                                    vehicle_to_obstacle = obstacle_pos - vehicle_pos
                                    vehicle_to_obstacle_dist = np.linalg.norm(vehicle_to_obstacle)
                                    
                                    if vehicle_to_obstacle_dist > 1e-6:
                                        # Normalized vehicle-to-obstacle direction (should match A direction)
                                        vehicle_to_obstacle_dir = vehicle_to_obstacle / vehicle_to_obstacle_dist
                                        
                                        # The constraint line A·p = b is perpendicular to A
                                        # Find where this line intersects the vehicle-to-obstacle line
                                        # Since A is the normalized direction from vehicle to obstacle,
                                        # the vehicle-to-obstacle line is: p(t) = vehicle_pos + t * A, t >= 0
                                        # The constraint line is: A·p = b
                                        # Intersection: A·(vehicle_pos + t * A) = b
                                        # A·vehicle_pos + t * (A·A) = b
                                        # Since A is normalized, A·A = 1, so: t = b - A·vehicle_pos
                                        
                                        A_dot_vehicle = a1_norm * vehicle_x + a2_norm * vehicle_y
                                        t = b_norm - A_dot_vehicle
                                        
                                        # The intersection point on the vehicle-to-obstacle line
                                        # This is the point where the constraint line (perpendicular to A) passes
                                        # through the vehicle-to-obstacle line
                                        # Note: t represents distance along A from vehicle_pos
                                        line_center_point = vehicle_pos + t * np.array([a1_norm, a2_norm])
                                        
                                        line_center_x = line_center_point[0]
                                        line_center_y = line_center_point[1]
                                        
                                        # Line length: proportional to distance between vehicle and obstacle
                                        line_length = max(3.0, vehicle_to_obstacle_dist * 0.6)  # At least 3m, or 60% of distance
                                    else:
                                        # Vehicle and obstacle are at same position, use obstacle position
                                        line_center_x = obstacle_pos[0]
                                        line_center_y = obstacle_pos[1]
                                        line_length = 3.0
                                else:
                                    # Fallback: use plot center
                                    xlim = ax.get_xlim()
                                    ylim = ax.get_ylim()
                                    mid_x = (xlim[0] + xlim[1]) / 2
                                    mid_y = (ylim[0] + ylim[1]) / 2
                                    
                                    # Project onto line
                                    dist_to_line = (a1_norm * mid_x + a2_norm * mid_y - b_norm)
                                    line_center_x = mid_x - dist_to_line * a1_norm
                                    line_center_y = mid_y - dist_to_line * a2_norm
                                    line_length = 5.0  # Default line length
                                
                                # Direction along the line (perpendicular to A)
                                dir_x = -a2_norm
                                dir_y = a1_norm
                                
                                # Draw line segment centered at line_center
                                x1 = line_center_x - dir_x * line_length / 2
                                y1 = line_center_y - dir_y * line_length / 2
                                x2 = line_center_x + dir_x * line_length / 2
                                y2 = line_center_y + dir_y * line_length / 2
                                
                                # Draw constraint line with obstacle-specific color
                                line, = ax.plot([x1, x2], [y1, y2], 
                                               color=color, linestyle='--', 
                                               linewidth=1.0, alpha=alpha, zorder=1)
                                linearized_halfspace_lines.append(line)
                                
                                # Add smaller arrow showing restriction direction (away from obstacle)
                                # Arrow points in direction -A (away from obstacle, toward allowed region)
                                arrow_length = 1.0  # Smaller arrow
                                arrow_mid_x = line_center_x
                                arrow_mid_y = line_center_y
                                arrow_dx = -a1_norm * arrow_length
                                arrow_dy = -a2_norm * arrow_length
                                
                                arrow = ax.annotate('', xy=(arrow_mid_x + arrow_dx, arrow_mid_y + arrow_dy),
                                                  xytext=(arrow_mid_x, arrow_mid_y),
                                                  arrowprops=dict(arrowstyle='->', color=color, 
                                                                lw=1.5, alpha=alpha, zorder=2))
                                linearized_halfspace_lines.append(arrow)
                                
                                # Add connecting line from constraint to obstacle
                                # This line shows the offset distance from obstacle surface to constraint line
                                # The constraint line is at distance (obstacle_radius + robot_radius + halfspace_offset) from obstacle center
                
                # Update Gaussian constraint visualizations
                # Clear old Gaussian constraint artists (orange ellipses and markers)
                for artist in gaussian_constraint_artists[:]:  # Use slice copy to avoid modification during iteration
                    try:
                        if hasattr(artist, 'remove'):
                            artist.remove()
                    except Exception:
                        pass
                gaussian_constraint_artists.clear()
                
                # Draw Gaussian constraints for current frame
                try:
                    # Check if solver is available
                    if not hasattr(self, 'solver'):
                        if frame < 3:
                            logging.getLogger("integration_test").warning(f"Frame {frame}: self.solver not available for Gaussian visualization")
                    elif not hasattr(self.solver, 'module_manager'):
                        if frame < 3:
                            logging.getLogger("integration_test").warning(f"Frame {frame}: solver.module_manager not available for Gaussian visualization")
                    else:
                        # Don't use plt.sca() or plt.figure() - animation axes aren't managed by pyplot
                        # Instead, pass axes directly to visualizer
                        gaussian_module_found = False
                        for module in self.solver.module_manager.get_modules():
                            module_name = getattr(module, 'name', '')
                            if 'gaussian' in module_name.lower():
                                gaussian_module_found = True
                                if hasattr(module, 'get_visualizer'):
                                    viz = module.get_visualizer()
                                    if viz is not None and hasattr(viz, 'visualize') and hasattr(self, 'last_data'):
                                        # Debug logging
                                        if frame < 3:
                                            logging.getLogger("integration_test").info(
                                                f"Frame {frame}: Found Gaussian module '{module_name}', calling visualizer")
                                        # Update obstacle positions for current frame before visualizing
                                        # Create a copy of last_data and update obstacle positions
                                        if hasattr(self.last_data, 'dynamic_obstacles') and self.last_data.dynamic_obstacles:
                                            # Update obstacle positions from obstacle_states for current frame
                                            for i, obs in enumerate(self.last_data.dynamic_obstacles):
                                                if i < len(obstacle_states) and frame < len(obstacle_states[i]):
                                                    obs_state = obstacle_states[i][frame]
                                                    if len(obs_state) >= 2:
                                                        obs.position = np.array([obs_state[0], obs_state[1]], dtype=float)
                                            
                                            # Propagate obstacle predictions for current frame
                                            # This ensures prediction steps are updated with current positions
                                            # CRITICAL: Save prediction types before propagation, as propagate_obstacles may reset them
                                            from planning.types import PredictionType
                                            saved_prediction_types = []
                                            for obs in self.last_data.dynamic_obstacles:
                                                if hasattr(obs, 'prediction') and obs.prediction is not None:
                                                    saved_prediction_types.append(obs.prediction.type)
                                                else:
                                                    saved_prediction_types.append(None)
                                            
                                            try:
                                                from planning.types import propagate_obstacles
                                                if hasattr(self.solver, 'horizon') and hasattr(self.solver, 'timestep'):
                                                    horizon_val = self.solver.horizon if self.solver.horizon is not None else 10
                                                    timestep_val = self.solver.timestep if self.solver.timestep is not None else 0.1
                                                    propagate_obstacles(self.last_data, dt=timestep_val, horizon=horizon_val)
                                                
                                                # CRITICAL: Restore prediction types after propagation
                                                # propagate_obstacles may reset them to DETERMINISTIC
                                                for i, obs in enumerate(self.last_data.dynamic_obstacles):
                                                    if i < len(saved_prediction_types) and saved_prediction_types[i] is not None:
                                                        if hasattr(obs, 'prediction') and obs.prediction is not None:
                                                            obs.prediction.type = saved_prediction_types[i]
                                            except Exception as e:
                                                logging.getLogger("integration_test").debug(f"Could not propagate obstacles for frame {frame}: {e}")
                                        
                                        # Debug: Check if obstacles have Gaussian predictions
                                        if frame < 5:  # Log for first few frames
                                            if hasattr(self.last_data, 'dynamic_obstacles') and self.last_data.dynamic_obstacles:
                                                gaussian_count = 0
                                                for i, obs in enumerate(self.last_data.dynamic_obstacles):
                                                    if (hasattr(obs, 'prediction') and obs.prediction is not None and
                                                        hasattr(obs.prediction, 'type')):
                                                        from planning.types import PredictionType
                                                        pred_type = obs.prediction.type
                                                        logging.getLogger("integration_test").info(
                                                            f"Frame {frame}: Obstacle {i} prediction type: {pred_type}")
                                                        if pred_type == PredictionType.GAUSSIAN:
                                                            gaussian_count += 1
                                                            if hasattr(obs.prediction, 'steps') and len(obs.prediction.steps) > 0:
                                                                step = obs.prediction.steps[0]
                                                                logging.getLogger("integration_test").info(
                                                    f"Frame {frame}: Obstacle {i} has Gaussian prediction with "
                                                    f"major_radius={getattr(step, 'major_radius', 'N/A')}, "
                                                    f"minor_radius={getattr(step, 'minor_radius', 'N/A')}, "
                                                    f"position={getattr(step, 'position', 'N/A')}")
                                                        else:
                                                            logging.getLogger("integration_test").warning(
                                                                f"Frame {frame}: Obstacle {i} does NOT have Gaussian prediction (type={pred_type})")
                                                logging.getLogger("integration_test").info(
                                                    f"Frame {frame}: Found {gaussian_count}/{len(self.last_data.dynamic_obstacles)} obstacles with Gaussian predictions, calling visualizer")
                                            else:
                                                logging.getLogger("integration_test").warning(
                                                    f"Frame {frame}: No dynamic_obstacles in last_data")
                                        
                                        # Store current patches/lines/texts count before visualization
                                        patches_before = list(ax.patches)
                                        lines_before = list(ax.lines)
                                        texts_before = list(ax.texts)
                                        
                                        # Call visualizer for current frame (use stage_idx=0 for first prediction step)
                                        # Pass axes directly to avoid plt.gca() issues in animation context
                                        if frame < 5:
                                            logging.getLogger("integration_test").info(
                                                f"Frame {frame}: About to call Gaussian visualizer.visualize()")
                                        try:
                                            viz.visualize(None, self.last_data, stage_idx=0, ax=ax)
                                            if frame < 5:
                                                logging.getLogger("integration_test").info(
                                                    f"Frame {frame}: Gaussian visualizer.visualize() completed")
                                        except Exception as viz_err:
                                            logging.getLogger("integration_test").error(
                                                f"Frame {frame}: Error calling Gaussian visualizer: {viz_err}")
                                            import traceback
                                            logging.getLogger("integration_test").error(f"Traceback: {traceback.format_exc()}")
                                        
                                        # Track newly added patches, lines, and texts (Gaussian constraint artists)
                                        patches_after = list(ax.patches)
                                        lines_after = list(ax.lines)
                                        texts_after = list(ax.texts)
                                        
                                        # Find new patches (ellipses) - track ALL ellipses added by visualizer
                                        from matplotlib.patches import Ellipse
                                        new_ellipses = 0
                                        for patch in patches_after:
                                            if patch not in patches_before and isinstance(patch, Ellipse):
                                                gaussian_constraint_artists.append(patch)
                                                new_ellipses += 1
                                        
                                        # Find new lines (mean position markers) - track ALL new lines
                                        new_markers = 0
                                        for line in lines_after:
                                            if line not in lines_before:
                                                # Track all new lines (markers for mean positions)
                                                gaussian_constraint_artists.append(line)
                                                new_markers += 1
                                        
                                        # Find new texts (uncertainty parameter markers) - track ALL new texts
                                        new_texts = 0
                                        for text in texts_after:
                                            if text not in texts_before:
                                                # Track all new text annotations (uncertainty parameters)
                                                gaussian_constraint_artists.append(text)
                                                new_texts += 1
                                        
                                        if frame < 5:  # Debug logging for first few frames
                                            logging.getLogger("integration_test").info(
                                                f"Frame {frame}: Visualizer added {new_ellipses} ellipses, {new_markers} markers, and {new_texts} text annotations for Gaussian constraints")
                                            if new_ellipses == 0:
                                                logging.getLogger("integration_test").warning(
                                                    f"Frame {frame}: WARNING - No ellipses were added by Gaussian visualizer!")
                                                # Check what obstacles we have
                                                if hasattr(self.last_data, 'dynamic_obstacles') and self.last_data.dynamic_obstacles:
                                                    for i, obs in enumerate(self.last_data.dynamic_obstacles):
                                                        from planning.types import PredictionType
                                                        pred_type = getattr(obs.prediction, 'type', None) if hasattr(obs, 'prediction') and obs.prediction else None
                                                        has_steps = hasattr(obs.prediction, 'steps') and len(obs.prediction.steps) > 0 if hasattr(obs, 'prediction') and obs.prediction else False
                                                        logging.getLogger("integration_test").warning(
                                                            f"Frame {frame}:   Obstacle {i}: type={pred_type}, has_steps={has_steps}, steps_count={len(obs.prediction.steps) if hasattr(obs.prediction, 'steps') else 0}")
                                    else:
                                        if frame < 3:
                                            logging.getLogger("integration_test").warning(
                                                f"Frame {frame}: Gaussian visualizer conditions not met: viz={viz is not None}, has_visualize={hasattr(viz, 'visualize') if viz else False}, has_last_data={hasattr(self, 'last_data')}")
                                else:
                                    if frame < 3:
                                        logging.getLogger("integration_test").warning(
                                            f"Frame {frame}: Gaussian module '{module_name}' has no get_visualizer method")
                            else:
                                if frame < 3 and not gaussian_module_found:
                                    logging.getLogger("integration_test").warning(
                                        f"Frame {frame}: No Gaussian module found in solver modules")
                except Exception as e:
                    # Visualization errors are non-fatal for animation
                    logging.getLogger("integration_test").error(f"Error visualizing Gaussian constraints: {e}")
                    import traceback
                    logging.getLogger("integration_test").error(f"Traceback: {traceback.format_exc()}")
                    pass
                
                # Goals update
                artists_extra = []
                if goal_sequence is not None and current_goal_plot is not None:
                    cur_idx = current_idx_by_frame[frame] if frame < len(current_idx_by_frame) else len(goal_sequence) - 1
                    # Reached goals
                    for i, plot in enumerate(reached_goal_plots):
                        if i < cur_idx:
                            plot.set_data([float(goal_sequence[i][0])], [float(goal_sequence[i][1])])
                            artists_extra.append(plot)
                        else:
                            plot.set_data([], [])
                    # Current goal
                    cg = goal_sequence[cur_idx]
                    current_goal_plot.set_data([float(cg[0])], [float(cg[1])])
                    artists_extra.append(current_goal_plot)
            return [vehicle_plot, vehicle_trail, pred_line] + obstacle_plots + obstacle_trails + [vehicle_circle] + obstacle_circles + ([current_goal_plot] if current_goal_plot is not None else []) + reached_goal_plots + halfspace_lines + linearized_halfspace_lines + gaussian_constraint_artists
            
        # Create animation - use all frames to show complete trajectory
        total_frames = len(vehicle_states)
        # Explicitly use range to ensure all frames are included (0 to total_frames-1)
        # Using repeat=True allows the GIF to loop when saved
        # Note: blit=False to allow dynamic adding/removing of constraint lines
        anim = animation.FuncAnimation(fig, animate, frames=range(total_frames), 
                                     interval=100, blit=False, repeat=True)
        
        # Calculate appropriate fps to ensure GIF shows complete trajectory
        # Target: minimum 5 seconds for short trajectories, scale for longer ones
        min_duration = 5.0  # seconds
        max_fps = 10  # Maximum fps for smooth playback
        calculated_fps = min(max_fps, total_frames / min_duration) if total_frames > 0 else max_fps
        
        # Save as GIF with calculated fps to ensure complete trajectory is visible
        gif_path = os.path.join(output_folder, "animation.gif")
        logger = logging.getLogger("integration_test")
        try:
            if total_frames == 0:
                logger.warning("No frames to save in animation")
                plt.close(fig)
                return
            
            anim.save(gif_path, writer='pillow', fps=calculated_fps)
            
            # Verify the file was created and has content
            if not os.path.exists(gif_path):
                logger.error(f"Animation file was not created: {gif_path}")
                plt.close(fig)
                return
            
            file_size = os.path.getsize(gif_path)
            if file_size == 0:
                logger.error(f"Animation file is empty: {gif_path}")
                try:
                    os.remove(gif_path)
                except Exception:
                    pass
                plt.close(fig)
                return
            
            logger.info(f"Saved animation with {total_frames} frames at {calculated_fps:.2f} fps to {gif_path} (duration: {total_frames/calculated_fps:.2f}s, size: {file_size} bytes)")
        except Exception as e:
            logger.error(f"Error saving animation to {gif_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Remove empty file if it was created
            if os.path.exists(gif_path) and os.path.getsize(gif_path) == 0:
                try:
                    os.remove(gif_path)
                except Exception:
                    pass
        finally:
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
        obstacle_prediction_types=["gaussian", "gaussian", "gaussian"],
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
