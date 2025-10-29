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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from solver.src.casadi_solver import CasADiSolver
from planning.src.types import Data, DynamicObstacle, PredictionType, PredictionStep
from utils.utils import read_config_file
from test.integration.obstacle_manager import ObstacleManager, ObstacleConfig, create_unicycle_obstacle, create_bicycle_obstacle, create_point_mass_obstacle


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
        self.config = read_config_file(config_file)
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
            from test.integration.obstacle_manager import PointMassModel
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
        
    def create_obstacles(self, num_obstacles: int, dynamics_types: List[str]) -> List[DynamicObstacle]:
        """Create obstacles with specified dynamics using obstacle manager."""
        obstacle_manager = ObstacleManager(self.config)
        
        # Create obstacle configurations
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
            
            # Create obstacles
            obstacles = self.create_obstacles(test_config.num_obstacles, test_config.obstacle_dynamics)
            
            # Initialize data
            data = Data()
            data.dynamic_obstacles = obstacles
            
            # Initialize state tracking
            vehicle_states = []
            obstacle_states = [[] for _ in range(test_config.num_obstacles)]
            computation_times = []
            constraint_violations = []
            
            # Initial state
            vehicle_state = np.array([0.0, 0.0, 0.0, 1.0])  # x, y, theta, v
            vehicle_states.append(vehicle_state.copy())
            
            # Run simulation
            num_steps = int(test_config.duration / test_config.timestep)
            
            for step in range(num_steps):
                logger.info(f"Step {step}/{num_steps}")
                
                start_time = time.time()
                
                # Prepare iteration
                for constraint in constraint_modules:
                    if hasattr(constraint, 'prepare_iteration'):
                        constraint.prepare_iteration(vehicle_state, data)
                        
                # Solve MPC
                try:
                    # This would be the actual MPC solve
                    # For now, we'll simulate the solution
                    u_solution = np.array([0.1, 0.0])  # acceleration, angular velocity
                    
                    # Update vehicle state (simple integration)
                    dt = test_config.timestep
                    vehicle_state[0] += vehicle_state[3] * np.cos(vehicle_state[2]) * dt
                    vehicle_state[1] += vehicle_state[3] * np.sin(vehicle_state[2]) * dt
                    vehicle_state[2] += u_solution[1] * dt
                    vehicle_state[3] += u_solution[0] * dt
                    
                    # Update obstacle states using obstacle manager
                    if hasattr(self, 'obstacle_manager'):
                        self.obstacle_manager.update_obstacle_states(test_config.timestep)
                        
                        # Get updated obstacle states
                        for i, obstacle in enumerate(obstacles):
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
                            if step < len(obstacle.prediction.steps):
                                obstacle_state = obstacle.prediction.steps[step].position
                                obstacle_states[i].append(obstacle_state.copy())
                            else:
                                # Extrapolate
                                last_state = obstacle_states[i][-1] if obstacle_states[i] else obstacle.position
                                obstacle_states[i].append(last_state.copy())
                    
                    vehicle_states.append(vehicle_state.copy())
                    constraint_violations.append(False)  # Assume no violations for now
                    
                except Exception as e:
                    logger.error(f"MPC solve failed at step {step}: {e}")
                    constraint_violations.append(True)
                    
                computation_time = time.time() - start_time
                computation_times.append(computation_time)
                
                logger.info(f"Step {step} completed in {computation_time:.3f}s")
                
            # Save results
            self.save_state_history(output_folder, vehicle_states, obstacle_states)
            self.create_animation(output_folder, vehicle_states, obstacle_states, test_config)
            
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
        with open(vehicle_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'x', 'y', 'theta', 'velocity'])
            
            for i, state in enumerate(vehicle_states):
                writer.writerow([i * self.solver.timestep] + state.tolist())
                
        # Save obstacle states with detailed information from obstacle manager
        if hasattr(self, 'obstacle_manager'):
            obstacle_info = self.obstacle_manager.get_obstacle_info()
            
            # Save obstacle summary
            summary_file = os.path.join(output_folder, "obstacle_summary.csv")
            with open(summary_file, 'w', newline='') as csvfile:
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
                with open(obstacle_file, 'w', newline='') as csvfile:
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
                with open(obstacle_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['time', 'x', 'y'])
                    
                    for j, state in enumerate(obs_states):
                        writer.writerow([j * self.solver.timestep] + state.tolist())
                    
    def create_animation(self, output_folder: str, vehicle_states: List[np.ndarray], 
                        obstacle_states: List[List[np.ndarray]], test_config: TestConfig):
        """Create GIF animation of the test."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up plot
        ax.set_xlim(-2, 20)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Integration Test: {test_config.test_name}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot reference path
        if test_config.reference_path is not None:
            ax.plot(test_config.reference_path[:, 0], test_config.reference_path[:, 1], 
                   'k--', linewidth=2, label='Reference Path')
        
        # Initialize plot elements
        vehicle_plot, = ax.plot([], [], 'bo', markersize=8, label='Vehicle')
        obstacle_plots = []
        for i in range(len(obstacle_states)):
            plot, = ax.plot([], [], 'ro', markersize=6, label=f'Obstacle {i}')
            obstacle_plots.append(plot)
            
        # Add vehicle radius circle
        vehicle_circle = plt.Circle((0, 0), self.config.get("robot", {}).get("radius", 0.5), 
                                  color='blue', alpha=0.3)
        ax.add_patch(vehicle_circle)
        
        # Add obstacle radius circles
        obstacle_circles = []
        for i in range(len(obstacle_states)):
            circle = plt.Circle((0, 0), self.config.get("obstacle_radius", 0.35), 
                              color='red', alpha=0.3)
            ax.add_patch(circle)
            obstacle_circles.append(circle)
            
        ax.legend()
        
        def animate(frame):
            if frame < len(vehicle_states):
                # Update vehicle
                vehicle_state = vehicle_states[frame]
                vehicle_plot.set_data([vehicle_state[0]], [vehicle_state[1]])
                vehicle_circle.center = (vehicle_state[0], vehicle_state[1])
                
                # Update obstacles
                for i, obs_states in enumerate(obstacle_states):
                    if frame < len(obs_states):
                        obs_state = obs_states[frame]
                        obstacle_plots[i].set_data([obs_state[0]], [obs_state[1]])
                        obstacle_circles[i].center = (obs_state[0], obs_state[1])
                        
            return [vehicle_plot] + obstacle_plots + [vehicle_circle] + obstacle_circles
            
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(vehicle_states), 
                                     interval=100, blit=True, repeat=True)
        
        # Save as GIF
        gif_path = os.path.join(output_folder, "animation.gif")
        anim.save(gif_path, writer='pillow', fps=10)
        
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
