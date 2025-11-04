"""
Scenario sampling utilities for safe horizon constraints.
"""
import numpy as np
from typing import List, Tuple
from planning.types import DynamicObstacle, PredictionType, Scenario
from utils.utils import LOG_DEBUG, LOG_WARN


class ScenarioSampler:
    """Handles sampling of scenarios from obstacle predictions."""
    
    def __init__(self, num_scenarios: int = 100, enable_outlier_removal: bool = True):
        self.num_scenarios = num_scenarios
        self.enable_outlier_removal = enable_outlier_removal
        self.scenarios = []
        
    def sample_scenarios(self, obstacles: List[DynamicObstacle], 
                        horizon_length: int, timestep: float) -> List[Scenario]:
        """
        Sample scenarios from obstacle predictions.
        
        Args:
            obstacles: List of dynamic obstacles
            horizon_length: MPC horizon length
            timestep: Time step size
            
        Returns:
            List of sampled scenarios
        """
        scenarios = []
        
        for obstacle_idx, obstacle in enumerate(obstacles):
            if obstacle.prediction.type != PredictionType.GAUSSIAN:
                LOG_WARN(f"Obstacle {obstacle_idx} has non-Gaussian prediction, skipping")
                continue
                
            # Sample scenarios for this obstacle
            obstacle_scenarios = self._sample_obstacle_scenarios(
                obstacle, obstacle_idx, horizon_length, timestep
            )
            scenarios.extend(obstacle_scenarios)
        
        self.scenarios = scenarios
        
        # Optional outlier removal
        if self.enable_outlier_removal:
            scenarios = self._remove_outliers(scenarios)
            
        LOG_DEBUG(f"Sampled {len(scenarios)} scenarios from {len(obstacles)} obstacles")
        return scenarios
    
    def _sample_obstacle_scenarios(self, obstacle: DynamicObstacle, obstacle_idx: int,
                                 horizon_length: int, _timestep: float) -> List[Scenario]:
        """Sample scenarios for a single obstacle."""
        scenarios = []
        
        # For Gaussian predictions, sample from multivariate normal
        if obstacle.prediction.type == PredictionType.GAUSSIAN:
            scenarios = self._sample_gaussian_scenarios(
                obstacle, obstacle_idx, horizon_length, _timestep
            )
        
        return scenarios
    
    def _sample_gaussian_scenarios(self, obstacle: DynamicObstacle, obstacle_idx: int,
                                 horizon_length: int, timestep: float) -> List[Scenario]:
        """Sample scenarios from Gaussian prediction."""
        scenarios = []
        
        # Get prediction steps
        prediction_steps = obstacle.prediction.steps
        if not prediction_steps:
            LOG_WARN(f"No prediction steps for obstacle {obstacle_idx}")
            return scenarios
        
        # Sample scenarios for each time step
        for step in range(min(horizon_length, len(prediction_steps))):
            pred_step = prediction_steps[step]
            
            # Create covariance matrix from major/minor radii
            # Assuming circular uncertainty for simplicity
            sigma = max(pred_step.major_radius, pred_step.minor_radius)
            cov_matrix = np.array([[sigma**2, 0], [0, sigma**2]])
            
            # Sample positions
            mean_pos = pred_step.position[:2]  # Use only x, y
            sampled_positions = np.random.multivariate_normal(
                mean_pos, cov_matrix, self.num_scenarios
            )
            
            # Create scenarios
            for scenario_idx, pos in enumerate(sampled_positions):
                scenario = Scenario(scenario_idx, obstacle_idx)
                scenario.position = pos
                scenario.time_step = step
                scenario.radius = obstacle.radius
                scenarios.append(scenario)
        
        return scenarios
    
    def _remove_outliers(self, scenarios: List[Scenario], 
                        outlier_threshold: float = 2.0) -> List[Scenario]:
        """
        Remove outlier scenarios using statistical methods.
        
        Args:
            scenarios: List of scenarios to filter
            outlier_threshold: Z-score threshold for outlier detection
            
        Returns:
            Filtered list of scenarios
        """
        if len(scenarios) < 10:  # Not enough data for outlier detection
            return scenarios
        
        # Group scenarios by obstacle and time step
        grouped_scenarios = {}
        for scenario in scenarios:
            key = (scenario.obstacle_idx_, scenario.time_step)
            if key not in grouped_scenarios:
                grouped_scenarios[key] = []
            grouped_scenarios[key].append(scenario)
        
        filtered_scenarios = []
        
        for key, group_scenarios in grouped_scenarios.items():
            if len(group_scenarios) < 5:  # Not enough for outlier detection
                filtered_scenarios.extend(group_scenarios)
                continue
            
            # Extract positions
            positions = np.array([s.position for s in group_scenarios])
            
            # Compute mean and standard deviation
            mean_pos = np.mean(positions, axis=0)
            std_pos = np.std(positions, axis=0)
            
            # Filter outliers
            for scenario in group_scenarios:
                z_scores = np.abs((scenario.position - mean_pos) / (std_pos + 1e-6))
                max_z_score = np.max(z_scores)
                
                if max_z_score <= outlier_threshold:
                    filtered_scenarios.append(scenario)
        
        removed_count = len(scenarios) - len(filtered_scenarios)
        if removed_count > 0:
            LOG_DEBUG(f"Removed {removed_count} outlier scenarios")
        
        return filtered_scenarios
    
    def integrate_and_translate_to_mean_and_variance(self, obstacles: List[DynamicObstacle], 
                                                   timestep: float):
        """
        Integrate obstacle dynamics and translate to mean and variance representation.
        
        Args:
            obstacles: List of dynamic obstacles
            timestep: Time step size
        """
        for obstacle in obstacles:
            if obstacle.prediction.type != PredictionType.GAUSSIAN:
                continue
                
            # This method would integrate the obstacle dynamics over time
            # and update the prediction steps with mean and variance
            # For now, we'll implement a simplified version
            
            if not obstacle.prediction.steps:
                # Create initial prediction step
                from planning.types import PredictionStep
                initial_step = PredictionStep(
                    position=obstacle.position,
                    angle=obstacle.angle,
                    major_radius=obstacle.radius,
                    minor_radius=obstacle.radius
                )
                obstacle.prediction.steps = [initial_step]
            
            # Update prediction steps with integrated dynamics
            self._integrate_obstacle_dynamics(obstacle, timestep)
    
    def _integrate_obstacle_dynamics(self, obstacle: DynamicObstacle, _timestep: float):
        """Integrate obstacle dynamics over time."""
        # Simplified integration - in practice, this would use the actual
        # obstacle dynamics model
        
        if not obstacle.prediction.steps:
            return
        
        # For now, just propagate uncertainty
        for i, step in enumerate(obstacle.prediction.steps):
            # Increase uncertainty over time (simple model)
            time_factor = 1.0 + i * 0.1
            step.major_radius *= time_factor
            step.minor_radius *= time_factor
    
    def get_sampler(self):
        """Get the sampler instance (for compatibility with existing code)."""
        return self
    
    def reset(self):
        """Reset sampler state."""
        self.scenarios = []


class MonteCarloValidator:
    """Validates collision probability using Monte Carlo methods."""
    
    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
    
    def validate_collision_probability(self, robot_trajectory: List[np.ndarray],
                                    obstacles: List[DynamicObstacle],
                                    robot_radius: float,
                                    target_probability: float = 0.05) -> Tuple[bool, float]:
        """
        Validate collision probability using Monte Carlo simulation.
        
        Args:
            robot_trajectory: Robot trajectory over horizon
            obstacles: List of dynamic obstacles
            robot_radius: Robot radius
            target_probability: Target collision probability threshold
            
        Returns:
            Tuple of (is_safe, actual_probability)
        """
        collision_count = 0
        
        for _ in range(self.num_samples):
            if self._check_collision_sample(robot_trajectory, obstacles, robot_radius):
                collision_count += 1
        
        actual_probability = collision_count / self.num_samples
        is_safe = actual_probability <= target_probability
        
        LOG_DEBUG(f"Monte Carlo validation: P(collision) = {actual_probability:.4f}, "
                 f"target = {target_probability:.4f}, safe = {is_safe}")
        
        return is_safe, actual_probability
    
    def _check_collision_sample(self, robot_trajectory: List[np.ndarray],
                              obstacles: List[DynamicObstacle],
                              robot_radius: float) -> bool:
        """Check collision for a single Monte Carlo sample."""
        for step, robot_pos in enumerate(robot_trajectory):
            for obstacle in obstacles:
                if step >= len(obstacle.prediction.steps):
                    continue
                
                pred_step = obstacle.prediction.steps[step]
                
                # Sample obstacle position from prediction
                sigma = max(pred_step.major_radius, pred_step.minor_radius)
                obstacle_pos = np.random.multivariate_normal(
                    pred_step.position[:2], 
                    np.array([[sigma**2, 0], [0, sigma**2]])
                )
                
                # Check collision
                distance = np.linalg.norm(robot_pos - obstacle_pos)
                if distance < robot_radius + obstacle.radius:
                    return True
        
        return False
