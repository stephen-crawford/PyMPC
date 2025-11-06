
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from dataclasses import dataclass

from planning.types import DynamicObstacle, PredictionType, PredictionStep
from planning.dynamic_models import DynamicsModel, SecondOrderUnicycleModel, SecondOrderBicycleModel, PointMassModel

@dataclass
class ObstacleConfig:
    """Configuration for obstacle creation."""
    obstacle_id: int
    initial_position: np.ndarray
    initial_velocity: np.ndarray = None
    initial_angle: float = 0.0
    radius: float = 0.35
    dynamics_type: str = "unicycle"
    prediction_type: PredictionType = PredictionType.GAUSSIAN
    control_inputs: Optional[np.ndarray] = None
    uncertainty_params: Optional[Dict[str, float]] = None

class ObstacleManager:
    """Manages obstacles for integration tests with state integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize obstacle manager."""
        self.config = config
        self.obstacles: List[DynamicObstacle] = []
        self.obstacle_states: List[List[np.ndarray]] = []
        self.obstacle_dynamics: List[DynamicsModel] = []
        self.logger = logging.getLogger("obstacle_manager")
        
        # Default uncertainty parameters
        self.default_uncertainty = {
            "position_std": 0.1,
            "velocity_std": 0.05,
            "angle_std": 0.1,
            "uncertainty_growth": 0.02
        }
        
    def create_obstacle(self, obstacle_config: 'ObstacleConfig') -> DynamicObstacle:
        """Create a new obstacle with specified dynamics."""
        self.logger.info(f"Creating obstacle {obstacle_config.obstacle_id} with {obstacle_config.dynamics_type} dynamics")
        
        # Create obstacle
        obstacle = DynamicObstacle(
            index=obstacle_config.obstacle_id,
            position=obstacle_config.initial_position.copy(),
            angle=obstacle_config.initial_angle,
            radius=obstacle_config.radius
        )
        
        # Set prediction type
        obstacle.prediction.type = obstacle_config.prediction_type
        obstacle.prediction.steps = []
        
        # Create dynamics model
        dynamics_model = self._create_dynamics_model(obstacle_config.dynamics_type)
        self.obstacle_dynamics.append(dynamics_model)
        
        # Initialize state
        initial_state = self._create_initial_state(obstacle_config, dynamics_model)
        self.obstacle_states.append([initial_state.copy()])
        
        # Generate prediction steps
        self._generate_prediction_steps(obstacle, dynamics_model, obstacle_config)
        
        self.obstacles.append(obstacle)
        return obstacle
        
    def _create_dynamics_model(self, dynamics_type: str) -> DynamicsModel:
        """Create dynamics model for obstacle."""
        if dynamics_type == "unicycle":
            return SecondOrderUnicycleModel()
        elif dynamics_type == "bicycle":
            return SecondOrderBicycleModel()
        elif dynamics_type == "point_mass":
            return PointMassModel()
        else:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")
            
    def _create_initial_state(self, obstacle_config: 'ObstacleConfig', dynamics_model: DynamicsModel) -> np.ndarray:
        """Create initial state vector for obstacle."""
        if obstacle_config.dynamics_type == "unicycle":
            # State: [x, y, psi, v]
            initial_velocity = obstacle_config.initial_velocity if obstacle_config.initial_velocity is not None else np.array([1.0, 0.0])
            speed = np.linalg.norm(initial_velocity)
            return np.array([
                obstacle_config.initial_position[0],
                obstacle_config.initial_position[1],
                obstacle_config.initial_angle,
                speed
            ])
        elif obstacle_config.dynamics_type == "bicycle":
            # State: [x, y, psi, v, delta, spline]
            initial_velocity = obstacle_config.initial_velocity if obstacle_config.initial_velocity is not None else np.array([1.0, 0.0])
            speed = np.linalg.norm(initial_velocity)
            return np.array([
                obstacle_config.initial_position[0],
                obstacle_config.initial_position[1],
                obstacle_config.initial_angle,
                speed,
                0.0,  # delta
                0.0   # spline
            ])
        elif obstacle_config.dynamics_type == "point_mass":
            # State: [x, y, vx, vy]
            initial_velocity = obstacle_config.initial_velocity if obstacle_config.initial_velocity is not None else np.array([1.0, 0.0])
            return np.array([
                obstacle_config.initial_position[0],
                obstacle_config.initial_position[1],
                initial_velocity[0],
                initial_velocity[1]
            ])
        else:
            raise ValueError(f"Unknown dynamics type: {obstacle_config.dynamics_type}")
            
    def _generate_prediction_steps(self, obstacle: DynamicObstacle, dynamics_model: DynamicsModel, 
                                  obstacle_config: 'ObstacleConfig'):
        """Generate prediction steps for obstacle."""
        horizon_length = self.config.get("horizon", 10)
        timestep = self.config.get("timestep", 0.1)
        
        # Get uncertainty parameters
        uncertainty_params = obstacle_config.uncertainty_params or self.default_uncertainty
        
        # Generate prediction steps
        for step in range(horizon_length):
            # Integrate obstacle state
            current_state = self.obstacle_states[-1][-1] if self.obstacle_states[-1] else self._create_initial_state(obstacle_config, dynamics_model)
            
            # Create control input (can be specified or generated)
            if obstacle_config.control_inputs is not None:
                control_input = obstacle_config.control_inputs[step] if step < len(obstacle_config.control_inputs) else np.zeros(dynamics_model.nu)
            else:
                control_input = self._generate_control_input(dynamics_model, current_state, step)
            
            # Integrate state
            next_state = self._integrate_state(dynamics_model, current_state, control_input, timestep)
            
            # Store state
            if hasattr(next_state, 'copy'):
                self.obstacle_states[-1].append(next_state.copy())
            else:
                # Handle CasADi objects
                self.obstacle_states[-1].append(next_state)
            
            # Create prediction step
            position = next_state[:2]  # x, y
            angle = next_state[2] if len(next_state) > 2 else 0.0
            
            # Add uncertainty depending on prediction type
            if obstacle.prediction.type == PredictionType.GAUSSIAN:
                uncertainty_std = uncertainty_params["position_std"] + step * uncertainty_params["uncertainty_growth"]
                major_r = obstacle.radius + uncertainty_std
                minor_r = obstacle.radius + uncertainty_std
            else:
                # Deterministic (and any non-gaussian default): no growth
                uncertainty_std = 0.0
                major_r = obstacle.radius
                minor_r = obstacle.radius
            
            prediction_step = PredictionStep(
                position=position,
                angle=angle,
                major_radius=major_r,
                minor_radius=minor_r
            )
            
            obstacle.prediction.steps.append(prediction_step)
            
    def _generate_control_input(self, dynamics_model: DynamicsModel, current_state: np.ndarray, step: int) -> np.ndarray:
        """Generate control input for obstacle."""
        if dynamics_model.nu == 2:  # Unicycle: [a, w]
            # Simple control: maintain speed, slight turning
            return np.array([0.0, 0.1 * np.sin(step * 0.1)])
        elif dynamics_model.nu == 3:  # Bicycle: [a, w, slack]
            return np.array([0.0, 0.1 * np.sin(step * 0.1), 0.0])
        else:
            return np.zeros(dynamics_model.nu)
            
    def _integrate_state(self, dynamics_model: DynamicsModel, current_state: np.ndarray, 
                        control_input: np.ndarray, timestep: float) -> np.ndarray:
        """Integrate obstacle state using simple numeric integration."""
        try:
            # Use simple Euler integration for obstacles
            # This avoids CasADi symbolic issues
            
            if dynamics_model.__class__.__name__ == "SecondOrderUnicycleModel":
                # State: [x, y, psi, v]
                x, y, psi, v = current_state
                a, w = control_input
                
                # Simple Euler integration
                x_new = x + v * np.cos(psi) * timestep
                y_new = y + v * np.sin(psi) * timestep
                psi_new = psi + w * timestep
                v_new = v + a * timestep
                
                return np.array([x_new, y_new, psi_new, v_new])
                
            elif dynamics_model.__class__.__name__ == "SecondOrderBicycleModel":
                # State: [x, y, psi, v, delta, spline]
                x, y, psi, v, delta, spline = current_state
                a, w, slack = control_input
                
                # Simple Euler integration
                x_new = x + v * np.cos(psi) * timestep
                y_new = y + v * np.sin(psi) * timestep
                psi_new = psi + w * timestep
                v_new = v + a * timestep
                delta_new = delta + w * timestep  # Simplified
                spline_new = spline + v * timestep
                
                return np.array([x_new, y_new, psi_new, v_new, delta_new, spline_new])
                
            elif dynamics_model.__class__.__name__ == "PointMassModel":
                # State: [x, y, vx, vy]
                x, y, vx, vy = current_state
                ax, ay = control_input
                
                # Simple Euler integration
                x_new = x + vx * timestep
                y_new = y + vy * timestep
                vx_new = vx + ax * timestep
                vy_new = vy + ay * timestep
                
                return np.array([x_new, y_new, vx_new, vy_new])
                
            else:
                # Fallback: simple constant velocity
                return current_state + np.concatenate([control_input, np.zeros(len(current_state) - len(control_input))]) * timestep
                
        except Exception as e:
            self.logger.warning(f"Error integrating obstacle state: {e}")
            # Return current state as fallback
            return current_state.copy()
            
    def update_obstacle_states(self, timestep: float):
        """Update all obstacle states for one timestep."""
        for i, (obstacle, dynamics_model) in enumerate(zip(self.obstacles, self.obstacle_dynamics)):
            if i < len(self.obstacle_states) and len(self.obstacle_states[i]) > 0:
                current_state = self.obstacle_states[i][-1]
                
                # Generate control input
                control_input = self._generate_control_input(dynamics_model, current_state, len(self.obstacle_states[i]))
                
                # Integrate state
                next_state = self._integrate_state(dynamics_model, current_state, control_input, timestep)
                
                # Store new state
                self.obstacle_states[i].append(next_state.copy())
                
                # Update obstacle position
                obstacle.position = next_state[:2]
                if len(next_state) > 2:
                    obstacle.angle = next_state[2]
                    
    def get_obstacle_states(self, obstacle_id: int) -> List[np.ndarray]:
        """Get state history for specific obstacle."""
        if 0 <= obstacle_id < len(self.obstacle_states):
            return self.obstacle_states[obstacle_id]
        return []
        
    def get_all_obstacle_states(self) -> List[List[np.ndarray]]:
        """Get state history for all obstacles."""
        return self.obstacle_states
        
    def get_obstacle_at_time(self, obstacle_id: int, time_step: int) -> Optional[np.ndarray]:
        """Get obstacle state at specific time step."""
        if 0 <= obstacle_id < len(self.obstacle_states):
            states = self.obstacle_states[obstacle_id]
            if 0 <= time_step < len(states):
                return states[time_step]
        return None
        
    def create_obstacles_from_config(self, obstacle_configs: List['ObstacleConfig']) -> List[DynamicObstacle]:
        """Create multiple obstacles from configuration list."""
        obstacles = []
        for config in obstacle_configs:
            obstacle = self.create_obstacle(config)
            obstacles.append(obstacle)
        return obstacles
        
    def create_random_obstacles(self, num_obstacles: int, dynamics_types: List[str], 
                               bounds: Tuple[float, float, float, float] = (0.0, 20.0, -5.0, 5.0)) -> List[DynamicObstacle]:
        """Create random obstacles for testing."""
        obstacles = []
        
        for i in range(num_obstacles):
            dynamics_type = dynamics_types[i % len(dynamics_types)]
            
            # Random initial position
            x = np.random.uniform(bounds[0], bounds[1])
            y = np.random.uniform(bounds[2], bounds[3])
            
            # Random initial velocity
            speed = np.random.uniform(0.5, 2.0)
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            
            config = ObstacleConfig(
                obstacle_id=i,
                initial_position=np.array([x, y]),
                initial_velocity=velocity,
                initial_angle=angle,
                radius=self.config.get("obstacle_radius", 0.35),
                dynamics_type=dynamics_type,
                prediction_type=PredictionType.GAUSSIAN
            )
            
            obstacle = self.create_obstacle(config)
            obstacles.append(obstacle)
            
        return obstacles
        
    def reset_obstacles(self):
        """Reset all obstacles to initial states."""
        for i, obstacle in enumerate(self.obstacles):
            if i < len(self.obstacle_states) and len(self.obstacle_states[i]) > 0:
                initial_state = self.obstacle_states[i][0]
                obstacle.position = initial_state[:2]
                if len(initial_state) > 2:
                    obstacle.angle = initial_state[2]
                    
        # Reset state histories to initial states only
        for i in range(len(self.obstacle_states)):
            if len(self.obstacle_states[i]) > 0:
                self.obstacle_states[i] = [self.obstacle_states[i][0].copy()]
                
    def get_obstacle_info(self) -> Dict[str, Any]:
        """Get information about all obstacles."""
        info = {
            "num_obstacles": len(self.obstacles),
            "obstacle_details": []
        }
        
        for i, obstacle in enumerate(self.obstacles):
            obstacle_info = {
                "id": obstacle.index,
                "position": obstacle.position.tolist(),
                "angle": obstacle.angle,
                "radius": obstacle.radius,
                "prediction_type": obstacle.prediction.type.name,
                "num_prediction_steps": len(obstacle.prediction.steps),
                "dynamics_type": self.obstacle_dynamics[i].__class__.__name__ if i < len(self.obstacle_dynamics) else "Unknown",
                "state_history_length": len(self.obstacle_states[i]) if i < len(self.obstacle_states) else 0
            }
            info["obstacle_details"].append(obstacle_info)
            
        return info


# Convenience functions for creating obstacle configurations
def create_unicycle_obstacle(obstacle_id: int, position: np.ndarray, velocity: np.ndarray = None, 
                           angle: float = 0.0, radius: float = 0.35) -> ObstacleConfig:
    """Create unicycle obstacle configuration."""
    if velocity is None:
        velocity = np.array([1.0, 0.0])
    return ObstacleConfig(
        obstacle_id=obstacle_id,
        initial_position=position,
        initial_velocity=velocity,
        initial_angle=angle,
        radius=radius,
        dynamics_type="unicycle",
        prediction_type=PredictionType.DETERMINISTIC
    )


def create_bicycle_obstacle(obstacle_id: int, position: np.ndarray, velocity: np.ndarray = None, 
                          angle: float = 0.0, radius: float = 0.35) -> ObstacleConfig:
    """Create bicycle obstacle configuration."""
    if velocity is None:
        velocity = np.array([1.0, 0.0])
    return ObstacleConfig(
        obstacle_id=obstacle_id,
        initial_position=position,
        initial_velocity=velocity,
        initial_angle=angle,
        radius=radius,
        dynamics_type="bicycle",
        prediction_type=PredictionType.DETERMINISTIC
    )


def create_point_mass_obstacle(obstacle_id: int, position: np.ndarray, velocity: np.ndarray = None, 
                             radius: float = 0.35) -> ObstacleConfig:
    """Create point mass obstacle configuration."""
    if velocity is None:
        velocity = np.array([1.0, 0.0])
    return ObstacleConfig(
        obstacle_id=obstacle_id,
        initial_position=position,
        initial_velocity=velocity,
        initial_angle=0.0,
        radius=radius,
        dynamics_type="point_mass",
        prediction_type=PredictionType.DETERMINISTIC
    )

