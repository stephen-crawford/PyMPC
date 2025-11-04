"""
Main scenario module for safe horizon constraints.
"""
import numpy as np
from typing import List, Dict
from planning.types import Data, Scenario, ScenarioStatus, ScenarioSolveStatus, SupportSubsample
from modules.constraints.scenario_utils.math_utils import (
    Polytope, ScenarioConstraint, compute_sample_size, linearize_collision_constraint,
    construct_free_space_polytope, validate_polytope_feasibility
)
from modules.constraints.scenario_utils.sampler import ScenarioSampler, MonteCarloValidator
from utils.utils import LOG_DEBUG, LOG_WARN


class DiscManager:
    """Manages constraints for a single disc of the robot."""
    
    def __init__(self, disc_id: int, robot_radius: float, max_constraints: int):
        self.disc_id = disc_id
        self.robot_radius = robot_radius
        self.max_constraints = max_constraints
        self.polytopes = []  # One polytope per time step
        self.active_constraints = []
        
    def add_polytope(self, polytope: Polytope, time_step: int):
        """Add a polytope for a specific time step."""
        while len(self.polytopes) <= time_step:
            self.polytopes.append(Polytope([]))
        
        self.polytopes[time_step] = polytope
        
    def get_constraints_for_step(self, time_step: int) -> List[ScenarioConstraint]:
        """Get constraints for a specific time step."""
        if time_step >= len(self.polytopes):
            return []
        
        polytope = self.polytopes[time_step]
        constraints = []
        
        for i, halfspace in enumerate(polytope.halfspaces):
            if i >= self.max_constraints:
                break
                
            # Convert halfspace to scenario constraint
            A = halfspace.A
            b = halfspace.b
            
            if A.shape[0] > 0 and A.shape[1] >= 2:
                constraint = ScenarioConstraint(
                    a1=A[0, 0], a2=A[0, 1], b=b[0],
                    scenario_idx=i, obstacle_idx=0, time_step=time_step
                )
                constraints.append(constraint)
        
        return constraints


class SafeHorizonModule:
    """Main module for safe horizon scenario-based constraints."""
    
    def __init__(self, solver, config: Dict):
        self.solver = solver
        self.config = config
        
        # Parameters
        self.epsilon_p = config.get("epsilon_p", 0.1)  # Constraint violation probability
        self.beta = config.get("beta", 0.01)  # Confidence level
        self.n_bar = config.get("n_bar", 10)  # Support dimension
        self.num_removal = config.get("num_removal", 0)  # Number of scenarios to remove
        self.robot_radius = config.get("robot_radius", 0.5)
        self.horizon_length = config.get("horizon_length", 10)
        self.max_constraints_per_disc = config.get("max_constraints_per_disc", 24)
        self.num_discs = config.get("num_discs", 1)
        
        # Components
        self.sampler = ScenarioSampler(
            num_scenarios=config.get("num_scenarios", 100),
            enable_outlier_removal=config.get("enable_outlier_removal", True)
        )
        self.validator = MonteCarloValidator()
        
        # State
        self.disc_manager = []
        for i in range(self.num_discs):
            self.disc_manager.append(DiscManager(i, self.robot_radius, self.max_constraints_per_disc))
        
        self.scenarios = []
        self.support_subsample = SupportSubsample()
        self.status = ScenarioStatus.NONE
        self.solve_status = ScenarioSolveStatus.SUCCESS
        
        LOG_DEBUG("SafeHorizonModule initialized")
    
    def update(self, data: Data):
        """Update module with new data."""
        try:
            # Check if we have dynamic obstacles
            if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles):
                LOG_WARN("No dynamic obstacles in data")
                return
            
            # Sample scenarios
            self.scenarios = self.sampler.sample_scenarios(
                data.dynamic_obstacles, self.horizon_length, 
                getattr(self.solver, 'timestep', 0.1)
            )
            
            LOG_DEBUG(f"Updated with {len(self.scenarios)} scenarios")
            
        except Exception as e:
            LOG_WARN(f"Error updating SafeHorizonModule: {e}")
            return
    
    def optimize(self, data: Data) -> int:
        """
        Optimize scenario constraints.
        
        Returns:
            1 if successful, -1 if failed
        """
        try:
            LOG_DEBUG("Starting scenario optimization")
            
            # Update with current data
            self.update(data)
            
            if not self.scenarios:
                LOG_WARN("No scenarios available for optimization")
                return -1
            
            # Compute sample size
            sample_size = self.compute_sample_size()
            LOG_DEBUG(f"Computed sample size: {sample_size}")
            
            # Process scenarios for each disc and time step
            for disc_id in range(self.num_discs):
                for step in range(self.horizon_length):
                    self._process_scenarios_for_step(disc_id, step, data)
            
            self.status = ScenarioStatus.SUCCESS
            LOG_DEBUG("Scenario optimization completed successfully")
            return 1
            
        except Exception as e:
            LOG_WARN(f"Error in scenario optimization: {e}")
            self.status = ScenarioStatus.INFEASIBLE
            return -1
    
    def compute_sample_size(self) -> int:
        """Compute required sample size for scenario optimization."""
        return compute_sample_size(self.epsilon_p, self.beta, self.n_bar)
    
    def _process_scenarios_for_step(self, disc_id: int, step: int, _data: Data):
        """Process scenarios for a specific disc and time step."""
        # Get scenarios for this time step
        step_scenarios = [s for s in self.scenarios if s.time_step == step]
        
        if not step_scenarios:
            return
        
        # Formulate collision constraints
        constraints = self._formulate_collision_constraints(step_scenarios, disc_id, step)
        
        # Optional scenario removal with big-M relaxation
        if self.num_removal > 0:
            constraints = self.remove_scenarios_with_big_m(constraints, self.num_removal)
        
        # Construct free-space polytope
        polytope = construct_free_space_polytope(constraints)
        
        # Validate polytope feasibility
        if not validate_polytope_feasibility(polytope, []):
            LOG_WARN(f"Polytope infeasible for disc {disc_id}, step {step}")
            return
        
        # Add polytope to disc manager
        self.disc_manager[disc_id].add_polytope(polytope, step)
        
        # Track active constraints
        self._track_active_constraints(constraints, disc_id, step)
    
    def _formulate_collision_constraints(self, scenarios: List[Scenario], 
                                       _disc_id: int, step: int) -> List[ScenarioConstraint]:
        """Formulate collision constraints from scenarios."""
        constraints = []
        
        for scenario in scenarios:
            # Get robot position (simplified - in practice would come from solver state)
            robot_pos = np.array([0.0, 0.0])  # Placeholder
            
            # Linearize collision constraint
            constraint = linearize_collision_constraint(
                robot_pos, scenario.position, self.robot_radius, scenario.radius
            )
            constraint.scenario_idx = scenario.idx_
            constraint.obstacle_idx = scenario.obstacle_idx_
            constraint.time_step = step
            constraints.append(constraint)
        
        return constraints
    
    def _track_active_constraints(self, constraints: List[ScenarioConstraint], 
                                _disc_id: int, _step: int):
        """Track active constraints and support."""
        # Add constraints to support subsample
        for constraint in constraints:
            scenario = Scenario(constraint.scenario_idx, constraint.obstacle_idx)
            self.support_subsample.add(scenario)
        
        # Check support limits
        if self.support_subsample.size_ > self.n_bar:
            LOG_WARN(f"Support size {self.support_subsample.size_} exceeds limit {self.n_bar}")
            self.solve_status = ScenarioSolveStatus.SUPPORT_EXCEEDED
    
    def remove_scenarios_with_big_m(self, scenarios: List[ScenarioConstraint], 
                                  num_removal: int, _big_m: float = 1000.0) -> List[ScenarioConstraint]:
        """
        Remove scenarios using big-M relaxation method.
        
        Args:
            scenarios: List of scenario constraints
            num_removal: Number of scenarios to remove
            big_m: Big-M parameter for relaxation
            
        Returns:
            List of remaining scenarios after removal
        """
        if num_removal <= 0 or len(scenarios) <= num_removal:
            return scenarios
        
        # Sort scenarios by constraint violation potential (simplified heuristic)
        # In practice, this would use more sophisticated criteria
        sorted_scenarios = sorted(scenarios, key=lambda s: abs(s.b), reverse=True)
        
        # Remove the most restrictive scenarios
        remaining_scenarios = sorted_scenarios[num_removal:]
        
        LOG_DEBUG(f"Removed {num_removal} scenarios using big-M relaxation, "
                 f"{len(remaining_scenarios)} remaining")
        
        return remaining_scenarios
    
    def get_constraint_info(self) -> Dict:
        """Get information about current constraints."""
        total_constraints = 0
        for disc_manager in self.disc_manager:
            for polytope in disc_manager.polytopes:
                total_constraints += len(polytope.halfspaces)
        
        return {
            "total_scenarios": len(self.scenarios),
            "total_constraints": total_constraints,
            "support_size": self.support_subsample.size_,
            "support_limit": self.n_bar,
            "status": self.status,
            "solve_status": self.solve_status
        }
    
    def validate_parameters(self) -> bool:
        """Validate module parameters."""
        if self.epsilon_p <= 0 or self.epsilon_p >= 1:
            LOG_WARN(f"Invalid epsilon_p: {self.epsilon_p}")
            return False
        
        if self.beta <= 0 or self.beta >= 1:
            LOG_WARN(f"Invalid beta: {self.beta}")
            return False
        
        if self.n_bar <= 0:
            LOG_WARN(f"Invalid n_bar: {self.n_bar}")
            return False
        
        if self.robot_radius <= 0:
            LOG_WARN(f"Invalid robot_radius: {self.robot_radius}")
            return False
        
        return True
    
    def is_data_ready(self, data: Data) -> bool:
        """Check if required data is available."""
        try:
            if not (hasattr(data, 'dynamic_obstacles') and data.dynamic_obstacles):
                return False
            
            # Check that obstacles have valid predictions
            for obstacle in data.dynamic_obstacles:
                if not hasattr(obstacle, 'prediction') or not obstacle.prediction:
                    return False
                
                if obstacle.prediction.type.name not in ['GAUSSIAN', 'MULTIMODAL']:
                    return False
            
            return True
            
        except Exception as e:
            LOG_WARN(f"Error checking data readiness: {e}")
            return False
    
    def reset(self):
        """Reset module state."""
        self.scenarios = []
        self.support_subsample.reset()
        self.status = ScenarioStatus.NONE
        self.solve_status = ScenarioSolveStatus.SUCCESS
        
        for disc_manager in self.disc_manager:
            disc_manager.polytopes = []
            disc_manager.active_constraints = []
        
        self.sampler.reset()
        LOG_DEBUG("SafeHorizonModule reset")


class ScenarioSolver:
    """Wrapper for scenario solver with safe horizon module."""
    
    def __init__(self, solver_id: int, solver):
        self.solver_id = solver_id
        self.solver = solver
        self.scenario_module = SafeHorizonModule(solver, {})
        self.exit_code = 0
        self.solver_timeout = 0.1
        
    def get_sampler(self):
        """Get the sampler from the scenario module."""
        return self.scenario_module.sampler
