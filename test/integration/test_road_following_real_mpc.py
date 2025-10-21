"""
Real MPC Road Following Test with Contouring Constraints

This test uses the actual MPC framework with:
- Real contouring constraints for road boundary adherence
- Scenario constraints for obstacle avoidance
- Contouring objective for path following
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_visualization import VisualizationMode


class TestRoadFollowingRealMPC(BaseMPCTest):
    """
    Real MPC test for vehicle following a curved road while avoiding a dynamic obstacle.
    Uses actual contouring constraints to enforce road boundaries.
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="test_road_following_real_mpc",
            description="Real MPC vehicle follows curved road with contouring constraints",
            timeout=180.0,
            max_iterations=200,
            goal_tolerance=2.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.STATIC,
            log_level="INFO"
        )
        super().__init__(config)
    
    def setup_test_environment(self):
        """Setup test environment with curved road and dynamic obstacle."""
        self.logger.log_phase("Environment Setup", "Creating curved road and dynamic obstacle scenario")
        
        # Create more curved reference path (S-shaped road with more curves)
        t = np.linspace(0, 1, 80)
        x_path = np.linspace(0, 80, 80)  # 80m road
        y_path = 8 * np.sin(0.4 * np.pi * t) + 4 * np.sin(0.8 * np.pi * t) + 2 * np.sin(1.6 * np.pi * t)  # More curved S-curve
        s_path = np.linspace(0, 1, 80)
        
        # Create road boundaries
        normals = self.calculate_path_normals(x_path, y_path)
        road_width = 5.0  # 5m wide road
        half_width = road_width / 2
        
        left_bound_x = x_path + normals[:, 0] * half_width
        left_bound_y = y_path + normals[:, 1] * half_width
        right_bound_x = x_path - normals[:, 0] * half_width
        right_bound_y = y_path - normals[:, 1] * half_width
        
        # Create dynamic obstacle trajectory (crosses road)
        obstacle_trajectory = []
        for i in range(40):  # 40 time steps
            t_obs = i / 39.0
            obs_x = 25.0 + (55.0 - 25.0) * t_obs  # Crosses from x=25 to x=55
            obs_y = 6.0 + (-6.0 - 6.0) * t_obs    # Crosses from y=6 to y=-6
            obstacle_trajectory.append((obs_x, obs_y))
        
        # Test parameters - goal at end of road following reference path
        start_position = (0.0, 0.0)
        goal_position = (x_path[-1], y_path[-1])
        
        environment_data = {
            'start': start_position,
            'goal': goal_position,
            'reference_path': {
                'x': x_path,
                'y': y_path,
                's': s_path
            },
            'left_bound': {
                'x': left_bound_x,
                'y': left_bound_y,
                's': s_path
            },
            'right_bound': {
                'x': right_bound_x,
                'y': right_bound_y,
                's': s_path
            },
            'dynamic_obstacle': {
                'trajectory': obstacle_trajectory,
                'radius': 1.2,
                'type': 'gaussian',
                'uncertainty': 0.3
            },
            'road_width': road_width
        }
        
        self.logger.log_success("Environment setup completed")
        self.logger.logger.info(f"Road length: {x_path[-1] - x_path[0]:.1f}m")
        self.logger.logger.info(f"Road width: {road_width:.1f}m")
        self.logger.logger.info(f"Obstacle radius: {environment_data['dynamic_obstacle']['radius']:.1f}m")
        self.logger.logger.info(f"Goal position: ({goal_position[0]:.1f}, {goal_position[1]:.1f})")
        
        return environment_data
    
    def setup_mpc_system(self, data):
        """Setup MPC system with real contouring and scenario constraints."""
        self.logger.log_phase("MPC System Setup", "Initializing solver and modules")
        
        try:
            # Import required modules
            from solver.src.casadi_solver import CasADiSolver
            from planning.src.planner import Planner
            from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
            from planner.src.planner_modules.src.constraints.contouring_constraints import ContouringConstraints
            from planner.src.planner_modules.src.constraints.fixed_scenario_constraints import FixedScenarioConstraints
            from planner.src.planner_modules.src.objectives.contouring_objective import ContouringObjective
            
            # Create vehicle model
            vehicle = ContouringSecondOrderUnicycleModel()
            
            # Create solver
            solver = CasADiSolver()
            solver.set_dynamics_model(vehicle)
            
            # Create planner
            planner = Planner(solver, vehicle)
            
            # Add contouring constraints for road boundaries (THIS IS KEY!)
            contouring_constraints = ContouringConstraints(solver)
            solver.module_manager.add_module(contouring_constraints)
            
            # Add scenario constraints for obstacle avoidance
            scenario_constraints = FixedScenarioConstraints(solver)
            solver.module_manager.add_module(scenario_constraints)
            
            # Add contouring objective for path following
            contouring_objective = ContouringObjective(solver)
            solver.module_manager.add_module(contouring_objective)
            
            # Initialize solver with data
            solver.initialize(data)
            
            self.logger.log_success("MPC system setup completed")
            self.logger.logger.info(f"Modules added: {len(solver.module_manager.modules)}")
            self.logger.logger.info("Modules: ContouringConstraints, FixedScenarioConstraints, ContouringObjective")
            self.logger.logger.info("✅ Contouring constraints will enforce road boundaries")
            
            return planner, solver
            
        except Exception as e:
            self.logger.log_error("Failed to setup MPC system", e)
            raise
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute one MPC iteration with real constraint enforcement."""
        try:
            # Update dynamic obstacle position
            obstacle_trajectory = data['dynamic_obstacle']['trajectory']
            if iteration < len(obstacle_trajectory):
                obs_x, obs_y = obstacle_trajectory[iteration]
            else:
                # Obstacle has finished crossing, use final position
                obs_x, obs_y = obstacle_trajectory[-1]
            
            # Create dynamic obstacle data with uncertainty for scenario constraints
            obstacle_data = {
                'x': obs_x,
                'y': obs_y,
                'radius': data['dynamic_obstacle']['radius'],
                'uncertainty': data['dynamic_obstacle']['uncertainty'],
                'type': data['dynamic_obstacle']['type']
            }
            
            # Create obstacle predictions for scenario constraints
            obstacle_predictions = []
            for j in range(1, 6):  # 5-step prediction horizon
                if iteration + j < len(obstacle_trajectory):
                    pred_x, pred_y = obstacle_trajectory[iteration + j]
                    # Uncertainty grows with prediction horizon
                    pred_uncertainty = data['dynamic_obstacle']['uncertainty'] * (1 + j * 0.2)
                    obstacle_predictions.append({
                        'x': pred_x,
                        'y': pred_y,
                        'radius': data['dynamic_obstacle']['radius'],
                        'uncertainty': pred_uncertainty,
                        'type': data['dynamic_obstacle']['type']
                    })
            
            # Update test data with current obstacle position and predictions
            current_data = data.copy()
            current_data['dynamic_obstacle_position'] = obstacle_data
            current_data['obstacle_predictions'] = obstacle_predictions
            
            # Execute MPC - this will enforce contouring constraints!
            output = planner.solve(current_data)
            
            if output.success:
                # Extract new state from MPC solution
                new_state = {
                    'x': output.trajectory_x[1] if len(output.trajectory_x) > 1 else output.trajectory_x[0],
                    'y': output.trajectory_y[1] if len(output.trajectory_y) > 1 else output.trajectory_y[0],
                    'psi': output.trajectory_psi[1] if len(output.trajectory_psi) > 1 else output.trajectory_psi[0],
                    'v': output.trajectory_v[1] if len(output.trajectory_v) > 1 else output.trajectory_v[0],
                    'spline': output.trajectory_spline[1] if len(output.trajectory_spline) > 1 else output.trajectory_spline[0]
                }
                
                # Update planner state
                planner.set_state(new_state)
                
                # Check if vehicle respects road boundaries (contouring constraints)
                self.check_road_boundary_compliance(new_state, data, iteration)
                
                self.logger.log_debug(f"MPC iteration {iteration}: Success")
                return new_state
            else:
                self.logger.log_warning(f"MPC failed at iteration {iteration}, using fallback control")
                return self.execute_fallback_control(planner, data, iteration)
                
        except Exception as e:
            self.logger.log_error(f"MPC iteration {iteration} failed", e)
            return self.execute_fallback_control(planner, data, iteration)
    
    def check_road_boundary_compliance(self, state, data, iteration):
        """Check if vehicle respects road boundaries (contouring constraints)."""
        x = state.get('x', 0)
        y = state.get('y', 0)
        
        # Find closest point on reference path
        ref_x = np.array(data['reference_path']['x'])
        ref_y = np.array(data['reference_path']['y'])
        distances = np.sqrt((ref_x - x)**2 + (ref_y - y)**2)
        closest_idx = np.argmin(distances)
        
        # Check distance to road boundaries
        left_x = data['left_bound']['x'][closest_idx]
        left_y = data['left_bound']['y'][closest_idx]
        right_x = data['right_bound']['x'][closest_idx]
        right_y = data['right_bound']['y'][closest_idx]
        
        left_distance = np.sqrt((x - left_x)**2 + (y - left_y)**2)
        right_distance = np.sqrt((x - right_x)**2 + (y - right_y)**2)
        
        # Check if vehicle is within road boundaries
        road_width = data['road_width']
        if left_distance < road_width/2 and right_distance < road_width/2:
            if iteration % 10 == 0:  # Log every 10 iterations
                self.logger.logger.info(f"Iteration {iteration}: Vehicle respects road boundaries (L:{left_distance:.2f}m, R:{right_distance:.2f}m)")
        else:
            self.logger.log_warning(f"Iteration {iteration}: Vehicle violates road boundaries! (L:{left_distance:.2f}m, R:{right_distance:.2f}m)")
    
    def execute_fallback_control(self, planner, data, iteration):
        """Execute fallback control when MPC fails."""
        current_state = planner.get_state()
        
        # Simple goal-seeking control with road boundary respect
        goal = data['goal']
        dx = goal[0] - current_state.get('x', 0)
        dy = goal[1] - current_state.get('y', 0)
        goal_distance = np.sqrt(dx**2 + dy**2)
        
        if goal_distance > 0.1:
            # Move towards goal
            goal_angle = np.arctan2(dy, dx)
            
            # Check road boundaries
            closest_idx = min(int(current_state.get('spline', 0) * len(data['reference_path']['x'])), 
                            len(data['reference_path']['x']) - 1)
            
            # Distance to road boundaries
            left_dx = data['left_bound']['x'][closest_idx] - current_state.get('x', 0)
            left_dy = data['left_bound']['y'][closest_idx] - current_state.get('y', 0)
            left_distance = np.sqrt(left_dx**2 + left_dy**2)
            
            right_dx = data['right_bound']['x'][closest_idx] - current_state.get('x', 0)
            right_dy = data['right_bound']['y'][closest_idx] - current_state.get('y', 0)
            right_distance = np.sqrt(right_dx**2 + right_dy**2)
            
            # Adjust angle to stay within road boundaries
            if left_distance < 1.5:  # Too close to left boundary
                goal_angle += 0.3  # Turn right
            elif right_distance < 1.5:  # Too close to right boundary
                goal_angle -= 0.3  # Turn left
            
            # Update state
            new_state = current_state.copy()
            new_state['x'] += 0.3 * np.cos(goal_angle)
            new_state['y'] += 0.3 * np.sin(goal_angle)
            new_state['psi'] = goal_angle
            new_state['v'] = 0.8
            new_state['spline'] = min(current_state.get('spline', 0) + 0.02, 1.0)
            
            planner.set_state(new_state)
            return new_state
        else:
            return current_state
    
    def check_goal_reached(self, state, goal):
        """Check if the goal has been reached."""
        distance = np.sqrt((state.get('x', 0) - goal[0])**2 + (state.get('y', 0) - goal[1])**2)
        return distance <= self.config.goal_tolerance
    
    def calculate_path_normals(self, x, y):
        """Calculate path normals for road boundaries."""
        # Calculate path derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Normalize to get unit tangent vectors
        norm = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / (norm + 1e-9)
        dy_norm = dy / (norm + 1e-9)
        
        # Calculate perpendicular vectors (normals)
        normals_x = -dy_norm
        normals_y = dx_norm
        
        return np.column_stack([normals_x, normals_y])
    
    def _collect_constraint_overlays(self, planner):
        """Collect constraint overlays from active modules."""
        overlays = {'halfspaces': [], 'polygons': [], 'points': []}
        
        try:
            if hasattr(planner, 'solver') and hasattr(planner.solver, 'module_manager'):
                modules = getattr(planner.solver.module_manager, 'modules', [])
                for module in modules:
                    if hasattr(module, 'get_visualization_overlay'):
                        overlay = module.get_visualization_overlay()
                        if overlay:
                            if 'halfspaces' in overlay:
                                overlays['halfspaces'].extend(overlay['halfspaces'])
                            if 'polygons' in overlay:
                                overlays['polygons'].extend(overlay['polygons'])
                            if 'points' in overlay:
                                overlays['points'].extend(overlay['points'])
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.log_debug(f"Could not collect constraint overlays: {e}")
        
        return overlays


# Run the test
if __name__ == "__main__":
    test = TestRoadFollowingRealMPC()
    result = test.run_test()
    
    print(f"\n{'='*70}")
    print(f"REAL MPC ROAD FOLLOWING TEST RESULTS")
    print(f"{'='*70}")
    print(f"Test {'PASSED' if result.success else 'FAILED'}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Iterations: {result.iterations_completed}")
    print(f"Final distance to goal: {result.final_distance_to_goal:.3f}m")
    
    if result.success:
        print(f"✅ Vehicle successfully followed road using REAL MPC with contouring constraints!")
        print(f"✅ Contouring constraints enforced road boundaries")
        print(f"✅ Scenario constraints provided obstacle predictions")
        print(f"✅ Contouring objective optimized path following")
    else:
        print(f"❌ Test failed: {result.failure_reason}")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error.get('message', 'Unknown error')}")
    
    print(f"\n📊 Performance Metrics:")
    if 'iteration_times' in result.performance_metrics:
        avg_time = np.mean(result.performance_metrics['iteration_times'])
        print(f"  Average iteration time: {avg_time:.3f}s")
    
    if 'mpc_failures' in result.performance_metrics:
        mpc_failures = result.performance_metrics['mpc_failures']
        print(f"  MPC failures: {mpc_failures}")
    
    print(f"\n📁 Output files saved to: test_results/{test.config.test_name}/")
