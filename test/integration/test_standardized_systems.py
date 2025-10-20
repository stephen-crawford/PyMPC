"""
Test Standardized Systems Integration

This test verifies that the standardized logging, visualization, and testing
framework work correctly together.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_logging import get_test_logger
from utils.standardized_visualization import VisualizationConfig, VisualizationMode
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer


class TestStandardizedSystems(BaseMPCTest):
    """
    Test to verify standardized systems integration.
    
    This test demonstrates that all standardized systems work together
    correctly without requiring actual MPC components.
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="test_name",
            description="Test with visualization framework",
            timeout=120.0,
            max_iterations=200,
            goal_tolerance=1.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME,
            log_level="INFO"
        )
        super().__init__(config)
        
        # Enhanced visualization configuration
        self.viz_config = VisualizationConfig(
            mode=VisualizationMode.REALTIME,
            realtime=True,
            show_constraint_projection=True,
            save_animation=True,
            save_plots=True,
            fps=10,
            dpi=100,
            output_dir=f"test_results/{config.test_name}/visualizations"
        )
        
        # Initialize enhanced visualizer
        if config.enable_visualization:
            self.visualizer = TestVisualizationManager(config.test_name)
            self.visualizer.initialize(self.viz_config)
        
        # Initialize debugging tools
        self.constraint_analyzer = ConstraintAnalyzer()
        self.solver_diagnostics = SolverDiagnostics()
        self.trajectory_analyzer = TrajectoryAnalyzer()
    
    def setup_test_environment(self):
        """Setup simple test environment."""
        self.logger.log_phase("Environment Setup", "Creating simple test environment")
        
        # Create simple straight path
        x_path = np.linspace(0, 20, 20)
        y_path = np.zeros(20)
        s_path = np.linspace(0, 1, 20)
        
        reference_path = {
            'x': x_path, 'y': y_path, 's': s_path
        }
        
        # Create road boundaries
        left_bound = {
            'x': x_path, 'y': y_path + 2, 's': s_path
        }
        
        right_bound = {
            'x': x_path, 'y': y_path - 2, 's': s_path
        }
        
        environment_data = {
            'start': (0, 0),
            'goal': (20, 0),
            'reference_path': reference_path,
            'left_bound': left_bound,
            'right_bound': right_bound,
            'dynamic_obstacles': []
        }
        
        self.logger.log_success("Environment setup completed")
        return environment_data
    
    def setup_mpc_system(self, data):
        """Setup mock MPC system for testing."""
        self.logger.log_phase("MPC System Setup", "Creating mock MPC system")
        
        # Create mock planner and solver
        class MockPlanner:
            def __init__(self):
                self.state = {'x': 0, 'y': 0, 'psi': 0, 'v': 0, 'spline': 0}
                self.solver = MockSolver()
            
            def get_state(self):
                return self.state
            
            def set_state(self, state):
                self.state = state
            
            def update_data(self, data):
                pass
            
            def solve(self):
                return MockResult()
        
        class MockSolver:
            def __init__(self):
                self.status = 'success'
                self.num_variables = 10
                self.num_constraints = 5
        
        class MockResult:
            def __init__(self):
                self.control_inputs = {'a': 1.0, 'w': 0.1}
        
        planner = MockPlanner()
        solver = MockSolver()
        
        self.logger.log_success("Mock MPC system setup completed")
        return planner, solver
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute mock MPC iteration."""
        iteration_start = time.time()
        
        try:
            # Get current state
            current_state = planner.get_state()
            
            # Simulate MPC solve
            result = planner.solve()
            
            # Analyze solver performance
            solve_time = time.time() - iteration_start
            diagnostic = self.solver_diagnostics.analyze_solver_performance(
                planner.solver, solve_time, iteration
            )
            
            # Extract control inputs
            control_inputs = result.control_inputs
            
            # Apply control
            new_state = self.apply_control(current_state, control_inputs)
            planner.set_state(new_state)
            
            # Log progress
            if iteration % 5 == 0:
                distance = np.linalg.norm([
                    new_state.get('x', 0) - data['goal'][0],
                    new_state.get('y', 0) - data['goal'][1]
                ])
                self.logger.log_info(f"Iteration {iteration}: Distance to goal: {distance:.3f}")
            
            return new_state
            
        except Exception as e:
            self.logger.log_error(f"Mock MPC iteration {iteration} failed", e)
            # Use fallback control
            return self.execute_fallback_control(planner, data, iteration)
    
    def check_goal_reached(self, state, goal):
        """Check if goal has been reached."""
        distance = np.linalg.norm([state.get('x', 0) - goal[0], state.get('y', 0) - goal[1]])
        return distance <= self.config.goal_tolerance
    
    def apply_control(self, state, control_inputs):
        """Apply control inputs to get new state."""
        dt = 0.1
        
        # Extract control inputs
        a = control_inputs.get('a', 0)
        w = control_inputs.get('w', 0)
        
        # Apply dynamics
        x = state.get('x', 0)
        y = state.get('y', 0)
        psi = state.get('psi', 0)
        v = state.get('v', 0)
        
        new_x = x + v * np.cos(psi) * dt
        new_y = y + v * np.sin(psi) * dt
        new_psi = psi + w * dt
        new_v = max(0, v + a * dt)
        new_spline = state.get('spline', 0) + v * dt
        
        return {
            'x': new_x, 'y': new_y, 'psi': new_psi, 
            'v': new_v, 'spline': new_spline
        }
    
    def generate_fallback_control(self, state, data):
        """Generate fallback control when MPC fails."""
        goal = data['goal']
        dx = goal[0] - state.get('x', 0)
        dy = goal[1] - state.get('y', 0)
        goal_angle = np.arctan2(dy, dx)
        
        angle_error = goal_angle - state.get('psi', 0)
        
        # Normalize angle error
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        return {'a': 1.0, 'w': angle_error * 2.0}
    

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
    def execute_fallback_control(self, planner, data, iteration):
        """Execute fallback control when MPC fails."""
        current_state = planner.get_state()
        control_inputs = self.generate_fallback_control(current_state, data)
        new_state = self.apply_control(current_state, control_inputs)
        planner.set_state(new_state)
        
        self.logger.log_warning(f"Using fallback control at iteration {iteration}")
        return new_state


# Run the test
if __name__ == "__main__":
    print("🧪 Testing Standardized Systems Integration")
    print("=" * 50)
    
    test = TestStandardizedSystems()
    result = test.run_test()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Test: {result.test_name}")
    print(f"Success: {'✅ PASSED' if result.success else '❌ FAILED'}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Iterations: {result.iterations_completed}")
    print(f"Final distance to goal: {result.final_distance_to_goal:.3f}")
    
    if not result.success:
        print(f"Failure reason: {result.failure_reason}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
    
    # Print performance metrics
    if result.performance_metrics:
        print(f"\n📈 PERFORMANCE METRICS")
        print(f"MPC failures: {result.performance_metrics.get('mpc_failures', 0)}")
        if result.performance_metrics.get('iteration_times'):
            avg_time = np.mean(result.performance_metrics['iteration_times'])
            print(f"Average iteration time: {avg_time:.3f}s")
    
    print(f"\n🎯 Standardized systems integration: {'✅ WORKING' if result.success else '❌ FAILED'}")
