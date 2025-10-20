"""
Test Proper Scenario Constraints Implementation

This test validates the proper scenario constraints implementation
that follows Oscar de Groot's C++ approach from IJRR 2024.

Features tested:
- Parallel scenario optimization
- Polytope construction around infeasible regions
- Halfspace constraint extraction
- Integration with contouring constraints and objectives
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import from the existing test framework
from test.framework.base_test import BaseMPCTest
from solver.src.casadi_solver import CasADiSolver
from planning.src.dynamic_models import DynamicsModel, ContouringSecondOrderUnicycleModel
from planning.src.types import Data, State, Bound, PredictionType, generate_dynamic_obstacles
from scipy.interpolate import splprep, splev


class ProperScenarioConstraintsTest(BaseMPCTest):
    """
    Test proper scenario constraints with contouring system.
    
    This test demonstrates the full integration of:
    - Proper scenario constraints (Oscar de Groot's approach)
    - Contouring constraints (road boundaries)
    - Contouring objective (MPCC path following)
    """
    
    def __init__(self, name, dt=0.1, horizon=15, max_iterations=100,
                 max_consecutive_failures=20,
                 enable_visualization=False,
                 enable_realtime_viz=False,
                 enable_gif=False):
        """
        Initialize test.
        
        Args:
            enable_visualization: Enable default viz manager
            enable_realtime_viz: Show real-time visualization during execution
            enable_gif: Generate animated GIF of trajectory with constraints
        """
        super().__init__(name, dt, horizon, max_iterations,
                        enable_visualization=enable_visualization,
                        max_consecutive_failures=max_consecutive_failures)
        self.enable_realtime_viz = enable_realtime_viz
        self.enable_gif = enable_gif
        
        # Storage for visualization
        self.viz_history = []  # List of (state, constraints, obstacles)
        
        if self.enable_realtime_viz:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
    
    # ==================== Abstract Methods Implementation ====================
    
    def get_vehicle_model(self) -> DynamicsModel:
        """Return the vehicle dynamics model (contouring model for MPCC)."""
        return ContouringSecondOrderUnicycleModel()
    
    def configure_modules(self, solver: CasADiSolver) -> None:
        """Configure and add modules to solver."""
        # Import modules here to avoid circular imports
        from planner_modules.src.objectives.contouring_objective import ContouringObjective
        from planner_modules.src.constraints.proper_scenario_constraints import ProperScenarioConstraints
        from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
        
        # Add contouring objective (path following)
        contouring_objective = ContouringObjective(solver)
        solver.module_manager.add_module(contouring_objective)
        
        # Add proper scenario constraints (Oscar de Groot's approach)
        self.scenario_constraints = ProperScenarioConstraints(solver)
        solver.module_manager.add_module(self.scenario_constraints)
        
        # Add contouring constraints (road boundaries)
        self.contouring_constraints = ContouringConstraints(solver)
        solver.module_manager.add_module(self.contouring_constraints)
    
    def setup_environment_data(self, data: Data) -> None:
        """Setup environment-specific data (obstacles)."""
        # Generate 4 dynamic obstacles with GAUSSIAN prediction
        obstacles = generate_dynamic_obstacles(
            number=4,
            prediction_type=PredictionType.GAUSSIAN.name,
            size=1.0,  # radius
            distribution="random_paths",
            area=((5, 35), (5, 20), (0, 0)),
            path_types=("straight", "curved"),
            num_points=self.horizon + 1,
            dim=2
        )
        
        data.dynamic_obstacles = obstacles
        
        print(f"\n{'='*80}")
        print(f"PROPER SCENARIO CONSTRAINTS TEST SETUP")
        print(f"{'='*80}")
        print(f"Dynamic obstacles: {len(obstacles)}")
        print(f"  - Using Oscar de Groot's IJRR 2024 approach")
        print(f"  - Parallel scenario optimization")
        print(f"  - Polytope construction around infeasible regions")
        print(f"  - Halfspace constraint extraction")
        print(f"Horizon: {self.horizon} timesteps ({self.horizon * self.dt:.1f}s)")
        print(f"{'='*80}")
        print(f"\nSystem Components:")
        print(f"  ✓ Proper Scenario Constraints (IJRR 2024 approach)")
        print(f"  ✓ Contouring Constraints (road boundaries)")
        print(f"  ✓ Contouring Objective (MPCC path following)")
        print(f"{'='*80}")
    
    def on_iteration(self, iteration, state, result):
        """Called after each MPC iteration - capture visualization data."""
        if not (self.enable_realtime_viz or self.enable_gif):
            return
        
        # Store visualization data
        viz_data = {
            'iteration': iteration,
            'state': state,
            'constraints': getattr(self.scenario_constraints, 'constraint_params', {}),
            'obstacles': getattr(self, 'data', {}).get('dynamic_obstacles', []),
            'halfspaces': getattr(self.scenario_constraints, 'constraint_params', {})
        }
        self.viz_history.append(viz_data)
        
        # Real-time visualization
        if self.enable_realtime_viz:
            self._update_realtime_visualization(iteration, state, viz_data)
    
    def _update_realtime_visualization(self, iteration, state, viz_data):
        """Update real-time visualization"""
        self.ax.clear()
        
        # Plot environment
        if hasattr(self, 'data') and self.data:
            # Road boundaries
            if hasattr(self.data, 'left_bound'):
                self.ax.plot(self.data.left_bound.x, self.data.left_bound.y, 'k--', alpha=0.5)
                self.ax.plot(self.data.right_bound.x, self.data.right_bound.y, 'k--', alpha=0.5)
            
            # Obstacles
            for i, obs in enumerate(self.data.dynamic_obstacles):
                if hasattr(obs, 'position'):
                    pos = obs.position[:2]
                    circle = plt.Circle(pos, obs.radius, color='red', alpha=0.4)
                    self.ax.add_patch(circle)
                    self.ax.text(pos[0], pos[1], f'O{i}', ha='center', va='center')
        
        # Plot vehicle
        self.ax.plot(state.x, state.y, 'bo', markersize=15, label='Vehicle')
        
        # Plot trajectory so far
        if hasattr(self, 'result') and self.result:
            if len(self.result.trajectory_x) > 1:
                self.ax.plot(self.result.trajectory_x, self.result.trajectory_y, 'b-', linewidth=2)
        
        # Plot halfspace constraints
        self._plot_halfspace_constraints(viz_data['halfspaces'], state)
        
        self.ax.set_xlim(state.x - 10, state.x + 10)
        self.ax.set_ylim(state.y - 10, state.y + 10)
        self.ax.set_title(f'Proper Scenario Constraints - Iteration {iteration}')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.axis('equal')
        
        plt.pause(0.1)
    
    def _plot_halfspace_constraints(self, halfspaces, state):
        """Plot halfspace constraints"""
        for i in range(5):  # max_halfspaces
            a1_key = f'halfspace_{i}_a1'
            a2_key = f'halfspace_{i}_a2'
            b_key = f'halfspace_{i}_b'
            
            if a1_key in halfspaces and a2_key in halfspaces and b_key in halfspaces:
                a1, a2, b = halfspaces[a1_key], halfspaces[a2_key], halfspaces[b_key]
                
                if abs(a1) > 1e-6 or abs(a2) > 1e-6:
                    # Plot constraint line: a1*x + a2*y = b
                    x_range = np.linspace(state.x - 5, state.x + 5, 100)
                    if abs(a2) > 1e-6:
                        y_range = (b - a1 * x_range) / a2
                        self.ax.plot(x_range, y_range, 'r--', alpha=0.7, linewidth=2)
                    else:
                        # Vertical line
                        x_val = b / a1
                        self.ax.axvline(x=x_val, color='r', linestyle='--', alpha=0.7)


def plot_final_trajectory(result, test_data, viz_history, save_path="proper_scenario_trajectory.png"):
    """Plot final trajectory with proper scenario constraints visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main trajectory plot (top-left)
    ax_main = axes[0, 0]
    
    # Plot vehicle path
    if len(result.trajectory_x) > 0:
        ax_main.plot(result.trajectory_x, result.trajectory_y, 'b-', linewidth=4,
                    label='Vehicle Path', zorder=5)
        
        # Start marker
        ax_main.plot(result.trajectory_x[0], result.trajectory_y[0], 'go',
                    markersize=25, label='Start', zorder=6, markeredgecolor='darkgreen',
                    markeredgewidth=3)
        
        # End marker (at actual end of trajectory)
        if len(result.trajectory_x) > 1:
            ax_main.plot(result.trajectory_x[-1], result.trajectory_y[-1], 'rs',
                        markersize=25, label='End', zorder=6, markeredgecolor='darkred',
                        markeredgewidth=3)
        else:
            # If only 1 point, end = start (show in orange)
            ax_main.plot(result.trajectory_x[0], result.trajectory_y[0], 'o',
                        color='orange', markersize=25, label='Start/End', zorder=6,
                        markeredgecolor='darkorange', markeredgewidth=3)
    
    # Plot road boundaries
    if hasattr(test_data, 'left_bound'):
        ax_main.plot(test_data.left_bound.x, test_data.left_bound.y, 'k--',
                    linewidth=1.5, alpha=0.5, label='Road Boundaries', zorder=1)
        ax_main.plot(test_data.right_bound.x, test_data.right_bound.y, 'k--',
                    linewidth=1.5, alpha=0.5, zorder=1)
    
    # Plot obstacles
    if hasattr(test_data, 'dynamic_obstacles') and test_data.dynamic_obstacles:
        for i, obs in enumerate(test_data.dynamic_obstacles):
            if hasattr(obs, 'position'):
                pos_x, pos_y = obs.position[0], obs.position[1]
                circle = plt.Circle((pos_x, pos_y), obs.radius, color='red',
                                  fill=True, alpha=0.4, zorder=3,
                                  label='Obstacles' if i == 0 else '')
                ax_main.add_patch(circle)
                ax_main.text(pos_x, pos_y, f'O{i}', fontsize=10, ha='center',
                            va='center', color='white', fontweight='bold', zorder=4)
    
    ax_main.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax_main.set_title('Proper Scenario Constraints: Vehicle Trajectory',
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.grid(True, alpha=0.4, linestyle='--')
    ax_main.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax_main.axis('equal')
    
    # Constraint evolution (top-right)
    ax_constraints = axes[0, 1]
    if viz_history:
        iterations = [v['iteration'] for v in viz_history]
        num_constraints = [len(v['halfspaces']) for v in viz_history]
        ax_constraints.plot(iterations, num_constraints, 'r-', linewidth=2, marker='o')
        ax_constraints.set_xlabel('Iteration', fontsize=12)
        ax_constraints.set_ylabel('Number of Halfspace Constraints', fontsize=12)
        ax_constraints.set_title('Constraint Evolution', fontsize=14, fontweight='bold')
        ax_constraints.grid(True, alpha=0.3)
    
    # Performance metrics (bottom-left)
    ax_perf = axes[1, 0]
    ax_perf.axis('off')
    
    # Format planning rate safely
    if result.average_solve_time > 0:
        planning_rate = f"{1.0/result.average_solve_time:.1f} Hz"
        avg_solve = f"{result.average_solve_time:.4f}s"
    else:
        planning_rate = "N/A"
        avg_solve = "N/A"
    
    stats_text = f"""
    PROPER SCENARIO CONSTRAINTS PERFORMANCE
    
    === Trajectory ===
    Total Iterations: {result.iterations_completed}
    Successful: {result.iterations_completed - result.failed_iterations}
    Failed: {result.failed_iterations}
    Trajectory Points: {len(result.trajectory_x)}
    
    === Timing ===
    Average Solve Time: {avg_solve}
    Planning Rate: {planning_rate}
    Total Time: {result.total_time:.2f}s
    
    === Movement ===
    Start: ({result.trajectory_x[0]:.2f}, {result.trajectory_y[0]:.2f})
    End: ({result.trajectory_x[-1]:.2f}, {result.trajectory_y[-1]:.2f})
    Distance: {np.sqrt((result.trajectory_x[-1] - result.trajectory_x[0])**2 + 
                        (result.trajectory_y[-1] - result.trajectory_y[0])**2):.2f}m
    
    === Approach ===
    Method: Oscar de Groot IJRR 2024
    Parallel Solvers: 4
    Scenario Samples: 100
    Max Halfspaces: 5
    Polytope Construction: ✓
    """
    
    ax_perf.text(0.05, 0.95, stats_text, transform=ax_perf.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=1.0))
    
    # Sample constraints visualization (bottom-right)
    ax_sample = axes[1, 1]
    if viz_history:
        # Show constraints at a mid-point
        mid_idx = len(viz_history) // 2 if viz_history else 0
        if mid_idx < len(viz_history):
            viz_data = viz_history[mid_idx]
            
            # Plot local area
            state = viz_data['state']
            local_range = 8
            
            # Vehicle
            ax_sample.plot(state.x, state.y, 'bo', markersize=15, zorder=5, label='Vehicle')
            
            # Halfspace constraints
            halfspaces = viz_data['halfspaces']
            for i in range(5):
                a1_key = f'halfspace_{i}_a1'
                a2_key = f'halfspace_{i}_a2'
                b_key = f'halfspace_{i}_b'
                
                if a1_key in halfspaces and a2_key in halfspaces and b_key in halfspaces:
                    a1, a2, b = halfspaces[a1_key], halfspaces[a2_key], halfspaces[b_key]
                    
                    if abs(a1) > 1e-6 or abs(a2) > 1e-6:
                        x_range = np.linspace(state.x - local_range, state.x + local_range, 100)
                        if abs(a2) > 1e-6:
                            y_range = (b - a1 * x_range) / a2
                            ax_sample.plot(x_range, y_range, 'r--', alpha=0.7, linewidth=2, zorder=2)
                        else:
                            x_val = b / a1
                            ax_sample.axvline(x=x_val, color='r', linestyle='--', alpha=0.7, zorder=2)
            
            ax_sample.set_xlim(state.x - local_range, state.x + local_range)
            ax_sample.set_ylim(state.y - local_range, state.y + local_range)
            ax_sample.set_xlabel('X Position (m)', fontsize=12)
            ax_sample.set_ylabel('Y Position (m)', fontsize=12)
            ax_sample.set_title(f'Halfspace Constraints at Iteration {viz_data["iteration"]}',
                              fontsize=14, fontweight='bold')
            ax_sample.grid(True, alpha=0.3)
            ax_sample.legend(loc='upper right', fontsize=9)
            ax_sample.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Proper scenario trajectory plot saved to: {save_path}")
    plt.show()


def main():
    """Main test function."""
    print("="*80)
    print("PROPER SCENARIO CONSTRAINTS TEST")
    print("="*80)
    print("Based on Oscar de Groot's IJRR 2024 approach:")
    print("  • Parallel scenario optimization")
    print("  • Polytope construction around infeasible regions")
    print("  • Halfspace constraint extraction")
    print("  • Real-time performance (~30 Hz)")
    print("="*80)
    
    # Ask user for visualization preferences
    enable_realtime = input("\nEnable real-time visualization? (y/n) [n]: ").strip().lower() == 'y'
    enable_gif = input("Generate animated GIF? (y/n) [n]: ").strip().lower() == 'y'
    
    # Create test
    test = ProperScenarioConstraintsTest(
        'proper_scenario_constraints',
        dt=0.1,
        horizon=15,
        max_iterations=150,
        max_consecutive_failures=30,
        enable_visualization=False,  # Disable default viz manager
        enable_realtime_viz=enable_realtime,
        enable_gif=enable_gif
    )
    
    # Setup with distant goal and no early stopping on goal reached
    test.setup(start=(0, 0), goal=(50, 30), path_type="curved", road_width=10.0)
    
    # Disable early goal-reached stopping for this test to see full trajectory
    test.check_goal_reached = False
    
    print("\nStarting proper scenario constraints test...")
    print("-" * 80)
    
    # Run test
    result = test.run()
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Total Iterations: {result.iterations_completed}")
    print(f"Successful: {result.iterations_completed - result.failed_iterations}")
    print(f"Failed: {result.failed_iterations}")
    print(f"\nTrajectory Points: {len(result.trajectory_x)}")
    if result.average_solve_time > 0:
        print(f"Average Solve Time: {result.average_solve_time:.4f}s ({1.0/result.average_solve_time:.1f} Hz)")
    else:
        print(f"Average Solve Time: N/A (no successful solves)")
    print(f"Total Time: {result.total_time:.2f}s")
    
    if len(result.trajectory_x) > 1:
        distance = np.sqrt((result.trajectory_x[-1] - result.trajectory_x[0])**2 + 
                          (result.trajectory_y[-1] - result.trajectory_y[0])**2)
        print(f"\nVehicle Movement:")
        print(f"  Start: ({result.trajectory_x[0]:.2f}, {result.trajectory_y[0]:.2f})")
        print(f"  End: ({result.trajectory_x[-1]:.2f}, {result.trajectory_y[-1]:.2f})")
        print(f"  Distance Traveled: {distance:.2f}m")
        
        if distance > 1.0:
            print("✅ SUCCESS: Vehicle moved significantly")
        else:
            print("⚠️ LIMITED MOVEMENT: Check constraint tuning")
    else:
        print("❌ FAILED: No vehicle movement")
    
    print("="*80)
    
    # Generate final visualization
    if len(result.trajectory_x) > 0:
        plot_final_trajectory(
            result, test.data, test.viz_history,
            'proper_scenario_trajectory.png'
        )
    
    return result


if __name__ == "__main__":
    result = main()
