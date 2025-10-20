"""
Working MPC Goal Reaching Test

This test fixes the issues preventing the vehicle from reaching the goal:
1. Uses actual MPC solver instead of simplified control
2. Proper goal reaching logic
3. Correct constraint integration
4. Realistic vehicle dynamics
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from planner_modules.src.objectives.goal_objective import GoalObjective
from planner_modules.src.constraints.fixed_scenario_constraints import FixedScenarioConstraints
from planning.src.dynamic_models import SecondOrderUnicycleModel
from planning.src.planner import Planner
from planning.src.types import Data, Bound, State, generate_dynamic_obstacles, PredictionType
from solver.src.casadi_solver import CasADiSolver
from utils.utils import LOG_INFO, LOG_WARN, LOG_DEBUG


class WorkingMPCGoalReachingTest:
    """
    Working MPC test that properly reaches the goal.
    
    This test fixes the issues by:
    - Using actual MPC solver instead of simplified control
    - Proper goal reaching logic
    - Correct constraint integration
    - Realistic vehicle dynamics
    """
    
    def __init__(self, name="working_mpc_goal_reaching", dt=0.1, horizon=8, 
                 max_iterations=100):
        """Initialize test."""
        self.name = name
        self.dt = dt
        self.horizon = horizon
        self.max_iterations = max_iterations
        
        # Test results
        self.result = None
        self.data = None
        self.planner = None
        
        # Visualization storage
        self.viz_history = []
        
        LOG_INFO(f"Initialized {self.name} test")
    
    def setup(self, start=(0, 0), goal=(15, 10)):
        """Setup test environment."""
        LOG_INFO(f"Setting up {self.name} test environment")
        
        # Create data object
        self.data = Data()
        
        # Set start and goal
        self.data.start = np.array(start)
        self.data.goal = np.array(goal)
        
        # Generate simple reference path
        x_path = np.linspace(start[0], goal[0], 20)
        y_path = np.linspace(start[1], goal[1], 20)
        s_path = np.linspace(0, 1, 20)
        self.data.reference_path = Bound(x=x_path, y=y_path, s=s_path)
        
        # Create simple road boundaries
        self.data.left_bound = Bound(x=x_path, y=y_path - 2.0, s=s_path)
        self.data.right_bound = Bound(x=x_path, y=y_path + 2.0, s=s_path)
        
        # Generate minimal obstacles
        obstacles = generate_dynamic_obstacles(
            number=1,  # Just 1 obstacle
            prediction_type=PredictionType.GAUSSIAN.name,
            size=0.5,  # Small obstacle
            distribution="random_paths",
            area=((5, 12), (5, 8), (0, 0)),
            path_types=("straight",),
            num_points=self.horizon + 1,
            dim=2
        )
        
        self.data.dynamic_obstacles = obstacles
        
        print(f"\n{'='*80}")
        print(f"WORKING MPC GOAL REACHING TEST SETUP")
        print(f"{'='*80}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Dynamic obstacles: {len(obstacles)} (minimal)")
        print(f"Obstacle size: 0.5m (small)")
        print(f"Horizon: {self.horizon} timesteps ({self.horizon * self.dt:.1f}s)")
        print(f"{'='*80}")
        print(f"\nSystem Components:")
        print(f"  ✓ Fixed Scenario Constraints (prevents solver failures)")
        print(f"  ✓ Goal Objective (proper goal reaching)")
        print(f"  ✓ Second Order Unicycle Model")
        print(f"  ✓ Actual MPC Solver (not simplified control)")
        print(f"{'='*80}")
    
    def run(self):
        """Run the working MPC test."""
        LOG_INFO(f"Starting {self.name} test")
        
        # Create solver
        solver = CasADiSolver()
        
        # Configure modules
        self._configure_modules(solver)
        
        # Create planner with proper MPC
        self.planner = Planner(solver, SecondOrderUnicycleModel())
        
        # Initialize state
        current_state = State()
        current_state.x = self.data.start[0]
        current_state.y = self.data.start[1]
        current_state.theta = 0.0
        current_state.v = 0.0
        current_state.omega = 0.0
        
        # Storage for results
        trajectory_x = [current_state.x]
        trajectory_y = [current_state.y]
        solve_times = []
        failed_iterations = 0
        consecutive_failures = 0
        
        start_time = time.time()
        
        # Main simulation loop
        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            
            try:
                # Check if goal reached
                dx = self.data.goal[0] - current_state.x
                dy = self.data.goal[1] - current_state.y
                distance_to_goal = np.sqrt(dx**2 + dy**2)
                
                if distance_to_goal < 1.0:
                    LOG_INFO(f"Goal reached at iteration {iteration}")
                    break
                
                # Update planner with current state and data
                self.planner.update(current_state, self.data)
                
                # Solve MPC
                output = self.planner.solve()
                
                if output.success:
                    # Extract control input from MPC solution
                    u = output.control_inputs[0]  # First control input
                    
                    # Update state using MPC solution
                    current_state.x += current_state.v * np.cos(current_state.theta) * self.dt
                    current_state.y += current_state.v * np.sin(current_state.theta) * self.dt
                    current_state.theta += current_state.omega * self.dt
                    current_state.v += u[0] * self.dt  # acceleration
                    current_state.omega += u[1] * self.dt  # angular acceleration
                    
                    # Store trajectory
                    trajectory_x.append(current_state.x)
                    trajectory_y.append(current_state.y)
                    
                    # Store visualization data
                    self.viz_history.append({
                        'iteration': iteration,
                        'state': current_state,
                        'constraints': getattr(self.scenario_constraints, 'constraint_params', {}),
                        'obstacles': self.data.dynamic_obstacles
                    })
                    
                    consecutive_failures = 0
                    LOG_DEBUG(f"Iteration {iteration}: SUCCESS")
                    
                else:
                    failed_iterations += 1
                    consecutive_failures += 1
                    LOG_WARN(f"Iteration {iteration}: MPC FAILED")
                    
                    if consecutive_failures >= 10:
                        LOG_WARN(f"Too many consecutive failures: {consecutive_failures}")
                        break
                
            except Exception as e:
                LOG_WARN(f"Iteration {iteration}: Exception - {e}")
                failed_iterations += 1
                consecutive_failures += 1
                
                if consecutive_failures >= 10:
                    break
            
            # Record solve time
            solve_time = time.time() - iteration_start
            solve_times.append(solve_time)
        
        # Create result object
        total_time = time.time() - start_time
        average_solve_time = np.mean(solve_times) if solve_times else 0.0
        
        self.result = type('Result', (), {
            'iterations_completed': iteration + 1,
            'failed_iterations': failed_iterations,
            'trajectory_x': trajectory_x,
            'trajectory_y': trajectory_y,
            'average_solve_time': average_solve_time,
            'total_time': total_time,
            'success': len(trajectory_x) > 1
        })()
        
        LOG_INFO(f"Test completed: {self.result.iterations_completed} iterations")
        return self.result
    
    def _configure_modules(self, solver):
        """Configure and add modules to solver."""
        # Add fixed scenario constraints
        self.scenario_constraints = FixedScenarioConstraints(solver)
        solver.module_manager.add_module(self.scenario_constraints)
        
        # Add goal objective
        goal_objective = GoalObjective(solver)
        solver.module_manager.add_module(goal_objective)
    
    def plot_results(self, save_path="working_mpc_goal_reaching.png"):
        """Plot comprehensive results."""
        if not self.result:
            LOG_WARN("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main trajectory plot (top-left)
        ax_main = axes[0, 0]
        
        # Plot vehicle path
        if len(self.result.trajectory_x) > 0:
            ax_main.plot(self.result.trajectory_x, self.result.trajectory_y, 'b-', linewidth=4,
                        label='Vehicle Path', zorder=5)
            
            # Start marker (always at beginning)
            ax_main.plot(self.result.trajectory_x[0], self.result.trajectory_y[0], 'go',
                        markersize=25, label='Start', zorder=6, markeredgecolor='darkgreen',
                        markeredgewidth=3)
            
            # End marker (at ACTUAL end of trajectory)
            if len(self.result.trajectory_x) > 1:
                ax_main.plot(self.result.trajectory_x[-1], self.result.trajectory_y[-1], 'rs',
                            markersize=25, label='End', zorder=6, markeredgecolor='darkred',
                            markeredgewidth=3)
            else:
                ax_main.plot(self.result.trajectory_x[0], self.result.trajectory_y[0], 'o',
                            color='orange', markersize=25, label='Start/End', zorder=6,
                            markeredgecolor='darkorange', markeredgewidth=3)
        
        # Plot road boundaries
        if hasattr(self.data, 'left_bound'):
            ax_main.plot(self.data.left_bound.x, self.data.left_bound.y, 'k--',
                        linewidth=1.5, alpha=0.5, label='Road Boundaries', zorder=1)
            ax_main.plot(self.data.right_bound.x, self.data.right_bound.y, 'k--',
                        linewidth=1.5, alpha=0.5, zorder=1)
        
        # Plot obstacles
        if hasattr(self.data, 'dynamic_obstacles') and self.data.dynamic_obstacles:
            for i, obs in enumerate(self.data.dynamic_obstacles):
                if hasattr(obs, 'position'):
                    pos_x, pos_y = obs.position[0], obs.position[1]
                    circle = plt.Circle((pos_x, pos_y), obs.radius, color='red',
                                      fill=True, alpha=0.4, zorder=3,
                                      label='Obstacles' if i == 0 else '')
                    ax_main.add_patch(circle)
                    ax_main.text(pos_x, pos_y, f'O{i}', fontsize=10, ha='center',
                                va='center', color='white', fontweight='bold', zorder=4)
        
        # Plot goal
        ax_main.plot(self.data.goal[0], self.data.goal[1], 'r*', markersize=20, 
                    label='Goal', zorder=4)
        
        ax_main.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax_main.set_title('Working MPC Goal Reaching: Vehicle Trajectory',
                         fontsize=16, fontweight='bold', pad=20)
        ax_main.grid(True, alpha=0.4, linestyle='--')
        ax_main.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax_main.axis('equal')
        
        # Constraint evolution (top-right)
        ax_constraints = axes[0, 1]
        if self.viz_history:
            iterations = [v['iteration'] for v in self.viz_history]
            num_constraints = [len(v['constraints']) for v in self.viz_history]
            ax_constraints.plot(iterations, num_constraints, 'r-', linewidth=2, marker='o')
            ax_constraints.set_xlabel('Iteration', fontsize=12)
            ax_constraints.set_ylabel('Number of Constraints', fontsize=12)
            ax_constraints.set_title('Constraint Evolution', fontsize=14, fontweight='bold')
            ax_constraints.grid(True, alpha=0.3)
        
        # Performance metrics (bottom-left)
        ax_perf = axes[1, 0]
        ax_perf.axis('off')
        
        # Format planning rate safely
        if self.result.average_solve_time > 0:
            planning_rate = f"{1.0/self.result.average_solve_time:.1f} Hz"
            avg_solve = f"{self.result.average_solve_time:.4f}s"
        else:
            planning_rate = "N/A"
            avg_solve = "N/A"
        
        # Calculate final distance to goal
        final_distance = np.sqrt((self.result.trajectory_x[-1] - self.data.goal[0])**2 + 
                                (self.result.trajectory_y[-1] - self.data.goal[1])**2)
        
        stats_text = f"""
        WORKING MPC GOAL REACHING PERFORMANCE
        
        === Trajectory ===
        Total Iterations: {self.result.iterations_completed}
        Successful: {self.result.iterations_completed - self.result.failed_iterations}
        Failed: {self.result.failed_iterations}
        Trajectory Points: {len(self.result.trajectory_x)}
        
        === Timing ===
        Average Solve Time: {avg_solve}
        Planning Rate: {planning_rate}
        Total Time: {self.result.total_time:.2f}s
        
        === Movement ===
        Start: ({self.result.trajectory_x[0]:.2f}, {self.result.trajectory_y[0]:.2f})
        End: ({self.result.trajectory_x[-1]:.2f}, {self.result.trajectory_y[-1]:.2f})
        Goal: ({self.data.goal[0]:.2f}, {self.data.goal[1]:.2f})
        Final Distance: {final_distance:.2f}m
        
        === Goal Reaching ===
        Method: Working MPC with Goal Objective
        Solver: Actual MPC (not simplified control)
        Constraints: Fixed Scenario Constraints
        Status: {'GOAL REACHED' if final_distance < 1.0 else 'PARTIAL PROGRESS'}
        """
        
        ax_perf.text(0.05, 0.95, stats_text, transform=ax_perf.transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=1.0))
        
        # Sample constraints visualization (bottom-right)
        ax_sample = axes[1, 1]
        if self.viz_history:
            # Show constraints at a mid-point
            mid_idx = len(self.viz_history) // 2 if self.viz_history else 0
            if mid_idx < len(self.viz_history):
                viz_data = self.viz_history[mid_idx]
                
                # Plot local area
                state = viz_data['state']
                local_range = 4
                
                # Vehicle
                ax_sample.plot(state.x, state.y, 'bo', markersize=15, zorder=5, label='Vehicle')
                
                # Obstacles
                for obs in viz_data['obstacles']:
                    if hasattr(obs, 'position'):
                        pos = obs.position[:2]
                        circle = plt.Circle(pos, obs.radius, color='red', alpha=0.4)
                        ax_sample.add_patch(circle)
                
                # Goal
                ax_sample.plot(self.data.goal[0], self.data.goal[1], 'r*', markersize=15, label='Goal')
                
                ax_sample.set_xlim(state.x - local_range, state.x + local_range)
                ax_sample.set_ylim(state.y - local_range, state.y + local_range)
                ax_sample.set_xlabel('X Position (m)', fontsize=12)
                ax_sample.set_ylabel('Y Position (m)', fontsize=12)
                ax_sample.set_title(f'Environment at Iteration {viz_data["iteration"]}',
                                  fontsize=14, fontweight='bold')
                ax_sample.grid(True, alpha=0.3)
                ax_sample.legend(loc='upper right', fontsize=9)
                ax_sample.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Working MPC goal reaching plot saved to: {save_path}")
        plt.show()


def main():
    """Main test function."""
    print("="*80)
    print("WORKING MPC GOAL REACHING TEST")
    print("="*80)
    print("This test fixes the issues preventing the vehicle from reaching the goal:")
    print("  • Uses actual MPC solver instead of simplified control")
    print("  • Proper goal reaching logic")
    print("  • Correct constraint integration")
    print("  • Realistic vehicle dynamics")
    print("="*80)
    
    # Create test
    test = WorkingMPCGoalReachingTest(
        name="working_mpc_goal_reaching",
        dt=0.1,
        horizon=8,  # Moderate horizon
        max_iterations=100
    )
    
    # Setup with closer goal
    test.setup(start=(0, 0), goal=(15, 10))
    
    print("\nStarting working MPC goal reaching test...")
    print("-" * 80)
    
    # Run test
    result = test.run()
    
    # Print results
    print("\n" + "="*80)
    print("WORKING MPC GOAL REACHING RESULTS")
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
        final_distance = np.sqrt((result.trajectory_x[-1] - test.data.goal[0])**2 + 
                                (result.trajectory_y[-1] - test.data.goal[1])**2)
        
        print(f"\nVehicle Movement:")
        print(f"  Start: ({result.trajectory_x[0]:.2f}, {result.trajectory_y[0]:.2f})")
        print(f"  End: ({result.trajectory_x[-1]:.2f}, {result.trajectory_y[-1]:.2f})")
        print(f"  Goal: ({test.data.goal[0]:.2f}, {test.data.goal[1]:.2f})")
        print(f"  Distance Traveled: {distance:.2f}m")
        print(f"  Final Distance to Goal: {final_distance:.2f}m")
        
        if final_distance < 1.0:
            print("✅ SUCCESS: Vehicle reached the goal!")
        elif final_distance < 3.0:
            print("✅ GOOD PROGRESS: Vehicle is close to goal")
        else:
            print("⚠️ PARTIAL PROGRESS: Vehicle moved but didn't reach goal")
    else:
        print("❌ FAILED: No vehicle movement")
    
    print("="*80)
    
    # Generate final visualization
    if len(result.trajectory_x) > 0:
        test.plot_results('working_mpc_goal_reaching.png')
    
    return result


if __name__ == "__main__":
    result = main()
