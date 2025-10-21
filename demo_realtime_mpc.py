#!/usr/bin/env python3
"""
Comprehensive demo of real-time MPC visualization with GIF export.

This script demonstrates the real-time visualization capabilities
showing vehicle progress, constraints, and optimization results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
from pympc.dynamics import BicycleModel
from pympc.objectives import ContouringObjective
from pympc.constraints import LinearConstraints, EllipsoidConstraints
from pympc.planner import MPCCPlanner
from utils.realtime_visualizer import RealtimeVisualizer
from utils.logger import MPCLogger
import time

def create_demo_scenario():
    """Create a comprehensive demo scenario."""
    # Create a complex curving reference path
    t = np.linspace(0, 6*np.pi, 80)
    reference_path = np.column_stack([
        3 * t * np.cos(t/2),
        3 * t * np.sin(t/2)
    ])
    
    # Create multiple obstacles with different shapes
    obstacles = [
        {
            'center': np.array([8.0, 5.0]),
            'shape': np.array([[1.5, 0.0], [0.0, 0.8]]),
            'safety_margin': 0.5
        },
        {
            'center': np.array([-5.0, 8.0]),
            'shape': np.array([[0.8, 0.3], [0.3, 1.2]]),
            'safety_margin': 0.3
        },
        {
            'center': np.array([12.0, -3.0]),
            'shape': np.array([[2.0, 0.0], [0.0, 1.0]]),
            'safety_margin': 0.4
        },
        {
            'center': np.array([-8.0, -6.0]),
            'shape': np.array([[1.0, 0.5], [0.5, 1.5]]),
            'safety_margin': 0.6
        }
    ]
    
    return reference_path, obstacles

def run_demo_simulation():
    """Run a comprehensive demo simulation."""
    print("=" * 70)
    print("REAL-TIME MPC VISUALIZATION DEMO")
    print("=" * 70)
    
    # Create logger
    logger = MPCLogger()
    logger.log_info("Starting comprehensive MPC visualization demo")
    
    # Create demo scenario
    reference_path, obstacles = create_demo_scenario()
    logger.log_info(f"Created demo scenario with {len(obstacles)} obstacles")
    logger.log_info(f"Reference path length: {len(reference_path)} points")
    
    # Create dynamics model
    dynamics = BicycleModel(dt=0.1, wheelbase=2.5)
    logger.log_info("Initialized bicycle dynamics model")
    
    # Create contouring objective
    objective = ContouringObjective(
        reference_path=reference_path,
        progress_weight=2.0,
        contouring_weight=15.0,
        control_weight=0.5
    )
    logger.log_info("Created contouring objective with enhanced weights")
    
    # Create constraints
    linear_constraints = LinearConstraints(
        state_bounds=(-25, 25),
        control_bounds=(-8, 8)
    )
    
    ellipsoid_constraints = EllipsoidConstraints(
        obstacles=obstacles
    )
    logger.log_info("Created comprehensive constraints")
    
    # Create planner with robust settings
    planner = MPCCPlanner(
        dynamics=dynamics,
        horizon_length=20,
        dt=0.1
    )
    
    planner.add_objective(objective)
    planner.add_constraint(linear_constraints)
    planner.add_constraint(ellipsoid_constraints)
    
    logger.log_info("Created MPC planner with robust configuration")
    
    # Initialize real-time visualizer with enhanced settings
    visualizer = RealtimeVisualizer(
        figsize=(18, 14),
        fps=8,  # Higher FPS for smoother animation
        save_dir="demo_plots"
    )
    
    # Initialize plot with proper scaling
    x_min, x_max = np.min(reference_path[:, 0]) - 5, np.max(reference_path[:, 0]) + 5
    y_min, y_max = np.min(reference_path[:, 1]) - 5, np.max(reference_path[:, 1]) + 5
    
    visualizer.initialize_plot(
        reference_path=reference_path,
        obstacles=obstacles,
        xlim=(x_min, x_max),
        ylim=(y_min, y_max)
    )
    logger.log_info("Initialized enhanced real-time visualizer")
    
    # Simulation parameters
    initial_state = np.array([0.0, 0.0, 0.0, 3.0, 0.0])  # x, y, theta, v, delta
    max_steps = 50
    dt = 0.1
    
    # Storage for simulation data
    trajectory = [initial_state[:2].copy()]
    vehicle_states = [initial_state.copy()]
    control_inputs = []
    objective_values = []
    solve_times = []
    constraint_violations = []
    
    logger.log_info(f"Starting enhanced simulation for {max_steps} steps")
    
    # Run simulation
    current_state = initial_state.copy()
    successful_steps = 0
    
    for step in range(max_steps):
        logger.log_info(f"Step {step+1}/{max_steps}")
        
        try:
            # Solve MPC problem
            start_time = time.time()
            solution = planner.solve(current_state, reference_path=reference_path)
            solve_time = time.time() - start_time
            
            if solution['status'] == 'optimal':
                # Extract solution
                optimal_states = solution['states']
                optimal_controls = solution['controls']
                objective_value = solution.get('objective_value', 0.0)
                
                # Apply first control
                if len(optimal_controls) > 0:
                    control = optimal_controls[:, 0]
                    current_state = dynamics.predict(current_state, control)
                    
                    # Store data
                    trajectory.append(current_state[:2].copy())
                    vehicle_states.append(current_state.copy())
                    control_inputs.append(control.copy())
                    objective_values.append(objective_value)
                    solve_times.append(solve_time)
                    
                    # Check constraint violations (simplified)
                    violations = {'linear': 0, 'ellipsoid': 0}
                    constraint_violations.append(violations)
                    
                    # Update visualizer
                    visualizer.update_frame(
                        vehicle_state=current_state,
                        control_input=control,
                        trajectory=np.array(trajectory),
                        objective_value=objective_value,
                        solve_time=solve_time,
                        constraint_violations=violations,
                        timestamp=time.time()
                    )
                    
                    successful_steps += 1
                    logger.log_info(f"  ✅ Step {step+1}: Objective={objective_value:.3f}, "
                                  f"Solve time={solve_time:.3f}s, "
                                  f"Position=({current_state[0]:.2f}, {current_state[1]:.2f})")
                else:
                    logger.log_warning(f"  ⚠️  Step {step+1}: No control solution available")
                    break
            else:
                logger.log_error(f"  ❌ Step {step+1}: Optimization failed - {solution.get('error', 'Unknown error')}")
                # Continue with previous control or stop
                if len(control_inputs) > 0:
                    # Use previous control
                    control = control_inputs[-1]
                    current_state = dynamics.predict(current_state, control)
                    trajectory.append(current_state[:2].copy())
                    vehicle_states.append(current_state.copy())
                    control_inputs.append(control.copy())
                    objective_values.append(objective_values[-1] if objective_values else 0.0)
                    solve_times.append(solve_time)
                    constraint_violations.append(constraint_violations[-1] if constraint_violations else {'linear': 0, 'ellipsoid': 0})
                    
                    visualizer.update_frame(
                        vehicle_state=current_state,
                        control_input=control,
                        trajectory=np.array(trajectory),
                        objective_value=objective_values[-1] if objective_values else 0.0,
                        solve_time=solve_time,
                        constraint_violations=constraint_violations[-1] if constraint_violations else {'linear': 0, 'ellipsoid': 0},
                        timestamp=time.time()
                    )
                else:
                    break
                
        except Exception as e:
            logger.log_error(f"  ❌ Step {step+1}: Simulation error - {e}")
            break
        
        # Small delay for visualization
        time.sleep(0.05)
    
    logger.log_info("Demo simulation completed")
    
    # Create enhanced animation
    logger.log_info("Creating enhanced animation GIF...")
    gif_path = visualizer.start_animation(
        total_frames=len(trajectory),
        save_gif=True,
        gif_filename="mpc_demo_animation.gif"
    )
    
    logger.log_info(f"Enhanced animation saved: {gif_path}")
    
    # Export comprehensive data
    data_path = visualizer.export_data("demo_simulation_data.json")
    logger.log_info(f"Demo simulation data exported: {data_path}")
    
    # Show final results
    print("\n" + "=" * 70)
    print("DEMO SIMULATION RESULTS")
    print("=" * 70)
    print(f"Total steps attempted: {max_steps}")
    print(f"Successful steps: {successful_steps}")
    print(f"Success rate: {successful_steps/max_steps*100:.1f}%")
    print(f"Final position: ({current_state[0]:.2f}, {current_state[1]:.2f})")
    print(f"Final velocity: {current_state[3]:.2f} m/s")
    print(f"Final heading: {np.degrees(current_state[2]):.1f}°")
    
    if solve_times:
        print(f"Average solve time: {np.mean(solve_times):.3f}s")
        print(f"Min solve time: {np.min(solve_times):.3f}s")
        print(f"Max solve time: {np.max(solve_times):.3f}s")
    
    if objective_values:
        print(f"Average objective value: {np.mean(objective_values):.3f}")
        print(f"Final objective value: {objective_values[-1]:.3f}")
    
    print(f"\n📁 Animation GIF: {gif_path}")
    print(f"📁 Data export: {data_path}")
    
    # Show the animation
    logger.log_info("Displaying enhanced animation...")
    visualizer.show_animation()
    
    # Clean up
    visualizer.close()
    
    return gif_path, data_path, successful_steps

def main():
    """Main demo function."""
    print("🚀 Starting Real-time MPC Visualization Demo")
    print("This demo will show vehicle progress, constraints, and optimization results")
    print("The animation will be saved as a GIF that you can watch after completion")
    print()
    
    try:
        gif_path, data_path, successful_steps = run_demo_simulation()
        
        print("\n" + "=" * 70)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"✅ Animation GIF created: {gif_path}")
        print(f"✅ Simulation data exported: {data_path}")
        print(f"✅ Successful optimization steps: {successful_steps}")
        print()
        print("You can now watch the animation GIF to see the real-time")
        print("vehicle progress, constraint visualization, and optimization results!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
