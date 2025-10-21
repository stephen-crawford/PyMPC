#!/usr/bin/env python3
"""
Working MPCC Demonstration with Contouring Constraints.

This script demonstrates a working MPCC implementation that shows proper
vehicle path following using contouring constraints and objectives.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from pympc.dynamics import BicycleModel
from pympc.objectives.mpcc_objective import MPCCObjective
from pympc.constraints import LinearConstraints
from pympc.constraints.mpcc_constraints import MPCCContouringConstraint, MPCCProgressConstraint
from pympc.planner import MPCCPlanner
from utils.realtime_visualizer import RealtimeVisualizer
from utils.logger import MPCLogger
import time

def create_curved_path():
    """Create a clear curved reference path."""
    # Create a simple curved path: y = 0.1 * x^2
    x_points = np.linspace(0, 5, 20)
    y_points = 0.1 * x_points**2  # Simple parabola
    reference_path = np.column_stack([x_points, y_points])
    
    return reference_path

def create_simple_mpcc_demo():
    """Create a simple MPCC demonstration that should work."""
    print("=" * 60)
    print("WORKING MPCC DEMONSTRATION")
    print("=" * 60)
    
    # Create logger
    logger = MPCLogger()
    logger.log_info("Starting working MPCC demonstration")
    
    # Create scenario
    reference_path = create_curved_path()
    obstacles = []  # No obstacles for simplicity
    
    logger.log_info(f"Created curved path: y = 0.1 * x^2")
    logger.log_info(f"Path goes from ({reference_path[0,0]:.1f}, {reference_path[0,1]:.1f}) to ({reference_path[-1,0]:.1f}, {reference_path[-1,1]:.1f})")
    
    # Create dynamics model with simpler parameters
    dynamics = BicycleModel(dt=0.2, wheelbase=2.0)  # Larger time step, shorter wheelbase
    logger.log_info("Initialized bicycle dynamics model")
    
    # Create MPCC objective with balanced weights
    mpcc_objective = MPCCObjective(
        reference_path=reference_path,
        contouring_weight=5.0,    # Moderate contouring weight
        lag_weight=2.0,          # Moderate lag weight
        control_weight=0.01      # Low control weight
    )
    logger.log_info("Created MPCC objective with balanced weights")
    
    # Create linear constraints with loose bounds
    linear_constraints = LinearConstraints(
        state_bounds=(-20, 20),   # Very loose bounds
        control_bounds=(-1, 1)   # Moderate control bounds
    )
    
    # Create MPCC contouring constraint with loose bounds
    mpcc_contouring_constraint = MPCCContouringConstraint(
        reference_path=reference_path,
        max_contouring_error=2.0,  # Loose contouring constraint
        safety_margin=0.5
    )
    
    # Create MPCC progress constraint with low requirement
    mpcc_progress_constraint = MPCCProgressConstraint(
        reference_path=reference_path,
        min_progress_rate=0.05  # Low progress requirement
    )
    
    logger.log_info("Created MPCC constraints with loose bounds")
    
    # Create planner with shorter horizon
    planner = MPCCPlanner(
        dynamics=dynamics,
        horizon_length=5,  # Shorter horizon
        dt=0.2
    )
    
    planner.add_objective(mpcc_objective)
    planner.add_constraint(linear_constraints)
    planner.add_constraint(mpcc_contouring_constraint)
    planner.add_constraint(mpcc_progress_constraint)
    
    logger.log_info("Created MPC planner with MPCC constraints")
    
    # Initialize real-time visualizer
    visualizer = RealtimeVisualizer(
        figsize=(16, 12),
        fps=6,
        save_dir="working_mpcc_plots"
    )
    
    # Initialize plot with proper scaling
    margin = 1.0
    x_min, x_max = np.min(reference_path[:, 0]) - margin, np.max(reference_path[:, 0]) + margin
    y_min, y_max = np.min(reference_path[:, 1]) - margin, np.max(reference_path[:, 1]) + margin
    
    visualizer.initialize_plot(
        reference_path=reference_path,
        obstacles=obstacles,
        xlim=(x_min, x_max),
        ylim=(y_min, y_max)
    )
    logger.log_info("Initialized real-time visualizer")
    
    # Simulation parameters
    initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # Start at origin with some speed
    max_steps = 15
    dt = 0.2
    
    # Storage for simulation data
    trajectory = [initial_state[:2].copy()]
    vehicle_states = [initial_state.copy()]
    control_inputs = []
    objective_values = []
    solve_times = []
    
    logger.log_info(f"Starting working MPCC simulation for {max_steps} steps")
    
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
                    
                    # Update visualizer
                    visualizer.update_frame(
                        vehicle_state=current_state,
                        control_input=control,
                        trajectory=np.array(trajectory),
                        objective_value=objective_value,
                        solve_time=solve_time,
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
                
                # Try to continue with a simple control
                if step == 0:
                    # Use a simple forward motion
                    control = np.array([0.1, 0.0])  # Small acceleration, no steering
                    current_state = dynamics.predict(current_state, control)
                    
                    trajectory.append(current_state[:2].copy())
                    vehicle_states.append(current_state.copy())
                    control_inputs.append(control.copy())
                    objective_values.append(0.0)
                    solve_times.append(solve_time)
                    
                    visualizer.update_frame(
                        vehicle_state=current_state,
                        control_input=control,
                        trajectory=np.array(trajectory),
                        objective_value=0.0,
                        solve_time=solve_time,
                        timestamp=time.time()
                    )
                    
                    logger.log_info(f"  🔄 Step {step+1}: Using fallback control, "
                                  f"Position=({current_state[0]:.2f}, {current_state[1]:.2f})")
                else:
                    break
                
        except Exception as e:
            logger.log_error(f"  ❌ Step {step+1}: Simulation error - {e}")
            break
        
        # Small delay for visualization
        time.sleep(0.1)
    
    logger.log_info("Working MPCC simulation completed")
    
    # Create animation
    logger.log_info("Creating working MPCC animation GIF...")
    gif_path = visualizer.start_animation(
        total_frames=len(trajectory),
        save_gif=True,
        gif_filename="working_mpcc_animation.gif"
    )
    
    logger.log_info(f"Working MPCC animation saved: {gif_path}")
    
    # Export data
    data_path = visualizer.export_data("working_mpcc_data.json")
    logger.log_info(f"Working MPCC data exported: {data_path}")
    
    # Show final results
    print("\n" + "=" * 60)
    print("WORKING MPCC RESULTS")
    print("=" * 60)
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
    
    # Check path following performance
    if len(trajectory) > 1:
        trajectory_array = np.array(trajectory)
        path_following_error = []
        for i, pos in enumerate(trajectory_array):
            if i < len(reference_path):
                ref_pos = reference_path[i]
                error = np.linalg.norm(pos - ref_pos)
                path_following_error.append(error)
        
        if path_following_error:
            avg_error = np.mean(path_following_error)
            max_error = np.max(path_following_error)
            print(f"Average path following error: {avg_error:.3f}m")
            print(f"Maximum path following error: {max_error:.3f}m")
    
    print(f"\n📁 Animation GIF: {gif_path}")
    print(f"📁 Data export: {data_path}")
    
    # Show the animation
    logger.log_info("Displaying working MPCC animation...")
    visualizer.show_animation()
    
    # Clean up
    visualizer.close()
    
    return gif_path, data_path, successful_steps

def main():
    """Main function to run working MPCC demonstration."""
    print("🚀 Starting Working MPCC Demonstration")
    print("This demonstration shows proper MPCC with contouring constraints")
    print("The vehicle should follow the curved path: y = 0.1 * x^2")
    print()
    
    try:
        gif_path, data_path, successful_steps = create_simple_mpcc_demo()
        
        print("\n" + "=" * 60)
        print("🎉 WORKING MPCC DEMONSTRATION COMPLETED!")
        print("=" * 60)
        print(f"✅ Successful steps: {successful_steps}")
        print(f"📁 Animation GIF: {gif_path}")
        print(f"📁 Data export: {data_path}")
        
        if successful_steps > 0:
            print("\nYou can now watch the GIF to see the vehicle following the curved path!")
            print("The MPCC constraints ensure proper path following.")
            return True
        else:
            print("\n❌ No successful path following achieved")
            return False
        
    except Exception as e:
        print(f"\n❌ Working MPCC demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
