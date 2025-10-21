#!/usr/bin/env python3
"""
Manual path following test that creates a trajectory that actually follows the path.

This test manually creates a trajectory that follows the curved path to demonstrate
that the visualization system works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
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

def create_path_following_trajectory(reference_path, num_steps=25):
    """Create a trajectory that actually follows the reference path."""
    trajectory = []
    vehicle_states = []
    control_inputs = []
    objective_values = []
    solve_times = []
    
    for step in range(num_steps):
        # Get reference point for this step
        if step < len(reference_path):
            ref_point = reference_path[step]
        else:
            ref_point = reference_path[-1]
        
        # Create vehicle state that follows the path
        x = ref_point[0]
        y = ref_point[1]
        
        # Calculate heading to follow the path
        if step < len(reference_path) - 1:
            next_point = reference_path[step + 1]
            dx = next_point[0] - ref_point[0]
            dy = next_point[1] - ref_point[1]
            theta = np.arctan2(dy, dx)
        else:
            theta = 0.0
        
        # Vehicle state: [x, y, theta, v, delta]
        vehicle_state = np.array([x, y, theta, 1.0, 0.0])
        
        # Control input: [acceleration, steering_rate]
        control_input = np.array([0.1, 0.0])
        
        # Simulate some objective value and solve time
        objective_value = 10.0 + step * 0.5
        solve_time = 0.01 + step * 0.001
        
        trajectory.append([x, y])
        vehicle_states.append(vehicle_state)
        control_inputs.append(control_input)
        objective_values.append(objective_value)
        solve_times.append(solve_time)
    
    return trajectory, vehicle_states, control_inputs, objective_values, solve_times

def run_manual_path_following():
    """Run manual path following test."""
    print("=" * 60)
    print("MANUAL PATH FOLLOWING TEST")
    print("=" * 60)
    
    # Create logger
    logger = MPCLogger()
    logger.log_info("Starting manual path following test")
    
    # Create scenario
    reference_path = create_curved_path()
    obstacles = []  # No obstacles for simplicity
    
    logger.log_info(f"Created curved path: y = 0.1 * x^2")
    logger.log_info(f"Path goes from ({reference_path[0,0]:.1f}, {reference_path[0,1]:.1f}) to ({reference_path[-1,0]:.1f}, {reference_path[-1,1]:.1f})")
    
    # Create trajectory that follows the path
    trajectory, vehicle_states, control_inputs, objective_values, solve_times = create_path_following_trajectory(reference_path, 25)
    
    logger.log_info(f"Created trajectory with {len(trajectory)} points")
    logger.log_info(f"Trajectory goes from ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f}) to ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f})")
    
    # Initialize real-time visualizer
    visualizer = RealtimeVisualizer(
        figsize=(16, 12),
        fps=8,
        save_dir="manual_path_following_plots"
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
    
    # Simulate the trajectory
    logger.log_info("Starting manual path following simulation")
    
    for step in range(len(trajectory)):
        logger.log_info(f"Step {step+1}/{len(trajectory)}")
        
        # Get current state
        current_state = vehicle_states[step]
        control_input = control_inputs[step]
        objective_value = objective_values[step]
        solve_time = solve_times[step]
        
        # Update visualizer
        visualizer.update_frame(
            vehicle_state=current_state,
            control_input=control_input,
            trajectory=np.array(trajectory[:step+1]),
            objective_value=objective_value,
            solve_time=solve_time,
            timestamp=time.time()
        )
        
        logger.log_info(f"  ✅ Step {step+1}: Objective={objective_value:.3f}, "
                      f"Solve time={solve_time:.3f}s, "
                      f"Position=({current_state[0]:.2f}, {current_state[1]:.2f})")
        
        # Small delay for visualization
        time.sleep(0.1)
    
    logger.log_info("Manual path following simulation completed")
    
    # Create animation
    logger.log_info("Creating manual path following animation GIF...")
    gif_path = visualizer.start_animation(
        total_frames=len(trajectory),
        save_gif=True,
        gif_filename="manual_path_following_animation.gif"
    )
    
    logger.log_info(f"Manual path following animation saved: {gif_path}")
    
    # Export data
    data_path = visualizer.export_data("manual_path_following_data.json")
    logger.log_info(f"Manual path following data exported: {data_path}")
    
    # Show final results
    print("\n" + "=" * 60)
    print("MANUAL PATH FOLLOWING RESULTS")
    print("=" * 60)
    print(f"Total steps: {len(trajectory)}")
    print(f"Final position: ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f})")
    print(f"Reference final position: ({reference_path[-1,0]:.2f}, {reference_path[-1,1]:.2f})")
    
    # Check path following performance
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
    logger.log_info("Displaying manual path following animation...")
    visualizer.show_animation()
    
    # Clean up
    visualizer.close()
    
    return gif_path, data_path, len(trajectory)

def main():
    """Main function to run manual path following test."""
    print("🚀 Starting Manual Path Following Test")
    print("This test manually creates a trajectory that follows the curved path")
    print("The vehicle should follow the curved path: y = 0.1 * x^2")
    print("This demonstrates that the visualization system works correctly")
    print()
    
    try:
        gif_path, data_path, successful_steps = run_manual_path_following()
        
        print("\n" + "=" * 60)
        print("🎉 MANUAL PATH FOLLOWING TEST COMPLETED!")
        print("=" * 60)
        print(f"✅ Successful steps: {successful_steps}")
        print(f"📁 Animation GIF: {gif_path}")
        print(f"📁 Data export: {data_path}")
        
        if successful_steps > 0:
            print("\nYou can now watch the GIF to see the vehicle following the curved path!")
            print("This demonstrates that the visualization system works correctly.")
            return True
        else:
            print("\n❌ No successful path following achieved")
            return False
        
    except Exception as e:
        print(f"\n❌ Manual path following test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
