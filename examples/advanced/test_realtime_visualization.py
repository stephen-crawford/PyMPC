#!/usr/bin/env python3
"""
Test script for real-time visualization of MPC vehicle progress.

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

def create_test_scenario():
    """Create a test scenario with obstacles and reference path."""
    # Create a curving reference path
    t = np.linspace(0, 4*np.pi, 50)
    reference_path = np.column_stack([
        2 * t * np.cos(t),
        2 * t * np.sin(t)
    ])
    
    # Create obstacles
    obstacles = [
        {
            'center': np.array([5.0, 3.0]),
            'shape': np.array([[1.0, 0.0], [0.0, 0.5]]),
            'safety_margin': 0.5
        },
        {
            'center': np.array([-3.0, 4.0]),
            'shape': np.array([[0.8, 0.2], [0.2, 0.6]]),
            'safety_margin': 0.3
        },
        {
            'center': np.array([8.0, -2.0]),
            'shape': np.array([[1.2, 0.0], [0.0, 0.8]]),
            'safety_margin': 0.4
        }
    ]
    
    return reference_path, obstacles

def run_realtime_simulation():
    """Run a real-time MPC simulation with visualization."""
    print("=" * 60)
    print("REAL-TIME MPC VISUALIZATION TEST")
    print("=" * 60)
    
    # Create logger
    logger = MPCLogger()
    logger.log_info("Starting real-time MPC visualization test")
    
    # Create test scenario
    reference_path, obstacles = create_test_scenario()
    logger.log_info(f"Created test scenario with {len(obstacles)} obstacles")
    
    # Create dynamics model
    dynamics = BicycleModel(dt=0.1)
    logger.log_info("Initialized bicycle dynamics model")
    
    # Create contouring objective
    objective = ContouringObjective(
        reference_path=reference_path,
        progress_weight=1.0,
        contouring_weight=10.0,
        control_weight=0.1
    )
    logger.log_info("Created contouring objective")
    
    # Create constraints
    linear_constraints = LinearConstraints(
        state_bounds=(-20, 20),
        control_bounds=(-5, 5)
    )
    
    ellipsoid_constraints = EllipsoidConstraints(
        obstacles=obstacles
    )
    logger.log_info("Created linear and ellipsoid constraints")
    
    # Create planner
    planner = MPCCPlanner(
        dynamics=dynamics,
        horizon_length=15,
        dt=0.1
    )
    
    planner.add_objective(objective)
    planner.add_constraint(linear_constraints)
    planner.add_constraint(ellipsoid_constraints)
    
    logger.log_info("Created MPC planner with constraints")
    
    # Initialize real-time visualizer
    visualizer = RealtimeVisualizer(
        figsize=(16, 12),
        fps=5,  # 5 FPS for smoother animation
        save_dir="realtime_plots"
    )
    
    # Initialize plot
    visualizer.initialize_plot(
        reference_path=reference_path,
        obstacles=obstacles,
        xlim=(-15, 15),
        ylim=(-15, 15)
    )
    logger.log_info("Initialized real-time visualizer")
    
    # Simulation parameters
    initial_state = np.array([0.0, 0.0, 0.0, 2.0, 0.0])  # x, y, theta, v, delta
    max_steps = 30
    dt = 0.1
    
    # Storage for simulation data
    trajectory = [initial_state[:2].copy()]  # Store x, y positions
    vehicle_states = [initial_state.copy()]
    control_inputs = []
    objective_values = []
    solve_times = []
    
    logger.log_info(f"Starting simulation for {max_steps} steps")
    
    # Run simulation
    current_state = initial_state.copy()
    
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
                    
                    logger.log_info(f"  Step {step+1}: Objective={objective_value:.3f}, "
                                  f"Solve time={solve_time:.3f}s, "
                                  f"Position=({current_state[0]:.2f}, {current_state[1]:.2f})")
                else:
                    logger.log_warning(f"  Step {step+1}: No control solution available")
                    break
            else:
                logger.log_error(f"  Step {step+1}: Optimization failed - {solution.get('error', 'Unknown error')}")
                break
                
        except Exception as e:
            logger.log_error(f"  Step {step+1}: Simulation error - {e}")
            break
        
        # Small delay for visualization
        time.sleep(0.1)
    
    logger.log_info("Simulation completed")
    
    # Create animation
    logger.log_info("Creating animation GIF...")
    gif_path = visualizer.start_animation(
        total_frames=len(trajectory),
        save_gif=True,
        gif_filename="mpc_realtime_animation.gif"
    )
    
    logger.log_info(f"Animation saved: {gif_path}")
    
    # Export data
    data_path = visualizer.export_data("simulation_data.json")
    logger.log_info(f"Simulation data exported: {data_path}")
    
    # Show final results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total steps completed: {len(trajectory)-1}")
    print(f"Final position: ({current_state[0]:.2f}, {current_state[1]:.2f})")
    print(f"Average solve time: {np.mean(solve_times):.3f}s")
    print(f"Average objective value: {np.mean(objective_values):.3f}")
    print(f"Animation GIF: {gif_path}")
    print(f"Data export: {data_path}")
    
    # Show the animation
    logger.log_info("Displaying animation...")
    visualizer.show_animation()
    
    # Clean up
    visualizer.close()
    
    return gif_path, data_path

def test_visualization_components():
    """Test individual visualization components."""
    print("\n" + "=" * 40)
    print("TESTING VISUALIZATION COMPONENTS")
    print("=" * 40)
    
    # Test 1: Basic visualizer initialization
    print("Testing visualizer initialization...")
    try:
        visualizer = RealtimeVisualizer()
        print("✅ Visualizer initialization successful")
    except Exception as e:
        print(f"❌ Visualizer initialization failed: {e}")
        return False
    
    # Test 2: Plot initialization
    print("Testing plot initialization...")
    try:
        reference_path = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        obstacles = [{'center': np.array([1.5, 1.5]), 'shape': np.eye(2), 'safety_margin': 0.2}]
        
        visualizer.initialize_plot(reference_path, obstacles)
        print("✅ Plot initialization successful")
    except Exception as e:
        print(f"❌ Plot initialization failed: {e}")
        return False
    
    # Test 3: Frame update
    print("Testing frame update...")
    try:
        vehicle_state = np.array([0.5, 0.5, 0.0, 1.0, 0.0])
        control_input = np.array([0.1, 0.0])
        trajectory = np.array([[0, 0], [0.5, 0.5]])
        
        visualizer.update_frame(
            vehicle_state=vehicle_state,
            control_input=control_input,
            trajectory=trajectory,
            objective_value=1.0,
            solve_time=0.1
        )
        print("✅ Frame update successful")
    except Exception as e:
        print(f"❌ Frame update failed: {e}")
        return False
    
    # Test 4: Animation creation
    print("Testing animation creation...")
    try:
        # Add more frames
        for i in range(5):
            vehicle_state = np.array([i*0.2, i*0.2, 0.0, 1.0, 0.0])
            control_input = np.array([0.1, 0.0])
            trajectory = np.array([[j*0.2, j*0.2] for j in range(i+2)])
            
            visualizer.update_frame(
                vehicle_state=vehicle_state,
                control_input=control_input,
                trajectory=trajectory,
                objective_value=1.0 + i*0.1,
                solve_time=0.1 + i*0.01
            )
        
        # Create animation
        gif_path = visualizer.start_animation(
            total_frames=6,
            save_gif=True,
            gif_filename="test_animation.gif"
        )
        print(f"✅ Animation creation successful: {gif_path}")
    except Exception as e:
        print(f"❌ Animation creation failed: {e}")
        return False
    
    # Test 5: Data export
    print("Testing data export...")
    try:
        data_path = visualizer.export_data("test_data.json")
        print(f"✅ Data export successful: {data_path}")
    except Exception as e:
        print(f"❌ Data export failed: {e}")
        return False
    
    # Clean up
    visualizer.close()
    
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("REAL-TIME VISUALIZATION TEST SUITE")
    print("=" * 60)
    
    # Test components first
    if not test_visualization_components():
        print("\n❌ Component tests failed. Aborting main test.")
        return False
    
    print("\n✅ All component tests passed!")
    
    # Run main simulation
    try:
        gif_path, data_path = run_realtime_simulation()
        print(f"\n🎉 Real-time visualization test completed successfully!")
        print(f"📁 Animation GIF: {gif_path}")
        print(f"📁 Data export: {data_path}")
        return True
    except Exception as e:
        print(f"\n❌ Main simulation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
