#!/usr/bin/env python3
"""
Demo script for PyMPC framework.

This script demonstrates the basic usage of the PyMPC framework
with contouring control and obstacle avoidance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add pympc to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pympc'))

from pympc.core.dynamics import create_dynamics_model
from pympc.core.planner import create_mpc_planner
from pympc.modules.constraints.contouring_constraints import ContouringConstraints
from pympc.modules.constraints.ellipsoid_constraints import EllipsoidConstraints
from pympc.modules.objectives.contouring_objective import ContouringObjective


def create_curved_path():
    """Create a curved reference path."""
    t = np.linspace(0, 4*np.pi, 100)
    x = t * np.cos(t/4)
    y = t * np.sin(t/4)
    return np.column_stack([x, y])


def create_obstacles():
    """Create obstacles for the test."""
    obstacles = [
        {'center': [10.0, 5.0], 'radius': 1.5},
        {'center': [20.0, -3.0], 'radius': 1.0},
        {'center': [30.0, 8.0], 'radius': 2.0}
    ]
    return obstacles


def run_mpc_demo():
    """Run MPC demo."""
    print("PyMPC Demo")
    print("=" * 50)
    
    # Create reference path
    reference_path = create_curved_path()
    print(f"Created reference path with {len(reference_path)} points")
    
    # Create obstacles
    obstacles = create_obstacles()
    print(f"Created {len(obstacles)} obstacles")
    
    # Create dynamics model
    dynamics = create_dynamics_model("bicycle", dt=0.1)
    print(f"Created {dynamics.__class__.__name__} dynamics model")
    
    # Create planner
    planner = create_mpc_planner(
        dynamics_type="bicycle",
        horizon_length=20,
        dt=0.1
    )
    print("Created MPC planner")
    
    # Add contouring objective
    contouring_obj = ContouringObjective(
        contouring_weight=2.0,
        lag_weight=1.0,
        progress_weight=1.5,
        velocity_weight=0.1
    )
    contouring_obj.set_reference_path(reference_path)
    planner.add_objective(contouring_obj)
    print("Added contouring objective")
    
    # Add contouring constraints
    contouring_const = ContouringConstraints(
        road_width=8.0,
        safety_margin=0.5
    )
    contouring_const.set_reference_path(reference_path)
    planner.add_constraint(contouring_const)
    print("Added contouring constraints")
    
    # Add obstacle constraints
    ellipsoid_const = EllipsoidConstraints(safety_margin=1.0)
    for i, obstacle in enumerate(obstacles):
        ellipsoid_const.add_circular_obstacle(
            obstacle['center'], obstacle['radius'], f'obstacle_{i}'
        )
    planner.add_constraint(ellipsoid_const)
    print("Added obstacle constraints")
    
    # Run simulation
    print("\nRunning MPC simulation...")
    
    # Initial state
    initial_state = np.array([0.0, 0.0, 0.0, 5.0])  # [x, y, psi, v]
    current_state = initial_state.copy()
    
    trajectory = [current_state[:2].copy()]
    max_steps = 100
    
    for step in range(max_steps):
        try:
            # Solve MPC
            solution = planner.solve(current_state)
            
            if solution is None:
                print(f"MPC failed at step {step}")
                break
            
            # Apply first control input
            if 'controls' in solution and solution['controls'] is not None:
                control = solution['controls'][:, 0]
                
                # Simple dynamics integration
                dt = 0.1
                psi, v = current_state[2], current_state[3]
                a, delta_dot = control[0], control[1]
                
                # Update state
                current_state[0] += v * np.cos(psi) * dt
                current_state[1] += v * np.sin(psi) * dt
                current_state[2] += delta_dot * dt
                current_state[3] += a * dt
                
                # Store trajectory
                trajectory.append(current_state[:2].copy())
                
                # Check for goal reaching
                if np.linalg.norm(current_state[:2] - reference_path[-1]) < 2.0:
                    print(f"Goal reached at step {step}")
                    break
            else:
                print(f"No control solution at step {step}")
                break
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
    
    print(f"Simulation completed with {len(trajectory)} trajectory points")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot reference path
    ax.plot(reference_path[:, 0], reference_path[:, 1], 'b-', linewidth=2, label='Reference Path')
    
    # Plot vehicle trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Vehicle Trajectory')
    
    # Plot obstacles
    for obstacle in obstacles:
        circle = plt.Circle(obstacle['center'], obstacle['radius'], 
                          color='red', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)
    
    # Plot start and end points
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    
    # Set up plot
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('PyMPC Demo - Contouring Control with Obstacle Avoidance')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Save plot
    plt.savefig('pympc_demo.png', dpi=300, bbox_inches='tight')
    print("Demo visualization saved as 'pympc_demo.png'")
    
    plt.show()


def test_overactuated_system():
    """Test overactuated system."""
    print("\nTesting Overactuated System")
    print("=" * 50)
    
    # Create overactuated dynamics
    dynamics = create_dynamics_model("overactuated_unicycle", dt=0.1)
    print(f"Created {dynamics.__class__.__name__} dynamics model")
    
    # Create planner
    planner = create_mpc_planner(
        dynamics_type="overactuated_unicycle",
        horizon_length=20,
        dt=0.1
    )
    print("Created MPC planner for overactuated system")
    
    # Test with simple goal objective
    from pympc.modules.objectives.goal_objective import GoalObjective
    
    goal_obj = GoalObjective(
        goal_position=[10.0, 10.0],
        distance_weight=1.0,
        velocity_weight=0.1
    )
    planner.add_objective(goal_obj)
    print("Added goal objective")
    
    # Test solve
    initial_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0])  # [x, y, psi, v, omega]
    solution = planner.solve(initial_state)
    
    if solution is not None:
        print("Overactuated system test: PASS")
    else:
        print("Overactuated system test: FAIL")


def main():
    """Main function."""
    print("PyMPC Framework Demo")
    print("=" * 50)
    
    try:
        # Run main demo
        run_mpc_demo()
        
        # Test overactuated system
        test_overactuated_system()
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
