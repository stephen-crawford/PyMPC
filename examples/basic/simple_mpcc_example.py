#!/usr/bin/env python3
"""
Simple MPCC Example.

This example demonstrates the basic usage of the MPCC framework
with contouring constraints and objectives.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from pympc.dynamics import BicycleModel
from pympc.objectives.mpcc_objective import MPCCObjective
from pympc.constraints import LinearConstraints
from pympc.constraints.mpcc_constraints import MPCCContouringConstraint, MPCCProgressConstraint
from pympc.planner import MPCCPlanner

def main():
    """Main function demonstrating basic MPCC usage."""
    print("🚀 Simple MPCC Example")
    print("=" * 40)
    
    # 1. Create a reference path
    print("1. Creating reference path...")
    t = np.linspace(0, 4, 20)
    reference_path = np.column_stack([
        2 * t,  # x = 2t
        0.1 * t**2  # y = 0.1*t^2 (parabola)
    ])
    print(f"   Path: {len(reference_path)} points from (0,0) to ({reference_path[-1,0]:.1f},{reference_path[-1,1]:.1f})")
    
    # 2. Create dynamics model
    print("2. Creating dynamics model...")
    dynamics = BicycleModel(dt=0.1, wheelbase=2.0)
    print(f"   States: {dynamics.get_state_dimension()}, Controls: {dynamics.get_control_dimension()}")
    
    # 3. Create MPCC objective
    print("3. Creating MPCC objective...")
    mpcc_objective = MPCCObjective(
        reference_path=reference_path,
        contouring_weight=10.0,  # Weight for staying close to path
        lag_weight=5.0,          # Weight for progress along path
        control_weight=0.1       # Weight for control effort
    )
    print(f"   Path length: {mpcc_objective.get_path_length():.2f}m")
    
    # 4. Create constraints
    print("4. Creating constraints...")
    
    # Linear constraints (state and control bounds)
    linear_constraints = LinearConstraints(
        state_bounds=(-10, 10),
        control_bounds=(-2, 2)
    )
    
    # MPCC contouring constraint (stay within distance of path)
    contouring_constraint = MPCCContouringConstraint(
        reference_path=reference_path,
        max_contouring_error=1.0,  # Stay within 1m of path
        safety_margin=0.2
    )
    
    # MPCC progress constraint (make progress along path)
    progress_constraint = MPCCProgressConstraint(
        reference_path=reference_path,
        min_progress_rate=0.1  # Minimum 10% progress per step
    )
    
    print("   ✅ Linear constraints created")
    print("   ✅ Contouring constraint created")
    print("   ✅ Progress constraint created")
    
    # 5. Create MPC planner
    print("5. Creating MPC planner...")
    planner = MPCCPlanner(
        dynamics=dynamics,
        horizon_length=8,
        dt=0.1
    )
    
    # Add objective and constraints
    planner.add_objective(mpcc_objective)
    planner.add_constraint(linear_constraints)
    planner.add_constraint(contouring_constraint)
    planner.add_constraint(progress_constraint)
    
    print(f"   Horizon: {planner.get_horizon_length()}")
    print(f"   Objectives: {planner.get_objective_count()}")
    print(f"   Constraints: {planner.get_constraint_count()}")
    
    # 6. Setup and solve
    print("6. Setting up and solving MPC problem...")
    planner.setup_problem()
    print("   ✅ Problem setup complete")
    
    # Initial state
    initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # x, y, theta, v, delta
    print(f"   Initial state: {initial_state}")
    
    # Solve
    print("   Solving MPC problem...")
    solution = planner.solve(initial_state, reference_path=reference_path)
    
    if solution['status'] == 'optimal':
        print("   ✅ Optimization successful!")
        print(f"   Objective value: {solution.get('objective_value', 0):.3f}")
        print(f"   Solve time: {solution.get('solve_time', 0):.3f}s")
        
        # Extract solution
        optimal_states = solution['states']
        optimal_controls = solution['controls']
        
        print(f"   Optimal trajectory shape: {optimal_states.shape}")
        print(f"   Optimal controls shape: {optimal_controls.shape}")
        
        # Show first few states and controls
        print("   First few states:")
        for i in range(min(3, optimal_states.shape[1])):
            state = optimal_states[:, i]
            print(f"     Step {i}: x={state[0]:.2f}, y={state[1]:.2f}, θ={np.degrees(state[2]):.1f}°, v={state[3]:.2f}, δ={np.degrees(state[4]):.1f}°")
        
        print("   First few controls:")
        for i in range(min(3, optimal_controls.shape[1])):
            control = optimal_controls[:, i]
            print(f"     Step {i}: a={control[0]:.2f}, δ̇={np.degrees(control[1]):.1f}°/s")
        
        # Check constraint satisfaction
        print("7. Checking constraint satisfaction...")
        
        # Evaluate constraints
        contouring_violations = contouring_constraint.evaluate(optimal_states, optimal_controls)
        progress_violations = progress_constraint.evaluate(optimal_states, optimal_controls)
        
        print(f"   Contouring constraint violations: {np.sum(contouring_violations > 0)}/{len(contouring_violations)}")
        print(f"   Progress constraint violations: {np.sum(progress_violations > 0)}/{len(progress_violations)}")
        
        if np.all(contouring_violations <= 0):
            print("   ✅ All contouring constraints satisfied")
        else:
            print("   ⚠️  Some contouring constraints violated")
            
        if np.all(progress_violations <= 0):
            print("   ✅ All progress constraints satisfied")
        else:
            print("   ⚠️  Some progress constraints violated")
        
        print("\n🎉 MPCC Example Completed Successfully!")
        print("The vehicle can follow the reference path using MPCC formulation.")
        
    else:
        print(f"   ❌ Optimization failed: {solution.get('error', 'Unknown error')}")
        print("   This may be due to numerical issues or infeasible constraints.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
