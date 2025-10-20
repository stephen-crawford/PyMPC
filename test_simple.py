#!/usr/bin/env python3
"""
Simple test script to verify the MPC system works.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from solver.src.casadi_solver import CasADiSolver
from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
from planning.src.types import Data, ReferencePath
import numpy as np

def test_basic_solver():
    """Test basic solver functionality."""
    print("Testing basic solver functionality...")
    
    try:
        # Create solver
        solver = CasADiSolver(timestep=0.1, horizon=5)
        print("✓ Solver created successfully")
        
        # Create model
        model = ContouringSecondOrderUnicycleModel()
        solver.set_dynamics_model(model)
        print("✓ Model set successfully")
        
        # Create simple data
        data = Data()
        data.start = [0.0, 0.0]
        data.goal = [5.0, 5.0]
        data.goal_received = True
        data.planning_start_time = 0.0
        
        # Create reference path
        reference_path = ReferencePath()
        reference_path.set('x', [0, 5])
        reference_path.set('y', [0, 5])
        reference_path.set('s', [0, 1])
        data.reference_path = reference_path
        
        # Initialize solver
        solver.initialize(data)
        print("✓ Solver initialized successfully")
        
        # Test basic functionality
        print("✓ Basic solver test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_modules():
    """Test solver with modules."""
    print("\nTesting solver with modules...")
    
    try:
        # Create solver
        solver = CasADiSolver(timestep=0.1, horizon=5)
        model = ContouringSecondOrderUnicycleModel()
        solver.set_dynamics_model(model)
        
        # Add simple modules
        from planner_modules.src.objectives.goal_objective import GoalObjective
        from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
        
        goal_module = GoalObjective(solver)
        contouring_module = ContouringConstraints(solver)
        
        solver.module_manager.add_module(goal_module)
        solver.module_manager.add_module(contouring_module)
        print("✓ Modules added successfully")
        
        # Create test data
        data = Data()
        data.start = [0.0, 0.0]
        data.goal = [5.0, 5.0]
        data.goal_received = True
        data.planning_start_time = 0.0
        
        # Create reference path
        reference_path = ReferencePath()
        reference_path.set('x', [0, 5])
        reference_path.set('y', [0, 5])
        reference_path.set('s', [0, 1])
        data.reference_path = reference_path
        
        # Create road boundaries
        left_bound = ReferencePath()
        left_bound.set('x', [0, 5])
        left_bound.set('y', [0.5, 5.5])
        right_bound = ReferencePath()
        right_bound.set('x', [0, 5])
        right_bound.set('y', [-0.5, 4.5])
        
        data.left_bound = left_bound
        data.right_bound = right_bound
        
        # Initialize
        solver.initialize(data)
        print("✓ Solver with modules initialized successfully")
        
        print("✓ Module test passed")
        return True
        
    except Exception as e:
        print(f"✗ Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running MPC System Tests")
    print("=" * 40)
    
    success1 = test_basic_solver()
    success2 = test_with_modules()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
