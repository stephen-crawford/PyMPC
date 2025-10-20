"""
Comprehensive Test Suite for All Constraint Types

This test suite validates each constraint type individually and in combination,
following the C++ MPC libraries from:
- https://github.com/tud-amr/mpc_planner
- https://github.com/oscardegroot/scenario_module

Constraint types tested:
1. Linearized Constraints
2. Gaussian Constraints  
3. Ellipsoid Constraints
4. Decomposition Constraints
5. Scenario Constraints (Oscar de Groot's approach)
6. Contouring Constraints
7. Guidance Constraints

Each test validates:
- Individual constraint functionality
- Integration with objectives
- Performance metrics
- Visualization output
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from test.framework.base_test import BaseMPCTest
from solver.src.casadi_solver import CasADiSolver
from planning.src.dynamic_models import DynamicsModel, SecondOrderUnicycleModel, ContouringSecondOrderUnicycleModel
from planning.src.types import Data, State, Bound, PredictionType, generate_dynamic_obstacles
from scipy.interpolate import splprep, splev


class ConstraintTestSuite:
    """Comprehensive test suite for all constraint types"""
    
    def __init__(self):
        self.results = {}
        self.test_configs = {
            'linearized': {
                'constraint_class': 'LinearizedConstraints',
                'objective_class': 'GoalObjective',
                'model_class': 'SecondOrderUnicycleModel',
                'description': 'Linearized obstacle constraints with goal objective'
            },
            'gaussian': {
                'constraint_class': 'GaussianConstraints', 
                'objective_class': 'GoalObjective',
                'model_class': 'SecondOrderUnicycleModel',
                'description': 'Gaussian uncertainty constraints with goal objective'
            },
            'ellipsoid': {
                'constraint_class': 'EllipsoidConstraints',
                'objective_class': 'GoalObjective', 
                'model_class': 'SecondOrderUnicycleModel',
                'description': 'Ellipsoid uncertainty constraints with goal objective'
            },
            'decomp': {
                'constraint_class': 'DecompConstraints',
                'objective_class': 'GoalObjective',
                'model_class': 'SecondOrderUnicycleModel', 
                'description': 'Decomposition-based constraints with goal objective'
            },
            'scenario': {
                'constraint_class': 'ScenarioConstraints',
                'objective_class': 'GoalObjective',
                'model_class': 'SecondOrderUnicycleModel',
                'description': 'Scenario-based constraints with goal objective'
            },
            'proper_scenario': {
                'constraint_class': 'ProperScenarioConstraints',
                'objective_class': 'GoalObjective',
                'model_class': 'SecondOrderUnicycleModel',
                'description': 'Proper scenario constraints (Oscar de Groot IJRR 2024)'
            },
            'contouring': {
                'constraint_class': 'ContouringConstraints',
                'objective_class': 'ContouringObjective',
                'model_class': 'ContouringSecondOrderUnicycleModel',
                'description': 'Contouring constraints with MPCC objective'
            },
            'guidance': {
                'constraint_class': 'GuidanceConstraints',
                'objective_class': 'GoalObjective',
                'model_class': 'SecondOrderUnicycleModel',
                'description': 'Guidance-based constraints with goal objective'
            }
        }
    
    def run_all_tests(self, max_iterations=50, enable_viz=False):
        """Run all constraint type tests"""
        print("="*80)
        print("COMPREHENSIVE CONSTRAINT TEST SUITE")
        print("="*80)
        print("Testing all constraint types from C++ MPC libraries")
        print("="*80)
        
        for constraint_type, config in self.test_configs.items():
            print(f"\n{'='*60}")
            print(f"TESTING: {constraint_type.upper()} CONSTRAINTS")
            print(f"{'='*60}")
            print(f"Description: {config['description']}")
            print(f"Constraint: {config['constraint_class']}")
            print(f"Objective: {config['objective_class']}")
            print(f"Model: {config['model_class']}")
            print("-"*60)
            
            try:
                result = self._test_constraint_type(
                    constraint_type, 
                    config,
                    max_iterations=max_iterations,
                    enable_viz=enable_viz
                )
                self.results[constraint_type] = result
                print(f"✅ {constraint_type.upper()} TEST: {'PASSED' if result.success else 'FAILED'}")
                
            except Exception as e:
                print(f"❌ {constraint_type.upper()} TEST: FAILED - {e}")
                self.results[constraint_type] = {'success': False, 'error': str(e)}
        
        self._print_summary()
        return self.results
    
    def _test_constraint_type(self, constraint_type, config, max_iterations=50, enable_viz=False):
        """Test a specific constraint type"""
        
        # Create test class dynamically
        class DynamicConstraintTest(BaseMPCTest):
            def __init__(self, name, constraint_type, config, max_iterations, enable_viz):
                super().__init__(name, dt=0.1, horizon=10, max_iterations=max_iterations,
                               enable_visualization=enable_viz, max_consecutive_failures=10)
                self.constraint_type = constraint_type
                self.config = config
            
            def get_vehicle_model(self):
                if self.config['model_class'] == 'ContouringSecondOrderUnicycleModel':
                    return ContouringSecondOrderUnicycleModel()
                else:
                    return SecondOrderUnicycleModel()
            
            def configure_modules(self, solver):
                # Import constraint class
                constraint_module = self._get_constraint_module()
                constraint_class = getattr(constraint_module, self.config['constraint_class'])
                
                # Import objective class  
                objective_module = self._get_objective_module()
                objective_class = getattr(objective_module, self.config['objective_class'])
                
                # Add constraint
                constraint = constraint_class(solver)
                solver.module_manager.add_module(constraint)
                
                # Add objective
                objective = objective_class(solver)
                solver.module_manager.add_module(objective)
            
            def _get_constraint_module(self):
                if self.constraint_type == 'proper_scenario':
                    from planner_modules.src.constraints import proper_scenario_constraints
                    return proper_scenario_constraints
                else:
                    from planner_modules.src.constraints import scenario_constraints
                    return scenario_constraints
            
            def _get_objective_module(self):
                if self.config['objective_class'] == 'ContouringObjective':
                    from planner_modules.src.objectives import contouring_objective
                    return contouring_objective
                else:
                    from planner_modules.src.objectives import goal_objective
                    return goal_objective
            
            def setup_environment_data(self, data):
                # Generate obstacles for testing
                obstacles = generate_dynamic_obstacles(
                    number=3,
                    prediction_type=PredictionType.GAUSSIAN.name,
                    size=0.8,
                    distribution="random_paths", 
                    area=((5, 25), (5, 15), (0, 0)),
                    path_types=("straight", "curved"),
                    num_points=self.horizon + 1,
                    dim=2
                )
                data.dynamic_obstacles = obstacles
        
        # Run the test
        test = DynamicConstraintTest(
            f'{constraint_type}_constraint_test',
            constraint_type,
            config,
            max_iterations,
            enable_viz
        )
        
        test.setup(start=(0, 0), goal=(30, 20), path_type="curved", road_width=8.0)
        result = test.run()
        
        return result
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = 0
        failed = 0
        
        for constraint_type, result in self.results.items():
            status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
            print(f"{constraint_type.upper():<20} {status}")
            
            if result.get('success', False):
                passed += 1
                if 'iterations_completed' in result:
                    print(f"  Iterations: {result['iterations_completed']}")
                    print(f"  Successful: {result['iterations_completed'] - result.get('failed_iterations', 0)}")
                    if result.get('average_solve_time', 0) > 0:
                        print(f"  Solve Time: {result['average_solve_time']:.3f}s ({1.0/result['average_solve_time']:.1f} Hz)")
            else:
                failed += 1
                if 'error' in result:
                    print(f"  Error: {result['error']}")
        
        print("-"*80)
        print(f"TOTAL: {passed + failed} tests")
        print(f"PASSED: {passed}")
        print(f"FAILED: {failed}")
        print(f"SUCCESS RATE: {passed/(passed+failed)*100:.1f}%")
        print("="*80)


def main():
    """Main test function"""
    print("Starting comprehensive constraint test suite...")
    
    # Ask user for preferences
    enable_viz = input("Enable visualization? (y/n) [n]: ").strip().lower() == 'y'
    max_iterations = int(input("Max iterations per test [30]: ") or "30")
    
    # Run test suite
    suite = ConstraintTestSuite()
    results = suite.run_all_tests(max_iterations=max_iterations, enable_viz=enable_viz)
    
    return results


if __name__ == "__main__":
    results = main()
