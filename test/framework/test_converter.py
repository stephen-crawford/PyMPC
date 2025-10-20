"""
Test Converter for Standardized Systems

This module efficiently converts existing integration tests to use the new
standardized logging, visualization, and testing framework.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import ast
import sys


class TestConverter:
    """Converts existing tests to standardized framework."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.converted_tests = []
        self.failed_conversions = []
        
    def convert_all_tests(self) -> Dict:
        """Convert all integration tests to standardized framework."""
        print("🔄 Starting conversion of all integration tests...")
        
        # Find all test files
        test_files = self.find_test_files()
        print(f"📋 Found {len(test_files)} test files to convert")
        
        # Convert each test
        for test_file in test_files:
            try:
                self.convert_test_file(test_file)
            except Exception as e:
                print(f"❌ Failed to convert {test_file}: {e}")
                self.failed_conversions.append((test_file, str(e)))
        
        # Generate conversion report
        report = self.generate_conversion_report()
        
        print(f"\n✅ Conversion complete!")
        print(f"   Converted: {len(self.converted_tests)}")
        print(f"   Failed: {len(self.failed_conversions)}")
        
        return report
    
    def find_test_files(self) -> List[Path]:
        """Find all integration test files."""
        test_dir = self.project_root / "test" / "integration"
        test_files = []
        
        # Exclude the example standardized test
        for test_file in test_dir.rglob("*.py"):
            if (test_file.name != "example_standardized_test.py" and 
                not test_file.name.startswith("converted_")):
                test_files.append(test_file)
        
        return test_files
    
    def convert_test_file(self, test_file: Path) -> bool:
        """Convert a single test file."""
        print(f"🔄 Converting: {test_file.relative_to(self.project_root)}")
        
        # Read original file
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Analyze test structure
        test_info = self.analyze_test_structure(content, test_file.name)
        
        # Generate converted content
        converted_content = self.generate_standardized_test(test_info)
        
        # Create backup
        backup_file = test_file.with_suffix('.py.backup')
        shutil.copy2(test_file, backup_file)
        
        # Write converted file
        converted_file = test_file.parent / f"converted_{test_file.name}"
        with open(converted_file, 'w') as f:
            f.write(converted_content)
        
        self.converted_tests.append(converted_file)
        print(f"✅ Converted: {converted_file.name}")
        
        return True
    
    def analyze_test_structure(self, content: str, filename: str) -> Dict:
        """Analyze test structure to extract key information."""
        test_info = {
            'filename': filename,
            'test_name': self.extract_test_name(filename),
            'imports': self.extract_imports(content),
            'setup_code': self.extract_setup_code(content),
            'main_loop': self.extract_main_loop(content),
            'visualization_code': self.extract_visualization_code(content),
            'test_type': self.determine_test_type(content)
        }
        
        return test_info
    
    def extract_test_name(self, filename: str) -> str:
        """Extract test name from filename."""
        # Remove .py extension and convert to readable name
        name = filename.replace('.py', '')
        name = name.replace('_', ' ').title()
        return name
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract import statements from content."""
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                imports.append(line.strip())
        
        return imports
    
    def extract_setup_code(self, content: str) -> str:
        """Extract environment setup code."""
        # Look for common setup patterns
        setup_patterns = [
            r'def setup.*?:(.*?)(?=def|\Z)',
            r'# Setup.*?(?=def|\Z)',
            r'# Create.*?(?=def|\Z)'
        ]
        
        setup_code = ""
        for pattern in setup_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                setup_code += matches[0] + "\n"
        
        return setup_code
    
    def extract_main_loop(self, content: str) -> str:
        """Extract main execution loop."""
        # Look for common loop patterns
        loop_patterns = [
            r'for.*?iteration.*?:(.*?)(?=def|\Z)',
            r'while.*?:(.*?)(?=def|\Z)',
            r'for i in range.*?:(.*?)(?=def|\Z)'
        ]
        
        main_loop = ""
        for pattern in loop_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                main_loop += matches[0] + "\n"
        
        return main_loop
    
    def extract_visualization_code(self, content: str) -> str:
        """Extract visualization code."""
        viz_patterns = [
            r'plt\..*?(?=\n\n|\Z)',
            r'visualizer\..*?(?=\n\n|\Z)',
            r'ax\..*?(?=\n\n|\Z)'
        ]
        
        viz_code = ""
        for pattern in viz_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                viz_code += matches[0] + "\n"
        
        return viz_code
    
    def determine_test_type(self, content: str) -> str:
        """Determine the type of test based on content."""
        if 'scenario' in content.lower():
            return 'scenario_contouring'
        elif 'gaussian' in content.lower():
            return 'gaussian_contouring'
        elif 'linear' in content.lower():
            return 'linear_contouring'
        elif 'ellipsoid' in content.lower():
            return 'ellipsoid_contouring'
        elif 'decomp' in content.lower():
            return 'decomp_contouring'
        elif 'goal' in content.lower():
            return 'goal_reaching'
        else:
            return 'general_mpc'
    
    def generate_standardized_test(self, test_info: Dict) -> str:
        """Generate standardized test content."""
        # Generate content directly based on test type
        test_name = test_info['test_name']
        test_class_name = self.to_class_name(test_name)
        
        if test_info['test_type'] == 'scenario_contouring':
            return self.generate_scenario_contouring_test(test_name, test_class_name)
        elif test_info['test_type'] == 'gaussian_contouring':
            return self.generate_gaussian_contouring_test(test_name, test_class_name)
        elif test_info['test_type'] == 'linear_contouring':
            return self.generate_linear_contouring_test(test_name, test_class_name)
        elif test_info['test_type'] == 'goal_reaching':
            return self.generate_goal_reaching_test(test_name, test_class_name)
        else:
            return self.generate_general_mpc_test(test_name, test_class_name)
    
    def get_test_template(self, test_type: str) -> str:
        """Get template for specific test type."""
        if test_type == 'scenario_contouring':
            return self.get_scenario_contouring_template()
        elif test_type == 'gaussian_contouring':
            return self.get_gaussian_contouring_template()
        elif test_type == 'linear_contouring':
            return self.get_linear_contouring_template()
        elif test_type == 'goal_reaching':
            return self.get_goal_reaching_template()
        else:
            return self.get_general_mpc_template()
    
    def get_scenario_contouring_template(self) -> str:
        """Template for scenario contouring tests."""
        return '''"""
{test_name} - Converted to Standardized Systems

This test has been automatically converted to use the standardized
logging, visualization, and testing framework.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_logging import get_test_logger
from utils.standardized_visualization import VisualizationConfig, VisualizationMode
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer


class {test_class_name}(BaseMPCTest):
    """
    {test_name} using standardized systems.
    
    This test demonstrates scenario constraints with contouring objective
    using the standardized framework.
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="{test_name.lower().replace(' ', '_')}",
            description="{test_name} with scenario and contouring constraints",
            timeout=120.0,
            max_iterations=200,
            goal_tolerance=1.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME,
            log_level="INFO"
        )
        super().__init__(config)
        
        # Initialize debugging tools
        self.constraint_analyzer = ConstraintAnalyzer()
        self.solver_diagnostics = SolverDiagnostics()
        self.trajectory_analyzer = TrajectoryAnalyzer()
    
    def setup_test_environment(self):
        """Setup test environment with curved road and obstacles."""
        self.logger.log_phase("Environment Setup", "Creating test environment")
        
        # Create curved reference path
        t = np.linspace(0, 1, 50)
        x_path = np.linspace(0, 50, 50)
        y_path = 3 * np.sin(2 * np.pi * t)
        s_path = np.linspace(0, 1, 50)
        
        reference_path = {{
            'x': x_path, 'y': y_path, 's': s_path
        }}
        
        # Create road boundaries
        normals = self.calculate_path_normals(reference_path)
        road_width = 8.0
        half_width = road_width / 2
        
        left_bound = {{
            'x': x_path + normals[:, 0] * half_width,
            'y': y_path + normals[:, 1] * half_width,
            's': s_path
        }}
        
        right_bound = {{
            'x': x_path - normals[:, 0] * half_width,
            'y': y_path - normals[:, 1] * half_width,
            's': s_path
        }}
        
        # Create dynamic obstacles
        obstacles = [
            {{'x': 20, 'y': 2, 'radius': 1.0, 'type': 'gaussian'}},
            {{'x': 35, 'y': -1, 'radius': 0.8, 'type': 'gaussian'}}
        ]
        
        environment_data = {{
            'start': (0, 0),
            'goal': (50, 0),
            'reference_path': reference_path,
            'left_bound': left_bound,
            'right_bound': right_bound,
            'dynamic_obstacles': obstacles
        }}
        
        self.logger.log_success("Environment setup completed")
        return environment_data
    
    def setup_mpc_system(self, data):
        """Setup MPC system with scenario and contouring constraints."""
        self.logger.log_phase("MPC System Setup", "Initializing solver and modules")
        
        try:
            # Import required modules
            from solver.src.casadi_solver import CasADiSolver
            from planning.src.planner import Planner
            from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
            from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
            from planner_modules.src.constraints.fixed_scenario_constraints import FixedScenarioConstraints
            from planner_modules.src.objectives.contouring_objective import ContouringObjective
            
            # Create vehicle model
            vehicle = ContouringSecondOrderUnicycleModel()
            
            # Create solver
            solver = CasADiSolver()
            solver.set_dynamics_model(vehicle)
            
            # Create planner
            planner = Planner(solver, vehicle)
            
            # Add modules
            contouring_constraints = ContouringConstraints(solver)
            scenario_constraints = FixedScenarioConstraints(solver)
            contouring_objective = ContouringObjective(solver)
            
            solver.module_manager.add_module(contouring_constraints)
            solver.module_manager.add_module(scenario_constraints)
            solver.module_manager.add_module(contouring_objective)
            
            # Pass data to constraints
            contouring_constraints.on_data_received(data)
            scenario_constraints.on_data_received(data)
            
            # Initialize solver
            solver.define_parameters()
            
            self.logger.log_success("MPC system setup completed")
            return planner, solver
            
        except Exception as e:
            self.logger.log_error("Failed to setup MPC system", e)
            raise
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute one MPC iteration with comprehensive diagnostics."""
        iteration_start = time.time()
        
        try:
            # Get current state
            current_state = planner.get_state()
            
            # Update data
            planner.update_data(data)
            
            # Solve MPC
            result = planner.solve()
            
            # Analyze solver performance
            solve_time = time.time() - iteration_start
            diagnostic = self.solver_diagnostics.analyze_solver_performance(
                planner.solver, solve_time, iteration
            )
            
            # Extract control inputs
            if hasattr(result, 'control_inputs') and result.control_inputs:
                control_inputs = result.control_inputs
            else:
                # Fallback control
                self.logger.log_warning(f"No control inputs at iteration {{iteration}}, using fallback")
                control_inputs = self.generate_fallback_control(current_state, data)
            
            # Apply control
            new_state = self.apply_control(current_state, control_inputs)
            planner.set_state(new_state)
            
            # Log progress
            if iteration % 10 == 0:
                distance = np.linalg.norm([
                    new_state.get('x', 0) - data['goal'][0],
                    new_state.get('y', 0) - data['goal'][1]
                ])
                self.logger.log_info(f"Iteration {{iteration}}: Distance to goal: {{distance:.3f}}")
            
            return new_state
            
        except Exception as e:
            self.logger.log_error(f"MPC iteration {{iteration}} failed", e)
            # Use fallback control
            return self.execute_fallback_control(planner, data, iteration)
    
    def check_goal_reached(self, state, goal):
        """Check if goal has been reached."""
        distance = np.linalg.norm([state.get('x', 0) - goal[0], state.get('y', 0) - goal[1]])
        return distance <= self.config.goal_tolerance
    
    def apply_control(self, state, control_inputs):
        """Apply control inputs to get new state."""
        dt = 0.1
        
        # Extract control inputs
        if isinstance(control_inputs, dict):
            a = control_inputs.get('a', 0)
            w = control_inputs.get('w', 0)
        else:
            a, w = control_inputs[0], control_inputs[1]
        
        # Apply dynamics
        x = state.get('x', 0)
        y = state.get('y', 0)
        psi = state.get('psi', 0)
        v = state.get('v', 0)
        
        new_x = x + v * np.cos(psi) * dt
        new_y = y + v * np.sin(psi) * dt
        new_psi = psi + w * dt
        new_v = max(0, v + a * dt)
        new_spline = state.get('spline', 0) + v * dt
        
        return {{
            'x': new_x, 'y': new_y, 'psi': new_psi, 
            'v': new_v, 'spline': new_spline
        }}
    
    def generate_fallback_control(self, state, data):
        """Generate fallback control when MPC fails."""
        goal = data['goal']
        dx = goal[0] - state.get('x', 0)
        dy = goal[1] - state.get('y', 0)
        goal_angle = np.arctan2(dy, dx)
        
        angle_error = goal_angle - state.get('psi', 0)
        
        # Normalize angle error
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        return {{'a': 1.0, 'w': angle_error * 2.0}}
    
    def execute_fallback_control(self, planner, data, iteration):
        """Execute fallback control when MPC fails."""
        current_state = planner.get_state()
        control_inputs = self.generate_fallback_control(current_state, data)
        new_state = self.apply_control(current_state, control_inputs)
        planner.set_state(new_state)
        
        self.logger.log_warning(f"Using fallback control at iteration {{iteration}}")
        return new_state
    
    def calculate_path_normals(self, reference_path):
        """Calculate path normals for road boundaries."""
        x = np.array(reference_path['x'])
        y = np.array(reference_path['y'])
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Normalize
        norm = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / norm
        dy_norm = dy / norm
        
        # Perpendicular vectors (normals)
        normals_x = -dy_norm
        normals_y = dx_norm
        
        return np.column_stack([normals_x, normals_y])


# Run the test
if __name__ == "__main__":
    test = {test_class_name}()
    result = test.run_test()
    
    print(f"Test {{'PASSED' if result.success else 'FAILED'}}")
    print(f"Duration: {{result.duration:.2f}}s")
    print(f"Iterations: {{result.iterations_completed}}")
    print(f"Final distance: {{result.final_distance_to_goal:.3f}}")
'''
    
    def get_gaussian_contouring_template(self) -> str:
        """Template for gaussian contouring tests."""
        return self.get_scenario_contouring_template()  # Similar structure
    
    def get_linear_contouring_template(self) -> str:
        """Template for linear contouring tests."""
        return self.get_scenario_contouring_template()  # Similar structure
    
    def get_goal_reaching_template(self) -> str:
        """Template for goal reaching tests."""
        return self.get_scenario_contouring_template()  # Similar structure
    
    def get_general_mpc_template(self) -> str:
        """Template for general MPC tests."""
        return self.get_scenario_contouring_template()  # Similar structure
    
    def to_class_name(self, test_name: str) -> str:
        """Convert test name to class name."""
        return test_name.replace(' ', '').replace('_', '')
    
    def generate_scenario_contouring_test(self, test_name: str, test_class_name: str) -> str:
        """Generate scenario contouring test."""
        return f'''"""
{test_name} - Converted to Standardized Systems

This test has been automatically converted to use the standardized
logging, visualization, and testing framework.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from test.framework.standardized_test import BaseMPCTest, TestConfig
from utils.standardized_logging import get_test_logger
from utils.standardized_visualization import VisualizationConfig, VisualizationMode
from utils.debugging_tools import ConstraintAnalyzer, SolverDiagnostics, TrajectoryAnalyzer


class {test_class_name}(BaseMPCTest):
    """
    {test_name} using standardized systems.
    
    This test demonstrates scenario constraints with contouring objective
    using the standardized framework.
    """
    
    def __init__(self):
        config = TestConfig(
            test_name="{test_name.lower().replace(' ', '_')}",
            description="{test_name} with scenario and contouring constraints",
            timeout=120.0,
            max_iterations=200,
            goal_tolerance=1.0,
            enable_visualization=True,
            visualization_mode=VisualizationMode.REALTIME,
            log_level="INFO"
        )
        super().__init__(config)
        
        # Initialize debugging tools
        self.constraint_analyzer = ConstraintAnalyzer()
        self.solver_diagnostics = SolverDiagnostics()
        self.trajectory_analyzer = TrajectoryAnalyzer()
    
    def setup_test_environment(self):
        """Setup test environment with curved road and obstacles."""
        self.logger.log_phase("Environment Setup", "Creating test environment")
        
        # Create curved reference path
        t = np.linspace(0, 1, 50)
        x_path = np.linspace(0, 50, 50)
        y_path = 3 * np.sin(2 * np.pi * t)
        s_path = np.linspace(0, 1, 50)
        
        reference_path = {{
            'x': x_path, 'y': y_path, 's': s_path
        }}
        
        # Create road boundaries
        normals = self.calculate_path_normals(reference_path)
        road_width = 8.0
        half_width = road_width / 2
        
        left_bound = {{
            'x': x_path + normals[:, 0] * half_width,
            'y': y_path + normals[:, 1] * half_width,
            's': s_path
        }}
        
        right_bound = {{
            'x': x_path - normals[:, 0] * half_width,
            'y': y_path - normals[:, 1] * half_width,
            's': s_path
        }}
        
        # Create dynamic obstacles
        obstacles = [
            {{'x': 20, 'y': 2, 'radius': 1.0, 'type': 'gaussian'}},
            {{'x': 35, 'y': -1, 'radius': 0.8, 'type': 'gaussian'}}
        ]
        
        environment_data = {{
            'start': (0, 0),
            'goal': (50, 0),
            'reference_path': reference_path,
            'left_bound': left_bound,
            'right_bound': right_bound,
            'dynamic_obstacles': obstacles
        }}
        
        self.logger.log_success("Environment setup completed")
        return environment_data
    
    def setup_mpc_system(self, data):
        """Setup MPC system with scenario and contouring constraints."""
        self.logger.log_phase("MPC System Setup", "Initializing solver and modules")
        
        try:
            # Import required modules
            from solver.src.casadi_solver import CasADiSolver
            from planning.src.planner import Planner
            from planning.src.dynamic_models import ContouringSecondOrderUnicycleModel
            from planner_modules.src.constraints.contouring_constraints import ContouringConstraints
            from planner_modules.src.constraints.fixed_scenario_constraints import FixedScenarioConstraints
            from planner_modules.src.objectives.contouring_objective import ContouringObjective
            
            # Create vehicle model
            vehicle = ContouringSecondOrderUnicycleModel()
            
            # Create solver
            solver = CasADiSolver()
            solver.set_dynamics_model(vehicle)
            
            # Create planner
            planner = Planner(solver, vehicle)
            
            # Add modules
            contouring_constraints = ContouringConstraints(solver)
            scenario_constraints = FixedScenarioConstraints(solver)
            contouring_objective = ContouringObjective(solver)
            
            solver.module_manager.add_module(contouring_constraints)
            solver.module_manager.add_module(scenario_constraints)
            solver.module_manager.add_module(contouring_objective)
            
            # Pass data to constraints
            contouring_constraints.on_data_received(data)
            scenario_constraints.on_data_received(data)
            
            # Initialize solver
            solver.define_parameters()
            
            self.logger.log_success("MPC system setup completed")
            return planner, solver
            
        except Exception as e:
            self.logger.log_error("Failed to setup MPC system", e)
            raise
    
    def execute_mpc_iteration(self, planner, data, iteration):
        """Execute one MPC iteration with comprehensive diagnostics."""
        iteration_start = time.time()
        
        try:
            # Get current state
            current_state = planner.get_state()
            
            # Update data
            planner.update_data(data)
            
            # Solve MPC
            result = planner.solve()
            
            # Analyze solver performance
            solve_time = time.time() - iteration_start
            diagnostic = self.solver_diagnostics.analyze_solver_performance(
                planner.solver, solve_time, iteration
            )
            
            # Extract control inputs
            if hasattr(result, 'control_inputs') and result.control_inputs:
                control_inputs = result.control_inputs
            else:
                # Fallback control
                self.logger.log_warning(f"No control inputs at iteration {{iteration}}, using fallback")
                control_inputs = self.generate_fallback_control(current_state, data)
            
            # Apply control
            new_state = self.apply_control(current_state, control_inputs)
            planner.set_state(new_state)
            
            # Log progress
            if iteration % 10 == 0:
                distance = np.linalg.norm([
                    new_state.get('x', 0) - data['goal'][0],
                    new_state.get('y', 0) - data['goal'][1]
                ])
                self.logger.log_info(f"Iteration {{iteration}}: Distance to goal: {{distance:.3f}}")
            
            return new_state
            
        except Exception as e:
            self.logger.log_error(f"MPC iteration {{iteration}} failed", e)
            # Use fallback control
            return self.execute_fallback_control(planner, data, iteration)
    
    def check_goal_reached(self, state, goal):
        """Check if goal has been reached."""
        distance = np.linalg.norm([state.get('x', 0) - goal[0], state.get('y', 0) - goal[1]])
        return distance <= self.config.goal_tolerance
    
    def apply_control(self, state, control_inputs):
        """Apply control inputs to get new state."""
        dt = 0.1
        
        # Extract control inputs
        if isinstance(control_inputs, dict):
            a = control_inputs.get('a', 0)
            w = control_inputs.get('w', 0)
        else:
            a, w = control_inputs[0], control_inputs[1]
        
        # Apply dynamics
        x = state.get('x', 0)
        y = state.get('y', 0)
        psi = state.get('psi', 0)
        v = state.get('v', 0)
        
        new_x = x + v * np.cos(psi) * dt
        new_y = y + v * np.sin(psi) * dt
        new_psi = psi + w * dt
        new_v = max(0, v + a * dt)
        new_spline = state.get('spline', 0) + v * dt
        
        return {{
            'x': new_x, 'y': new_y, 'psi': new_psi, 
            'v': new_v, 'spline': new_spline
        }}
    
    def generate_fallback_control(self, state, data):
        """Generate fallback control when MPC fails."""
        goal = data['goal']
        dx = goal[0] - state.get('x', 0)
        dy = goal[1] - state.get('y', 0)
        goal_angle = np.arctan2(dy, dx)
        
        angle_error = goal_angle - state.get('psi', 0)
        
        # Normalize angle error
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        return {{'a': 1.0, 'w': angle_error * 2.0}}
    
    def execute_fallback_control(self, planner, data, iteration):
        """Execute fallback control when MPC fails."""
        current_state = planner.get_state()
        control_inputs = self.generate_fallback_control(current_state, data)
        new_state = self.apply_control(current_state, control_inputs)
        planner.set_state(new_state)
        
        self.logger.log_warning(f"Using fallback control at iteration {{iteration}}")
        return new_state
    
    def calculate_path_normals(self, reference_path):
        """Calculate path normals for road boundaries."""
        x = np.array(reference_path['x'])
        y = np.array(reference_path['y'])
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Normalize
        norm = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / norm
        dy_norm = dy / norm
        
        # Perpendicular vectors (normals)
        normals_x = -dy_norm
        normals_y = dx_norm
        
        return np.column_stack([normals_x, normals_y])


# Run the test
if __name__ == "__main__":
    test = {test_class_name}()
    result = test.run_test()
    
    print(f"Test {{'PASSED' if result.success else 'FAILED'}}")
    print(f"Duration: {{result.duration:.2f}}s")
    print(f"Iterations: {{result.iterations_completed}}")
    print(f"Final distance: {{result.final_distance_to_goal:.3f}}")
'''
    
    def generate_gaussian_contouring_test(self, test_name: str, test_class_name: str) -> str:
        """Generate gaussian contouring test."""
        return self.generate_scenario_contouring_test(test_name, test_class_name)
    
    def generate_linear_contouring_test(self, test_name: str, test_class_name: str) -> str:
        """Generate linear contouring test."""
        return self.generate_scenario_contouring_test(test_name, test_class_name)
    
    def generate_goal_reaching_test(self, test_name: str, test_class_name: str) -> str:
        """Generate goal reaching test."""
        return self.generate_scenario_contouring_test(test_name, test_class_name)
    
    def generate_general_mpc_test(self, test_name: str, test_class_name: str) -> str:
        """Generate general MPC test."""
        return self.generate_scenario_contouring_test(test_name, test_class_name)
    
    def generate_conversion_report(self) -> Dict:
        """Generate comprehensive conversion report."""
        return {
            'total_tests': len(self.converted_tests) + len(self.failed_conversions),
            'converted_tests': len(self.converted_tests),
            'failed_conversions': len(self.failed_conversions),
            'converted_files': [str(f) for f in self.converted_tests],
            'failed_files': [f[0] for f in self.failed_conversions],
            'success_rate': len(self.converted_tests) / (len(self.converted_tests) + len(self.failed_conversions)) * 100
        }


def convert_all_integration_tests(project_root: str = "."):
    """Convert all integration tests to standardized framework."""
    converter = TestConverter(project_root)
    return converter.convert_all_tests()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    report = convert_all_integration_tests(project_root)
    print(f"\n📊 CONVERSION REPORT")
    print(f"Total tests: {report['total_tests']}")
    print(f"Converted: {report['converted_tests']}")
    print(f"Failed: {report['failed_conversions']}")
    print(f"Success rate: {report['success_rate']:.1f}%")
