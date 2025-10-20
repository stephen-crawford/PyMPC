"""
Debugging Tools and Diagnostic Capabilities for PyMPC

This module provides comprehensive debugging tools including:
- Constraint analysis and violation detection
- Solver diagnostics and performance monitoring
- State trajectory analysis
- Automatic problem detection and suggestions
- Interactive debugging interfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from utils.standardized_logging import get_current_logger


@dataclass
class ConstraintViolation:
    """Information about a constraint violation."""
    constraint_name: str
    violation_amount: float
    expected_bounds: Tuple[float, float]
    actual_value: float
    violation_percentage: float
    iteration: int
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class SolverDiagnostic:
    """Diagnostic information about solver performance."""
    solver_type: str
    num_variables: int
    num_constraints: int
    solve_time: float
    iterations: int
    status: str
    objective_value: Optional[float] = None
    constraint_violations: List[ConstraintViolation] = None


class ConstraintAnalyzer:
    """Analyzes constraint violations and provides diagnostic information."""
    
    def __init__(self):
        self.logger = get_current_logger()
        self.violations = []
        self.constraint_history = []
        
    def analyze_constraint_violations(self, constraints: Dict, bounds: Dict, 
                                    iteration: int) -> List[ConstraintViolation]:
        """Analyze constraint violations and return detailed information."""
        violations = []
        
        for constraint_name, constraint_value in constraints.items():
            if constraint_name in bounds:
                lower_bound, upper_bound = bounds[constraint_name]
                
                # Check for violations
                if constraint_value < lower_bound:
                    violation_amount = lower_bound - constraint_value
                    actual_value = constraint_value
                elif constraint_value > upper_bound:
                    violation_amount = constraint_value - upper_bound
                    actual_value = constraint_value
                else:
                    continue  # No violation
                
                # Calculate violation percentage
                bound_range = upper_bound - lower_bound
                violation_percentage = (violation_amount / bound_range) * 100 if bound_range > 0 else 0
                
                # Determine severity
                if violation_percentage < 5:
                    severity = 'low'
                elif violation_percentage < 20:
                    severity = 'medium'
                elif violation_percentage < 50:
                    severity = 'high'
                else:
                    severity = 'critical'
                
                violation = ConstraintViolation(
                    constraint_name=constraint_name,
                    violation_amount=violation_amount,
                    expected_bounds=(lower_bound, upper_bound),
                    actual_value=actual_value,
                    violation_percentage=violation_percentage,
                    iteration=iteration,
                    severity=severity
                )
                
                violations.append(violation)
                self.violations.append(violation)
        
        # Log violations
        if violations:
            self.logger.log_warning(f"Found {len(violations)} constraint violations at iteration {iteration}")
            for violation in violations:
                self.logger.log_debug(f"  {violation.constraint_name}: {violation.violation_percentage:.1f}% violation ({violation.severity})")
        
        return violations
    
    def get_violation_summary(self) -> Dict:
        """Get summary of all constraint violations."""
        if not self.violations:
            return {'total_violations': 0}
        
        by_severity = {}
        by_constraint = {}
        
        for violation in self.violations:
            # Count by severity
            if violation.severity not in by_severity:
                by_severity[violation.severity] = 0
            by_severity[violation.severity] += 1
            
            # Count by constraint
            if violation.constraint_name not in by_constraint:
                by_constraint[violation.constraint_name] = 0
            by_constraint[violation.constraint_name] += 1
        
        return {
            'total_violations': len(self.violations),
            'by_severity': by_severity,
            'by_constraint': by_constraint,
            'most_violated': max(by_constraint.items(), key=lambda x: x[1]) if by_constraint else None
        }


class SolverDiagnostics:
    """Comprehensive solver diagnostics and performance monitoring."""
    
    def __init__(self):
        self.logger = get_current_logger()
        self.diagnostics_history = []
        self.performance_metrics = {
            'solve_times': [],
            'iteration_counts': [],
            'success_rate': 0.0,
            'average_solve_time': 0.0
        }
    
    def analyze_solver_performance(self, solver, solve_time: float, 
                                 iteration: int) -> SolverDiagnostic:
        """Analyze solver performance and return diagnostic information."""
        
        # Extract solver information
        solver_type = type(solver).__name__
        
        # Get solver statistics
        num_variables = getattr(solver, 'num_variables', 0)
        num_constraints = getattr(solver, 'num_constraints', 0)
        status = getattr(solver, 'status', 'unknown')
        objective_value = getattr(solver, 'objective_value', None)
        
        # Calculate iterations (if available)
        iterations = getattr(solver, 'iterations', 0)
        
        diagnostic = SolverDiagnostic(
            solver_type=solver_type,
            num_variables=num_variables,
            num_constraints=num_constraints,
            solve_time=solve_time,
            iterations=iterations,
            status=status,
            objective_value=objective_value
        )
        
        self.diagnostics_history.append(diagnostic)
        
        # Update performance metrics
        self.performance_metrics['solve_times'].append(solve_time)
        self.performance_metrics['iteration_counts'].append(iterations)
        
        # Log diagnostic information
        self.logger.log_debug(f"Solver diagnostic at iteration {iteration}", {
            'solver_type': solver_type,
            'solve_time': solve_time,
            'status': status,
            'variables': num_variables,
            'constraints': num_constraints
        })
        
        return diagnostic
    
    def detect_solver_issues(self) -> List[str]:
        """Detect potential solver issues and return suggestions."""
        issues = []
        
        if not self.diagnostics_history:
            return issues
        
        # Analyze solve times
        solve_times = [d.solve_time for d in self.diagnostics_history]
        avg_solve_time = np.mean(solve_times)
        max_solve_time = np.max(solve_times)
        
        if avg_solve_time > 1.0:
            issues.append(f"Slow average solve time: {avg_solve_time:.3f}s")
        
        if max_solve_time > 5.0:
            issues.append(f"Very slow solve time detected: {max_solve_time:.3f}s")
        
        # Analyze success rate
        successful_solves = sum(1 for d in self.diagnostics_history if d.status == 'success')
        success_rate = successful_solves / len(self.diagnostics_history)
        
        if success_rate < 0.8:
            issues.append(f"Low success rate: {success_rate:.1%}")
        
        # Analyze constraint-to-variable ratio
        if self.diagnostics_history:
            latest = self.diagnostics_history[-1]
            if latest.num_constraints > 0 and latest.num_variables > 0:
                ratio = latest.num_constraints / latest.num_variables
                if ratio > 10:
                    issues.append(f"High constraint-to-variable ratio: {ratio:.1f}")
                elif ratio < 0.5:
                    issues.append(f"Low constraint-to-variable ratio: {ratio:.1f}")
        
        return issues
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.diagnostics_history:
            return {'error': 'No diagnostic data available'}
        
        solve_times = [d.solve_time for d in self.diagnostics_history]
        iteration_counts = [d.iterations for d in self.diagnostics_history]
        
        return {
            'total_solves': len(self.diagnostics_history),
            'average_solve_time': np.mean(solve_times),
            'max_solve_time': np.max(solve_times),
            'min_solve_time': np.min(solve_times),
            'average_iterations': np.mean(iteration_counts),
            'success_rate': sum(1 for d in self.diagnostics_history if d.status == 'success') / len(self.diagnostics_history),
            'solver_types': list(set(d.solver_type for d in self.diagnostics_history))
        }


class TrajectoryAnalyzer:
    """Analyzes vehicle trajectory for issues and optimization opportunities."""
    
    def __init__(self):
        self.logger = get_current_logger()
        self.trajectory_data = []
        
    def analyze_trajectory(self, trajectory_x: List[float], trajectory_y: List[float],
                         reference_path: Optional[Dict] = None) -> Dict:
        """Analyze trajectory for issues and provide recommendations."""
        
        if len(trajectory_x) < 2:
            return {'error': 'Insufficient trajectory data'}
        
        # Convert to numpy arrays
        x = np.array(trajectory_x)
        y = np.array(trajectory_y)
        
        # Calculate trajectory metrics
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_distance = np.sum(distances)
        
        # Calculate curvature
        if len(x) > 2:
            dx = np.gradient(x)
            dy = np.gradient(y)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
            max_curvature = np.max(curvature)
        else:
            max_curvature = 0
        
        # Calculate path efficiency
        start_point = np.array([x[0], y[0]])
        end_point = np.array([x[-1], y[-1]])
        straight_line_distance = np.linalg.norm(end_point - start_point)
        path_efficiency = straight_line_distance / total_distance if total_distance > 0 else 0
        
        # Analyze deviations from reference path
        reference_deviation = 0
        if reference_path and 'x' in reference_path and 'y' in reference_path:
            ref_x = np.array(reference_path['x'])
            ref_y = np.array(reference_path['y'])
            
            # Calculate average deviation
            deviations = []
            for i in range(len(x)):
                # Find closest point on reference path
                distances_to_ref = np.sqrt((ref_x - x[i])**2 + (ref_y - y[i])**2)
                closest_idx = np.argmin(distances_to_ref)
                deviations.append(distances_to_ref[closest_idx])
            
            reference_deviation = np.mean(deviations)
        
        # Detect issues
        issues = []
        if path_efficiency < 0.5:
            issues.append(f"Low path efficiency: {path_efficiency:.2f}")
        
        if max_curvature > 1.0:
            issues.append(f"High curvature detected: {max_curvature:.2f}")
        
        if reference_deviation > 2.0:
            issues.append(f"Large deviation from reference path: {reference_deviation:.2f}m")
        
        # Calculate smoothness metrics
        velocity_changes = np.diff(distances)
        acceleration_changes = np.diff(velocity_changes)
        smoothness = np.std(acceleration_changes) if len(acceleration_changes) > 0 else 0
        
        if smoothness > 1.0:
            issues.append(f"Trajectory not smooth: acceleration std = {smoothness:.2f}")
        
        analysis = {
            'total_distance': total_distance,
            'straight_line_distance': straight_line_distance,
            'path_efficiency': path_efficiency,
            'max_curvature': max_curvature,
            'reference_deviation': reference_deviation,
            'smoothness': smoothness,
            'issues': issues,
            'trajectory_length': len(x)
        }
        
        self.logger.log_debug("Trajectory analysis completed", analysis)
        return analysis
    
    def plot_trajectory_analysis(self, trajectory_x: List[float], trajectory_y: List[float],
                               reference_path: Optional[Dict] = None, save_path: str = None):
        """Create comprehensive trajectory analysis plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Trajectory Analysis', fontsize=16)
        
        # Main trajectory plot
        ax1 = axes[0, 0]
        ax1.plot(trajectory_x, trajectory_y, 'b-', linewidth=2, label='Actual Trajectory')
        ax1.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Start')
        ax1.plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=10, label='End')
        
        if reference_path and 'x' in reference_path and 'y' in reference_path:
            ax1.plot(reference_path['x'], reference_path['y'], 'g--', alpha=0.7, label='Reference Path')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Trajectory Overview')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Velocity profile
        ax2 = axes[0, 1]
        if len(trajectory_x) > 1:
            distances = np.sqrt(np.diff(trajectory_x)**2 + np.diff(trajectory_y)**2)
            ax2.plot(distances, 'b-', linewidth=2)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity Profile')
            ax2.grid(True, alpha=0.3)
        
        # Curvature analysis
        ax3 = axes[1, 0]
        if len(trajectory_x) > 2:
            x = np.array(trajectory_x)
            y = np.array(trajectory_y)
            dx = np.gradient(x)
            dy = np.gradient(y)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
            ax3.plot(curvature, 'r-', linewidth=2)
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Curvature (1/m)')
            ax3.set_title('Curvature Profile')
            ax3.grid(True, alpha=0.3)
        
        # Deviation from reference
        ax4 = axes[1, 1]
        if reference_path and 'x' in reference_path and 'y' in reference_path:
            ref_x = np.array(reference_path['x'])
            ref_y = np.array(reference_path['y'])
            deviations = []
            
            for i in range(len(trajectory_x)):
                distances_to_ref = np.sqrt((ref_x - trajectory_x[i])**2 + (ref_y - trajectory_y[i])**2)
                deviations.append(np.min(distances_to_ref))
            
            ax4.plot(deviations, 'g-', linewidth=2)
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Deviation (m)')
            ax4.set_title('Deviation from Reference Path')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.log_success(f"Trajectory analysis plot saved to: {save_path}")
        
        plt.show()


class ProblemDetector:
    """Automatically detects common MPC problems and provides solutions."""
    
    def __init__(self):
        self.logger = get_current_logger()
        self.detected_problems = []
        
    def detect_common_problems(self, solver_diagnostics: SolverDiagnostics,
                             constraint_analyzer: ConstraintAnalyzer,
                             trajectory_analyzer: TrajectoryAnalyzer) -> List[Dict]:
        """Detect common MPC problems and return solutions."""
        
        problems = []
        
        # Check for overconstrained problems
        if solver_diagnostics.diagnostics_history:
            latest = solver_diagnostics.diagnostics_history[-1]
            if latest.num_constraints > latest.num_variables * 5:
                problems.append({
                    'type': 'overconstrained',
                    'severity': 'high',
                    'description': f'Too many constraints ({latest.num_constraints}) relative to variables ({latest.num_variables})',
                    'solutions': [
                        'Reduce number of obstacles',
                        'Increase constraint tolerance',
                        'Use constraint relaxation',
                        'Simplify constraint formulations'
                    ]
                })
        
        # Check for constraint violations
        violation_summary = constraint_analyzer.get_violation_summary()
        if violation_summary['total_violations'] > 0:
            critical_violations = violation_summary['by_severity'].get('critical', 0)
            if critical_violations > 0:
                problems.append({
                    'type': 'constraint_violations',
                    'severity': 'critical',
                    'description': f'{critical_violations} critical constraint violations detected',
                    'solutions': [
                        'Check constraint bounds',
                        'Verify constraint formulations',
                        'Increase constraint tolerance',
                        'Review obstacle positions and sizes'
                    ]
                })
        
        # Check for solver performance issues
        solver_issues = solver_diagnostics.detect_solver_issues()
        if solver_issues:
            problems.append({
                'type': 'solver_performance',
                'severity': 'medium',
                'description': 'Solver performance issues detected',
                'details': solver_issues,
                'solutions': [
                    'Optimize solver parameters',
                    'Reduce problem complexity',
                    'Use warm starting',
                    'Check numerical conditioning'
                ]
            })
        
        # Check for trajectory issues
        if hasattr(trajectory_analyzer, 'trajectory_data') and trajectory_analyzer.trajectory_data:
            trajectory_analysis = trajectory_analyzer.analyze_trajectory(
                trajectory_analyzer.trajectory_data['x'],
                trajectory_analyzer.trajectory_data['y']
            )
            
            if trajectory_analysis['issues']:
                problems.append({
                    'type': 'trajectory_issues',
                    'severity': 'medium',
                    'description': 'Trajectory quality issues detected',
                    'details': trajectory_analysis['issues'],
                    'solutions': [
                        'Adjust objective weights',
                        'Improve reference path',
                        'Check vehicle dynamics model',
                        'Optimize control parameters'
                    ]
                })
        
        self.detected_problems = problems
        
        # Log detected problems
        for problem in problems:
            self.logger.log_warning(f"Detected problem: {problem['type']} ({problem['severity']})")
            self.logger.log_debug(f"  Description: {problem['description']}")
            for solution in problem['solutions']:
                self.logger.log_debug(f"  Solution: {solution}")
        
        return problems
    
    def generate_problem_report(self) -> Dict:
        """Generate comprehensive problem detection report."""
        return {
            'total_problems': len(self.detected_problems),
            'by_severity': {
                severity: sum(1 for p in self.detected_problems if p['severity'] == severity)
                for severity in ['low', 'medium', 'high', 'critical']
            },
            'by_type': {
                problem_type: sum(1 for p in self.detected_problems if p['type'] == problem_type)
                for problem_type in set(p['type'] for p in self.detected_problems)
            },
            'problems': self.detected_problems
        }


# Export main classes
__all__ = [
    'ConstraintViolation', 'SolverDiagnostic', 'ConstraintAnalyzer',
    'SolverDiagnostics', 'TrajectoryAnalyzer', 'ProblemDetector'
]
