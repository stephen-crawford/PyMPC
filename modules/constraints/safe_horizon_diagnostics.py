"""
Diagnostic output generator for Safe Horizon MPC.

This module generates detailed diagnostic output that verifies the calculations
match the C++ reference implementation behavior.

Reference: https://github.com/tud-amr/mpc_planner
"""
import os
import numpy as np
import json
import csv
from typing import List, Dict, Optional
from datetime import datetime
from modules.constraints.scenario_utils.scenario_module import SafeHorizonModule
from modules.constraints.scenario_utils.math_utils import ScenarioConstraint
from planning.types import Scenario
from utils.utils import LOG_INFO, LOG_DEBUG


class SafeHorizonDiagnostics:
    """Generates detailed diagnostic output for Safe Horizon MPC verification."""
    
    def __init__(self, output_folder: str):
        """
        Initialize diagnostics output generator.
        
        Args:
            output_folder: Path to test output folder
        """
        self.output_folder = output_folder
        self.iteration_data = []  # Store data for each MPC iteration
        self.current_iteration = 0
        
    def start_iteration(self, iteration: int, vehicle_state: np.ndarray):
        """Start recording data for a new MPC iteration."""
        self.current_iteration = iteration
        self.iteration_data.append({
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'vehicle_state': {
                'x': float(vehicle_state[0]) if len(vehicle_state) > 0 else 0.0,
                'y': float(vehicle_state[1]) if len(vehicle_state) > 1 else 0.0,
                'psi': float(vehicle_state[2]) if len(vehicle_state) > 2 else 0.0,
                'v': float(vehicle_state[3]) if len(vehicle_state) > 3 else 0.0,
            },
            'scenario_sampling': {},
            'support_sets': {},  # Per time step
            'constraints': {},  # Per time step
            'obstacle_trajectories': {},
            'verification': {}
        })
    
    def record_scenario_sampling(self, scenario_module: SafeHorizonModule, 
                                obstacles: List, timestep: float):
        """
        Record scenario sampling details.
        
        Reference: C++ mpc_planner - scenarios are sampled from Gaussian predictions
        """
        if not self.iteration_data:
            return
        
        data = self.iteration_data[-1]
        
        # Record sampling parameters
        data['scenario_sampling'] = {
            'num_scenarios': scenario_module.num_scenarios,
            'epsilon_p': scenario_module.epsilon_p,
            'beta': scenario_module.beta,
            'n_bar': scenario_module.n_bar,
            'computed_sample_size': scenario_module.compute_sample_size(),
            'timestep': timestep,
            'obstacles': []
        }
        
        # Record scenarios per obstacle
        for obs_idx, obstacle in enumerate(obstacles):
            if not hasattr(obstacle, 'prediction') or obstacle.prediction is None:
                continue
            
            obs_data = {
                'obstacle_id': obs_idx,
                'position': {
                    'x': float(obstacle.position[0]),
                    'y': float(obstacle.position[1])
                },
                'radius': float(obstacle.radius) if hasattr(obstacle, 'radius') else 0.35,
                'scenarios': []
            }
            
            # Count scenarios for this obstacle
            obstacle_scenarios = [s for s in scenario_module.scenarios 
                                if hasattr(s, 'obstacle_idx_') and s.obstacle_idx_ == obs_idx]
            obs_data['num_scenarios'] = len(obstacle_scenarios)
            
            # Record sample scenario trajectories (first 5 for brevity)
            for scenario in obstacle_scenarios[:5]:
                if hasattr(scenario, 'trajectory') and scenario.trajectory:
                    traj_points = []
                    for step, pos in enumerate(scenario.trajectory[:10]):  # First 10 steps
                        if isinstance(pos, np.ndarray) and len(pos) >= 2:
                            traj_points.append({
                                'step': step,
                                'x': float(pos[0]),
                                'y': float(pos[1])
                            })
                    obs_data['scenarios'].append({
                        'scenario_idx': scenario.idx_ if hasattr(scenario, 'idx_') else -1,
                        'trajectory': traj_points
                    })
            
            data['scenario_sampling']['obstacles'].append(obs_data)
    
    def record_support_set(self, disc_id: int, step: int, 
                          all_scenarios: List[Scenario],
                          selected_scenarios: List[Scenario],
                          reference_robot_pos: np.ndarray):
        """
        Record support set selection details.
        
        Reference: C++ mpc_planner - support set is selected per time step
        """
        if not self.iteration_data:
            return
        
        data = self.iteration_data[-1]
        key = f"disc_{disc_id}_step_{step}"
        
        if key not in data['support_sets']:
            data['support_sets'][key] = {
                'disc_id': disc_id,
                'step': step,
                'reference_robot_pos': {
                    'x': float(reference_robot_pos[0]),
                    'y': float(reference_robot_pos[1])
                },
                'total_scenarios': len(all_scenarios),
                'selected_scenarios': []
            }
        
        # Record selected scenarios with distances
        for scenario in selected_scenarios:
            # Get obstacle position for this step
            if hasattr(scenario, 'trajectory') and scenario.trajectory and step < len(scenario.trajectory):
                obstacle_pos = np.array([float(scenario.trajectory[step][0]), 
                                       float(scenario.trajectory[step][1])])
            else:
                obstacle_pos = np.array([float(scenario.position[0]), 
                                       float(scenario.position[1])])
            
            dist = np.linalg.norm(obstacle_pos - reference_robot_pos)
            
            data['support_sets'][key]['selected_scenarios'].append({
                'scenario_idx': scenario.idx_ if hasattr(scenario, 'idx_') else -1,
                'obstacle_idx': scenario.obstacle_idx_ if hasattr(scenario, 'obstacle_idx_') else -1,
                'obstacle_pos': {
                    'x': float(obstacle_pos[0]),
                    'y': float(obstacle_pos[1])
                },
                'distance_to_robot': float(dist),
                'radius': float(scenario.radius) if hasattr(scenario, 'radius') else 0.35
            })
        
        # Verify support set size matches n_bar
        n_bar = len(selected_scenarios)
        data['support_sets'][key]['support_set_size'] = n_bar
        data['support_sets'][key]['matches_n_bar'] = (n_bar <= data['scenario_sampling'].get('n_bar', 5))
    
    def record_constraints(self, disc_id: int, step: int,
                          constraints: List[ScenarioConstraint],
                          reference_robot_pos: np.ndarray):
        """
        Record constraint formulation details.
        
        Reference: C++ mpc_planner - constraints are linearized: a1*x + a2*y <= b
        """
        if not self.iteration_data:
            return
        
        data = self.iteration_data[-1]
        key = f"disc_{disc_id}_step_{step}"
        
        if key not in data['constraints']:
            data['constraints'][key] = {
                'disc_id': disc_id,
                'step': step,
                'reference_robot_pos': {
                    'x': float(reference_robot_pos[0]),
                    'y': float(reference_robot_pos[1])
                },
                'constraints': []
            }
        
        # Record each constraint
        for i, constraint in enumerate(constraints):
            if not isinstance(constraint, ScenarioConstraint):
                continue
            
            # Get obstacle position
            obstacle_pos = None
            if hasattr(constraint, 'obstacle_pos') and constraint.obstacle_pos is not None:
                if isinstance(constraint.obstacle_pos, (list, tuple, np.ndarray)):
                    obstacle_pos = np.array([float(constraint.obstacle_pos[0]), 
                                          float(constraint.obstacle_pos[1])])
            
            # Compute constraint value at reference position
            constraint_value = None
            if obstacle_pos is not None:
                # Constraint: a1*x + a2*y - b <= 0
                a1 = float(constraint.a1) if hasattr(constraint, 'a1') else 0.0
                a2 = float(constraint.a2) if hasattr(constraint, 'a2') else 0.0
                b = float(constraint.b) if hasattr(constraint, 'b') else 0.0
                constraint_value = a1 * reference_robot_pos[0] + a2 * reference_robot_pos[1] - b
            
            constraint_data = {
                'constraint_idx': i,
                'scenario_idx': constraint.scenario_idx if hasattr(constraint, 'scenario_idx') else -1,
                'obstacle_idx': constraint.obstacle_idx if hasattr(constraint, 'obstacle_idx') else -1,
                'a1': float(constraint.a1) if hasattr(constraint, 'a1') else 0.0,
                'a2': float(constraint.a2) if hasattr(constraint, 'a2') else 0.0,
                'b': float(constraint.b) if hasattr(constraint, 'b') else 0.0,
                'obstacle_pos': {
                    'x': float(obstacle_pos[0]) if obstacle_pos is not None else 0.0,
                    'y': float(obstacle_pos[1]) if obstacle_pos is not None else 0.0
                } if obstacle_pos is not None else None,
                'constraint_value_at_reference': float(constraint_value) if constraint_value is not None else None,
                'satisfied': constraint_value is not None and constraint_value <= 1e-6,
                'obstacle_radius': float(constraint.obstacle_radius) if hasattr(constraint, 'obstacle_radius') else 0.35
            }
            
            # Verify constraint formulation matches C++ reference
            # C++: a1 = (obs_x - robot_x) / dist, a2 = (obs_y - robot_y) / dist
            #      b = a1*obs_x + a2*obs_y - (robot_radius + obstacle_radius)
            if obstacle_pos is not None:
                diff = obstacle_pos - reference_robot_pos
                dist = np.linalg.norm(diff)
                if dist > 1e-6:
                    expected_a1 = diff[0] / dist
                    expected_a2 = diff[1] / dist
                    robot_radius = 0.5  # Default
                    safety_margin = robot_radius + constraint_data['obstacle_radius']
                    expected_b = expected_a1 * obstacle_pos[0] + expected_a2 * obstacle_pos[1] - safety_margin
                    
                    constraint_data['verification'] = {
                        'expected_a1': float(expected_a1),
                        'expected_a2': float(expected_a2),
                        'expected_b': float(expected_b),
                        'actual_a1': constraint_data['a1'],
                        'actual_a2': constraint_data['a2'],
                        'actual_b': constraint_data['b'],
                        'a1_match': abs(constraint_data['a1'] - expected_a1) < 1e-4,
                        'a2_match': abs(constraint_data['a2'] - expected_a2) < 1e-4,
                        'b_match': abs(constraint_data['b'] - expected_b) < 1e-4,
                        'formulation_correct': (
                            abs(constraint_data['a1'] - expected_a1) < 1e-4 and
                            abs(constraint_data['a2'] - expected_a2) < 1e-4 and
                            abs(constraint_data['b'] - expected_b) < 1e-4
                        )
                    }
            
            data['constraints'][key]['constraints'].append(constraint_data)
        
        data['constraints'][key]['num_constraints'] = len(constraints)
    
    def record_obstacle_trajectories(self, obstacles: List, horizon: int, timestep: float):
        """
        Record obstacle trajectory predictions.
        
        Reference: C++ mpc_planner - obstacle predictions are propagated over horizon
        """
        if not self.iteration_data:
            return
        
        data = self.iteration_data[-1]
        
        for obs_idx, obstacle in enumerate(obstacles):
            if not hasattr(obstacle, 'prediction') or obstacle.prediction is None:
                continue
            
            if not hasattr(obstacle.prediction, 'steps') or not obstacle.prediction.steps:
                continue
            
            obs_traj = {
                'obstacle_id': obs_idx,
                'current_position': {
                    'x': float(obstacle.position[0]),
                    'y': float(obstacle.position[1])
                },
                'radius': float(obstacle.radius) if hasattr(obstacle, 'radius') else 0.35,
                'prediction_type': str(obstacle.prediction.type) if hasattr(obstacle.prediction, 'type') else 'unknown',
                'trajectory': []
            }
            
            # Record prediction steps
            for step_idx, pred_step in enumerate(obstacle.prediction.steps[:horizon]):
                if hasattr(pred_step, 'position') and pred_step.position is not None:
                    pos = pred_step.position
                    step_data = {
                        'step': step_idx,
                        'time': step_idx * timestep,
                        'position': {
                            'x': float(pos[0]),
                            'y': float(pos[1])
                        }
                    }
                    
                    # Record uncertainty if Gaussian
                    if hasattr(pred_step, 'major_radius') and hasattr(pred_step, 'minor_radius'):
                        step_data['uncertainty'] = {
                            'major_radius': float(pred_step.major_radius),
                            'minor_radius': float(pred_step.minor_radius),
                            'angle': float(pred_step.angle) if hasattr(pred_step, 'angle') else 0.0
                        }
                    
                    obs_traj['trajectory'].append(step_data)
            
            data['obstacle_trajectories'][f'obstacle_{obs_idx}'] = obs_traj
    
    def add_verification_summary(self, scenario_module: SafeHorizonModule):
        """
        Add verification summary comparing with C++ reference behavior.
        
        Reference: C++ mpc_planner behavior verification
        """
        if not self.iteration_data:
            return
        
        data = self.iteration_data[-1]
        
        verification = {
            'support_set_verification': {},
            'constraint_verification': {},
            'sample_size_verification': {}
        }
        
        # Verify support set sizes per step
        for key, support_data in data['support_sets'].items():
            step = support_data['step']
            n_bar = data['scenario_sampling'].get('n_bar', 5)
            actual_size = support_data['support_set_size']
            
            verification['support_set_verification'][key] = {
                'expected_size': n_bar,
                'actual_size': actual_size,
                'matches': actual_size <= n_bar,
                'verification': 'PASS' if actual_size <= n_bar else 'FAIL'
            }
        
        # Verify constraint formulation
        correct_formulations = 0
        total_constraints = 0
        for key, constraint_data in data['constraints'].items():
            for constraint in constraint_data['constraints']:
                total_constraints += 1
                if 'verification' in constraint and constraint['verification'].get('formulation_correct', False):
                    correct_formulations += 1
        
        verification['constraint_verification'] = {
            'total_constraints': total_constraints,
            'correct_formulations': correct_formulations,
            'formulation_accuracy': correct_formulations / total_constraints if total_constraints > 0 else 0.0,
            'verification': 'PASS' if correct_formulations == total_constraints else 'PARTIAL'
        }
        
        # Verify sample size calculation
        computed_size = scenario_module.compute_sample_size()
        expected_size = computed_size  # Should match computed value
        verification['sample_size_verification'] = {
            'computed_sample_size': computed_size,
            'num_scenarios': scenario_module.num_scenarios,
            'meets_requirement': scenario_module.num_scenarios >= computed_size,
            'verification': 'PASS' if scenario_module.num_scenarios >= computed_size else 'FAIL'
        }
        
        data['verification'] = verification
    
    def save_diagnostics(self):
        """Save all diagnostic data to files."""
        if not self.iteration_data:
            return
        
        # Save JSON summary
        json_file = os.path.join(self.output_folder, 'safe_horizon_diagnostics.json')
        with open(json_file, 'w') as f:
            json.dump(self.iteration_data, f, indent=2, default=str)
        
        # Save CSV files for easy analysis
        self._save_support_sets_csv()
        self._save_constraints_csv()
        self._save_obstacle_trajectories_csv()
        self._save_verification_summary_csv()
        
        LOG_INFO(f"SafeHorizonDiagnostics: Saved diagnostic output to {self.output_folder}")
    
    def _save_support_sets_csv(self):
        """Save support set selection details to CSV."""
        csv_file = os.path.join(self.output_folder, 'safe_horizon_support_sets.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'disc_id', 'step', 'reference_x', 'reference_y',
                           'total_scenarios', 'support_set_size', 'scenario_idx', 
                           'obstacle_idx', 'obstacle_x', 'obstacle_y', 'distance_to_robot'])
            
            for iter_data in self.iteration_data:
                iteration = iter_data['iteration']
                for key, support_data in iter_data['support_sets'].items():
                    disc_id = support_data['disc_id']
                    step = support_data['step']
                    ref_pos = support_data['reference_robot_pos']
                    
                    for scenario in support_data['selected_scenarios']:
                        writer.writerow([
                            iteration, disc_id, step,
                            ref_pos['x'], ref_pos['y'],
                            support_data['total_scenarios'],
                            support_data['support_set_size'],
                            scenario['scenario_idx'],
                            scenario['obstacle_idx'],
                            scenario['obstacle_pos']['x'],
                            scenario['obstacle_pos']['y'],
                            scenario['distance_to_robot']
                        ])
    
    def _save_constraints_csv(self):
        """Save constraint formulation details to CSV."""
        csv_file = os.path.join(self.output_folder, 'safe_horizon_constraints.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'disc_id', 'step', 'constraint_idx', 'scenario_idx',
                           'obstacle_idx', 'a1', 'a2', 'b', 'obstacle_x', 'obstacle_y',
                           'constraint_value', 'satisfied', 'a1_match', 'a2_match', 'b_match',
                           'formulation_correct'])
            
            for iter_data in self.iteration_data:
                iteration = iter_data['iteration']
                for key, constraint_data in iter_data['constraints'].items():
                    disc_id = constraint_data['disc_id']
                    step = constraint_data['step']
                    
                    for constraint in constraint_data['constraints']:
                        verification = constraint.get('verification', {})
                        writer.writerow([
                            iteration, disc_id, step,
                            constraint['constraint_idx'],
                            constraint['scenario_idx'],
                            constraint['obstacle_idx'],
                            constraint['a1'], constraint['a2'], constraint['b'],
                            constraint['obstacle_pos']['x'] if constraint['obstacle_pos'] else 0.0,
                            constraint['obstacle_pos']['y'] if constraint['obstacle_pos'] else 0.0,
                            constraint['constraint_value_at_reference'],
                            constraint['satisfied'],
                            verification.get('a1_match', False),
                            verification.get('a2_match', False),
                            verification.get('b_match', False),
                            verification.get('formulation_correct', False)
                        ])
    
    def _save_obstacle_trajectories_csv(self):
        """Save obstacle trajectory predictions to CSV."""
        csv_file = os.path.join(self.output_folder, 'safe_horizon_obstacle_trajectories.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'obstacle_id', 'step', 'time', 'x', 'y',
                           'major_radius', 'minor_radius', 'angle'])
            
            for iter_data in self.iteration_data:
                iteration = iter_data['iteration']
                for key, traj_data in iter_data['obstacle_trajectories'].items():
                    obstacle_id = traj_data['obstacle_id']
                    for step_data in traj_data['trajectory']:
                        uncertainty = step_data.get('uncertainty', {})
                        writer.writerow([
                            iteration, obstacle_id,
                            step_data['step'], step_data['time'],
                            step_data['position']['x'],
                            step_data['position']['y'],
                            uncertainty.get('major_radius', 0.0),
                            uncertainty.get('minor_radius', 0.0),
                            uncertainty.get('angle', 0.0)
                        ])
    
    def _save_verification_summary_csv(self):
        """Save verification summary to CSV."""
        csv_file = os.path.join(self.output_folder, 'safe_horizon_verification.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'verification_type', 'metric', 'value', 'expected', 'status'])
            
            for iter_data in self.iteration_data:
                iteration = iter_data['iteration']
                verification = iter_data.get('verification', {})
                
                # Support set verification
                for key, support_verif in verification.get('support_set_verification', {}).items():
                    writer.writerow([
                        iteration, 'support_set', 'size',
                        support_verif['actual_size'],
                        support_verif['expected_size'],
                        support_verif['verification']
                    ])
                
                # Constraint verification
                constraint_verif = verification.get('constraint_verification', {})
                if constraint_verif:
                    writer.writerow([
                        iteration, 'constraint', 'formulation_accuracy',
                        constraint_verif['formulation_accuracy'],
                        1.0,
                        constraint_verif['verification']
                    ])
                
                # Sample size verification
                sample_verif = verification.get('sample_size_verification', {})
                if sample_verif:
                    writer.writerow([
                        iteration, 'sample_size', 'meets_requirement',
                        sample_verif['num_scenarios'],
                        sample_verif['computed_sample_size'],
                        sample_verif['verification']
                    ])
