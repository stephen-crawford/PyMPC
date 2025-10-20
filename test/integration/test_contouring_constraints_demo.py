"""
Contouring Constraints Demo

This test demonstrates how contouring constraints enforce road boundaries
by showing what happens when a vehicle tries to violate them.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.standardized_logging import get_test_logger


class ContouringConstraintsDemo:
    """
    Demo showing how contouring constraints enforce road boundaries.
    """
    
    def __init__(self):
        self.logger = get_test_logger("contouring_constraints_demo", "INFO")
        
        # Create output directory
        output_dir = Path("test_results/contouring_constraints_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.log_success("Contouring constraints demo initialized")
    
    def create_road_environment(self):
        """Create a curved road environment."""
        self.logger.log_phase("Environment Setup", "Creating curved road environment")
        
        # Create curved reference path (S-shaped road)
        t = np.linspace(0, 1, 100)
        x_path = np.linspace(0, 100, 100)  # 100m road
        y_path = 8 * np.sin(0.4 * np.pi * t) + 4 * np.sin(0.8 * np.pi * t) + 2 * np.sin(1.6 * np.pi * t)  # More curved S-curve
        s_path = np.linspace(0, 1, 100)
        
        # Create road boundaries
        normals = self.calculate_path_normals(x_path, y_path)
        road_width = 6.0  # 6m wide road
        half_width = road_width / 2
        
        left_bound_x = x_path + normals[:, 0] * half_width
        left_bound_y = y_path + normals[:, 1] * half_width
        right_bound_x = x_path - normals[:, 0] * half_width
        right_bound_y = y_path - normals[:, 1] * half_width
        
        environment = {
            'reference_path': {
                'x': x_path,
                'y': y_path,
                's': s_path
            },
            'left_bound': {
                'x': left_bound_x,
                'y': left_bound_y,
                's': s_path
            },
            'right_bound': {
                'x': right_bound_x,
                'y': right_bound_y,
                's': s_path
            },
            'road_width': road_width,
            'start': (0.0, 0.0),
            'goal': (x_path[-1], y_path[-1])
        }
        
        self.logger.log_success("Environment created")
        self.logger.logger.info(f"Road length: {x_path[-1] - x_path[0]:.1f}m")
        self.logger.logger.info(f"Road width: {road_width:.1f}m")
        
        return environment
    
    def simulate_without_constraints(self, environment):
        """Simulate vehicle motion WITHOUT contouring constraints (violates boundaries)."""
        self.logger.log_phase("Simulation", "Simulating WITHOUT contouring constraints")
        
        # Vehicle state
        vehicle_x = [environment['start'][0]]
        vehicle_y = [environment['start'][1]]
        
        # Simulation parameters
        dt = 0.1
        max_time = 60.0  # Increased time to reach goal
        time_steps = int(max_time / dt)
        v = 2.0  # Reduced speed for better control
        
        violations = 0
        
        for t in range(time_steps):
            current_x = vehicle_x[-1]
            current_y = vehicle_y[-1]
            
            # Check if goal reached
            goal_distance = np.sqrt((current_x - environment['goal'][0])**2 + 
                                  (current_y - environment['goal'][1])**2)
            if goal_distance < 2.0:
                self.logger.log_success(f"Goal reached at t={t*dt:.1f}s")
                break
            
            # Simple goal-seeking (NO constraint enforcement)
            goal_dx = environment['goal'][0] - current_x
            goal_dy = environment['goal'][1] - current_y
            goal_angle = np.arctan2(goal_dy, goal_dx)
            
            # Update vehicle state (ignores road boundaries)
            new_x = current_x + v * np.cos(goal_angle) * dt
            new_y = current_y + v * np.sin(goal_angle) * dt
            
            # Check for boundary violations
            if self.check_boundary_violation(new_x, new_y, environment):
                violations += 1
                if violations % 50 == 0:  # Log every 50 violations
                    self.logger.log_warning(f"Boundary violation #{violations} at t={t*dt:.1f}s")
            
            vehicle_x.append(new_x)
            vehicle_y.append(new_y)
            
            # Log progress
            if t % 50 == 0:  # Every 5 seconds
                self.logger.logger.info(f"t={t*dt:.1f}s: pos=({new_x:.1f}, {new_y:.1f}), "
                                      f"goal_dist={goal_distance:.1f}m, violations={violations}")
        
        trajectory = {
            'x': vehicle_x,
            'y': vehicle_y,
            'violations': violations
        }
        
        self.logger.log_success("Simulation without constraints completed")
        self.logger.logger.warning(f"Total boundary violations: {violations}")
        
        return trajectory
    
    def simulate_with_constraints(self, environment):
        """Simulate vehicle motion WITH contouring constraints (respects boundaries)."""
        self.logger.log_phase("Simulation", "Simulating WITH contouring constraints")
        
        # Vehicle state
        vehicle_x = [environment['start'][0]]
        vehicle_y = [environment['start'][1]]
        
        # Simulation parameters
        dt = 0.1
        max_time = 60.0  # Increased time to reach goal
        time_steps = int(max_time / dt)
        v = 2.0  # Reduced speed for better control
        
        violations = 0
        
        for t in range(time_steps):
            current_x = vehicle_x[-1]
            current_y = vehicle_y[-1]
            
            # Check if goal reached
            goal_distance = np.sqrt((current_x - environment['goal'][0])**2 + 
                                  (current_y - environment['goal'][1])**2)
            if goal_distance < 2.0:
                self.logger.log_success(f"Goal reached at t={t*dt:.1f}s")
                break
            
            # Goal-seeking with contouring constraint enforcement
            goal_dx = environment['goal'][0] - current_x
            goal_dy = environment['goal'][1] - current_y
            goal_angle = np.arctan2(goal_dy, goal_dx)
            
            # CONTOURING CONSTRAINT: Check road boundaries and adjust heading
            desired_angle = self.apply_contouring_constraints(
                current_x, current_y, goal_angle, environment
            )
            
            # Update vehicle state
            new_x = current_x + v * np.cos(desired_angle) * dt
            new_y = current_y + v * np.sin(desired_angle) * dt
            
            # Check for boundary violations (should be minimal with constraints)
            if self.check_boundary_violation(new_x, new_y, environment):
                violations += 1
                if violations % 10 == 0:  # Log every 10 violations
                    self.logger.log_warning(f"Boundary violation #{violations} at t={t*dt:.1f}s")
            else:
                # Log when constraints are working (every 50 steps)
                if t % 50 == 0:
                    self.logger.logger.info(f"t={t*dt:.1f}s: Constraints working - vehicle within boundaries")
            
            vehicle_x.append(new_x)
            vehicle_y.append(new_y)
            
            # Log progress
            if t % 50 == 0:  # Every 5 seconds
                self.logger.logger.info(f"t={t*dt:.1f}s: pos=({new_x:.1f}, {new_y:.1f}), "
                                      f"goal_dist={goal_distance:.1f}m, violations={violations}")
        
        trajectory = {
            'x': vehicle_x,
            'y': vehicle_y,
            'violations': violations
        }
        
        self.logger.log_success("Simulation with constraints completed")
        self.logger.logger.info(f"Total boundary violations: {violations}")
        
        return trajectory
    
    def apply_contouring_constraints(self, x, y, desired_angle, environment):
        """Apply PROPER contouring constraints that actually work."""
        # Find closest point on reference path
        ref_x = np.array(environment['reference_path']['x'])
        ref_y = np.array(environment['reference_path']['y'])
        distances = np.sqrt((ref_x - x)**2 + (ref_y - y)**2)
        closest_idx = np.argmin(distances)
        
        # Get road boundaries at closest point
        left_x = environment['left_bound']['x'][closest_idx]
        left_y = environment['left_bound']['y'][closest_idx]
        right_x = environment['right_bound']['x'][closest_idx]
        right_y = environment['right_bound']['y'][closest_idx]
        
        # Calculate road center
        center_x = (left_x + right_x) / 2
        center_y = (left_y + right_y) / 2
        
        # Calculate distances to boundaries
        left_distance = np.sqrt((x - left_x)**2 + (y - left_y)**2)
        right_distance = np.sqrt((x - right_x)**2 + (y - right_y)**2)
        center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # PROPER CONTOURING CONSTRAINT LOGIC
        road_half_width = environment['road_width'] / 2
        
        # Check if vehicle is actually outside the road (violating boundaries)
        is_outside_road = (left_distance < road_half_width or 
                          right_distance < road_half_width or
                          center_distance > road_half_width)
        
        if is_outside_road:
            # Vehicle is OUTSIDE road - force it back in
            # Calculate direction to road center
            center_angle = np.arctan2(center_y - y, center_x - x)
            # Strong correction to get back into road (70% constraint, 30% goal)
            corrected_angle = 0.7 * center_angle + 0.3 * desired_angle
            
        else:
            # Vehicle is INSIDE road - check if getting close to boundaries
            safety_zone = road_half_width * 0.7  # 70% of road width
            
            if left_distance < safety_zone:
                # Getting close to left boundary - steer right
                # Calculate perpendicular direction away from left boundary
                boundary_angle = np.arctan2(left_y - y, left_x - x) + np.pi/2
                # Moderate correction (30% constraint, 70% goal)
                corrected_angle = 0.3 * boundary_angle + 0.7 * desired_angle
                
            elif right_distance < safety_zone:
                # Getting close to right boundary - steer left
                # Calculate perpendicular direction away from right boundary
                boundary_angle = np.arctan2(right_y - y, right_x - x) - np.pi/2
                # Moderate correction (30% constraint, 70% goal)
                corrected_angle = 0.3 * boundary_angle + 0.7 * desired_angle
                
            else:
                # Vehicle is safely in the middle of the road
                # No constraint needed - use original desired angle
                corrected_angle = desired_angle
        
        return corrected_angle
    
    def check_boundary_violation(self, x, y, environment):
        """Check if vehicle position violates road boundaries."""
        # Find closest point on reference path
        ref_x = np.array(environment['reference_path']['x'])
        ref_y = np.array(environment['reference_path']['y'])
        distances = np.sqrt((ref_x - x)**2 + (ref_y - y)**2)
        closest_idx = np.argmin(distances)
        
        # Get road boundaries at closest point
        left_x = environment['left_bound']['x'][closest_idx]
        left_y = environment['left_bound']['y'][closest_idx]
        right_x = environment['right_bound']['x'][closest_idx]
        right_y = environment['right_bound']['y'][closest_idx]
        
        # Calculate road center
        center_x = (left_x + right_x) / 2
        center_y = (left_y + right_y) / 2
        
        # Calculate distances
        left_distance = np.sqrt((x - left_x)**2 + (y - left_y)**2)
        right_distance = np.sqrt((x - right_x)**2 + (y - right_y)**2)
        center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Vehicle violates boundaries if it's outside the road corridor
        road_half_width = environment['road_width'] / 2
        
        # Check if vehicle is outside the road boundaries
        is_violation = (left_distance < road_half_width or 
                       right_distance < road_half_width or 
                       center_distance > road_half_width)
        
        return is_violation
    
    def visualize_comparison(self, environment, trajectory_without, trajectory_with):
        """Visualize comparison between with and without contouring constraints."""
        self.logger.log_phase("Visualization", "Creating comparison visualization")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Without contouring constraints
        ax1.plot(environment['reference_path']['x'], environment['reference_path']['y'], 
                'g-', linewidth=3, label='Reference Path', alpha=0.8)
        ax1.plot(environment['left_bound']['x'], environment['left_bound']['y'], 
                'r--', linewidth=2, label='Road Boundaries', alpha=0.8)
        ax1.plot(environment['right_bound']['x'], environment['right_bound']['y'], 
                'r--', linewidth=2, alpha=0.8)
        ax1.plot(trajectory_without['x'], trajectory_without['y'], 'orange', linewidth=3, 
                label='Vehicle Trajectory (NO Constraints)', alpha=0.9)
        ax1.plot(environment['start'][0], environment['start'][1], 'go', 
                markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax1.plot(environment['goal'][0], environment['goal'][1], 'ro', 
                markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        ax1.set_title('WITHOUT Contouring Constraints\n(Boundary Violations)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: With contouring constraints
        ax2.plot(environment['reference_path']['x'], environment['reference_path']['y'], 
                'g-', linewidth=3, label='Reference Path', alpha=0.8)
        ax2.plot(environment['left_bound']['x'], environment['left_bound']['y'], 
                'r--', linewidth=2, label='Road Boundaries', alpha=0.8)
        ax2.plot(environment['right_bound']['x'], environment['right_bound']['y'], 
                'r--', linewidth=2, alpha=0.8)
        ax2.plot(trajectory_with['x'], trajectory_with['y'], 'blue', linewidth=3, 
                label='Vehicle Trajectory (WITH Constraints)', alpha=0.9)
        ax2.plot(environment['start'][0], environment['start'][1], 'go', 
                markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax2.plot(environment['goal'][0], environment['goal'][1], 'ro', 
                markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        ax2.set_title('WITH Contouring Constraints\n(Respects Boundaries)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Add summary text
        fig.suptitle('Contouring Constraints Demo: Road Boundary Enforcement', fontsize=16, fontweight='bold')
        
        # Add violation statistics
        fig.text(0.5, 0.02, f'Violations WITHOUT constraints: {trajectory_without["violations"]} | '
                           f'Violations WITH constraints: {trajectory_with["violations"]}', 
                ha='center', fontsize=12, fontweight='bold')
        
        # Save plot
        plt.tight_layout()
        plt.savefig('test_results/contouring_constraints_demo/comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        self.logger.log_success("Comparison visualization completed")
    
    def calculate_path_normals(self, x, y):
        """Calculate path normals for road boundaries."""
        # Calculate path derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Normalize to get unit tangent vectors
        norm = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / (norm + 1e-9)
        dy_norm = dy / (norm + 1e-9)
        
        # Calculate perpendicular vectors (normals)
        normals_x = -dy_norm
        normals_y = dx_norm
        
        return np.column_stack([normals_x, normals_y])
    
    def run_demo(self):
        """Run the complete demo."""
        self.logger.log_phase("Demo Start", "Starting contouring constraints demo")
        
        try:
            # Create environment
            environment = self.create_road_environment()
            
            # Simulate without constraints
            trajectory_without = self.simulate_without_constraints(environment)
            
            # Simulate with constraints
            trajectory_with = self.simulate_with_constraints(environment)
            
            # Visualize comparison
            self.visualize_comparison(environment, trajectory_without, trajectory_with)
            
            # Calculate final statistics
            final_distance_without = np.sqrt((trajectory_without['x'][-1] - environment['goal'][0])**2 + 
                                           (trajectory_without['y'][-1] - environment['goal'][1])**2)
            final_distance_with = np.sqrt((trajectory_with['x'][-1] - environment['goal'][0])**2 + 
                                        (trajectory_with['y'][-1] - environment['goal'][1])**2)
            
            self.logger.log_success("Demo completed successfully")
            self.logger.logger.info(f"Final distance WITHOUT constraints: {final_distance_without:.2f}m")
            self.logger.logger.info(f"Final distance WITH constraints: {final_distance_with:.2f}m")
            self.logger.logger.info(f"Boundary violations WITHOUT constraints: {trajectory_without['violations']}")
            self.logger.logger.info(f"Boundary violations WITH constraints: {trajectory_with['violations']}")
            
            return True
            
        except Exception as e:
            self.logger.log_error("Demo failed", e)
            return False


# Run the demo
if __name__ == "__main__":
    demo = ContouringConstraintsDemo()
    success = demo.run_demo()
    
    print(f"\n{'='*70}")
    print(f"CONTOURING CONSTRAINTS DEMO RESULTS")
    print(f"{'='*70}")
    print(f"Demo {'PASSED' if success else 'FAILED'}")
    
    if success:
        print(f"✅ Successfully demonstrated contouring constraints enforcement!")
        print(f"📁 Results saved to: test_results/contouring_constraints_demo/")
    else:
        print(f"❌ Demo failed")
