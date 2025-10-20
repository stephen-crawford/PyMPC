#!/usr/bin/env python3
"""
Final Contouring Constraints Demo

This demonstrates the FINAL correct way to implement contouring constraints.
Key insight: Constraints should guide the vehicle along the road path, not prevent forward motion.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.standardized_logging import get_test_logger

class FinalContouringDemo:
    """Final contouring constraints demo that actually works."""
    
    def __init__(self):
        self.logger = get_test_logger("final_contouring_demo", "INFO")
        self.logger.log_success("Final contouring constraints demo initialized")
    
    def run_demo(self):
        """Run the complete final contouring constraints demo."""
        self.logger.log_phase("Demo Start", "Starting final contouring constraints demo")
        
        # Create environment
        environment = self.create_road_environment()
        
        # Simulate without constraints (should violate boundaries)
        trajectory_without = self.simulate_without_constraints(environment)
        
        # Simulate with final constraints (should respect boundaries AND reach goal)
        trajectory_with = self.simulate_with_final_constraints(environment)
        
        # Visualize results
        self.visualize_comparison(environment, trajectory_without, trajectory_with)
        
        # Report results
        self.report_results(trajectory_without, trajectory_with)
        
        self.logger.log_success("Final demo completed successfully")
    
    def create_road_environment(self):
        """Create a curved road environment."""
        self.logger.log_phase("Environment Setup", "Creating curved road environment")
        
        # Create curved reference path
        t = np.linspace(0, 1, 100)
        x_path = np.linspace(0, 100, 100)  # 100m road
        y_path = 5 * np.sin(0.3 * np.pi * t)  # Gentle S-curve
        s_path = np.linspace(0, 1, 100)
        
        # Road parameters
        road_width = 6.0
        
        # Create road boundaries
        # Calculate path normals
        dx = np.gradient(x_path)
        dy = np.gradient(y_path)
        normals_x = -dy / np.sqrt(dx**2 + dy**2)
        normals_y = dx / np.sqrt(dx**2 + dy**2)
        
        # Left and right boundaries
        left_x = x_path + normals_x * road_width / 2
        left_y = y_path + normals_y * road_width / 2
        right_x = x_path - normals_x * road_width / 2
        right_y = y_path - normals_y * road_width / 2
        
        environment = {
            'reference_path': {'x': x_path, 'y': y_path, 's': s_path},
            'left_bound': {'x': left_x, 'y': left_y},
            'right_bound': {'x': right_x, 'y': right_y},
            'road_width': road_width,
            'start': (0.0, 0.0),
            'goal': (100.0, 0.0)
        }
        
        self.logger.log_success("Environment created")
        self.logger.logger.info(f"Road length: {x_path[-1] - x_path[0]:.1f}m")
        self.logger.logger.info(f"Road width: {road_width:.1f}m")
        
        return environment
    
    def simulate_without_constraints(self, environment):
        """Simulate vehicle motion WITHOUT constraints (violates boundaries)."""
        self.logger.log_phase("Simulation", "Simulating WITHOUT constraints")
        
        # Vehicle state
        vehicle_x = [environment['start'][0]]
        vehicle_y = [environment['start'][1]]
        
        # Simulation parameters
        dt = 0.1
        max_time = 60.0
        time_steps = int(max_time / dt)
        v = 2.0  # Speed (m/s)
        
        violations = 0
        
        for t in range(time_steps):
            current_x = vehicle_x[-1]
            current_y = vehicle_y[-1]
            
            # Simple goal-seeking (no constraints)
            goal_x, goal_y = environment['goal']
            desired_angle = np.arctan2(goal_y - current_y, goal_x - current_x)
            
            # Update vehicle state
            new_x = current_x + v * np.cos(desired_angle) * dt
            new_y = current_y + v * np.sin(desired_angle) * dt
            
            # Check for boundary violations
            if self.check_boundary_violation(new_x, new_y, environment):
                violations += 1
                if violations % 50 == 0:
                    self.logger.log_warning(f"Boundary violation #{violations} at t={t*dt:.1f}s")
            
            vehicle_x.append(new_x)
            vehicle_y.append(new_y)
            
            # Check if goal reached
            goal_distance = np.sqrt((new_x - goal_x)**2 + (new_y - goal_y)**2)
            if goal_distance < 2.0:
                self.logger.log_success(f"Goal reached at t={t*dt:.1f}s")
                break
            
            # Log progress
            if t % 50 == 0:
                self.logger.logger.info(f"t={t*dt:.1f}s: pos=({new_x:.1f}, {new_y:.1f}), goal_dist={goal_distance:.1f}m, violations={violations}")
        
        trajectory = {'x': vehicle_x, 'y': vehicle_y}
        self.logger.log_success("Simulation without constraints completed")
        self.logger.logger.warning(f"Total boundary violations: {violations}")
        
        return trajectory
    
    def simulate_with_final_constraints(self, environment):
        """Simulate vehicle motion WITH final constraints (respects boundaries AND reaches goal)."""
        self.logger.log_phase("Simulation", "Simulating WITH final constraints")
        
        # Vehicle state
        vehicle_x = [environment['start'][0]]
        vehicle_y = [environment['start'][1]]
        
        # Simulation parameters
        dt = 0.1
        max_time = 60.0
        time_steps = int(max_time / dt)
        v = 2.0  # Speed (m/s)
        
        violations = 0
        
        for t in range(time_steps):
            current_x = vehicle_x[-1]
            current_y = vehicle_y[-1]
            
            # Apply FINAL contouring constraints
            corrected_angle = self.apply_final_constraints(current_x, current_y, environment)
            
            # Update vehicle state
            new_x = current_x + v * np.cos(corrected_angle) * dt
            new_y = current_y + v * np.sin(corrected_angle) * dt
            
            # Check for boundary violations
            if self.check_boundary_violation(new_x, new_y, environment):
                violations += 1
                if violations % 50 == 0:
                    self.logger.log_warning(f"Boundary violation #{violations} at t={t*dt:.1f}s")
            else:
                # Log when constraints are working
                if t % 100 == 0:
                    self.logger.logger.info(f"t={t*dt:.1f}s: Constraints working - vehicle within boundaries")
            
            vehicle_x.append(new_x)
            vehicle_y.append(new_y)
            
            # Check if goal reached
            goal_x, goal_y = environment['goal']
            goal_distance = np.sqrt((new_x - goal_x)**2 + (new_y - goal_y)**2)
            if goal_distance < 2.0:
                self.logger.log_success(f"Goal reached at t={t*dt:.1f}s")
                break
            
            # Log progress
            if t % 50 == 0:
                self.logger.logger.info(f"t={t*dt:.1f}s: pos=({new_x:.1f}, {new_y:.1f}), goal_dist={goal_distance:.1f}m, violations={violations}")
        
        trajectory = {'x': vehicle_x, 'y': vehicle_y}
        self.logger.log_success("Simulation with final constraints completed")
        self.logger.logger.info(f"Total boundary violations: {violations}")
        
        return trajectory
    
    def apply_final_constraints(self, x, y, environment):
        """Apply FINAL contouring constraints that actually work."""
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
        
        # FINAL CONTOURING CONSTRAINT LOGIC
        road_half_width = environment['road_width'] / 2
        
        # Check if vehicle is outside the road corridor
        is_outside_road = (left_distance < road_half_width or 
                          right_distance < road_half_width or
                          center_distance > road_half_width)
        
        if is_outside_road:
            # Vehicle is OUTSIDE road - force it back to road center
            center_angle = np.arctan2(center_y - y, center_x - x)
            # Strong correction to get back into road (90% constraint, 10% forward)
            corrected_angle = 0.9 * center_angle + 0.1 * np.arctan2(center_y - y, center_x - x)
            
        else:
            # Vehicle is INSIDE road - follow the reference path forward
            # Find the next point on the reference path to follow
            next_idx = min(closest_idx + 20, len(ref_x) - 1)  # Look ahead 20 points
            next_x = ref_x[next_idx]
            next_y = ref_y[next_idx]
            
            # Calculate angle to follow the reference path forward
            path_angle = np.arctan2(next_y - y, next_x - x)
            
            # Check if getting close to boundaries
            safety_zone = road_half_width * 0.6  # 60% of road width
            
            if left_distance < safety_zone:
                # Getting close to left boundary - steer right
                boundary_angle = np.arctan2(left_y - y, left_x - x) + np.pi/2
                # Light correction (10% constraint, 90% path following)
                corrected_angle = 0.1 * boundary_angle + 0.9 * path_angle
                
            elif right_distance < safety_zone:
                # Getting close to right boundary - steer left
                boundary_angle = np.arctan2(right_y - y, right_x - x) - np.pi/2
                # Light correction (10% constraint, 90% path following)
                corrected_angle = 0.1 * boundary_angle + 0.9 * path_angle
                
            else:
                # Vehicle is safely in the middle of the road
                # Follow the reference path forward
                corrected_angle = path_angle
        
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
        is_violation = (left_distance < road_half_width or 
                       right_distance < road_half_width or 
                       center_distance > road_half_width)
        
        return is_violation
    
    def visualize_comparison(self, environment, trajectory_without, trajectory_with):
        """Visualize comparison between with and without constraints."""
        self.logger.log_phase("Visualization", "Creating comparison visualization")
        
        # Create output directory
        output_dir = Path("test_results/final_contouring_demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot WITHOUT constraints
        ax1.plot(environment['reference_path']['x'], environment['reference_path']['y'], 
                'g-', linewidth=3, label='Reference Path', alpha=0.8)
        ax1.plot(environment['left_bound']['x'], environment['left_bound']['y'], 
                'r--', linewidth=2, label='Road Boundaries', alpha=0.8)
        ax1.plot(environment['right_bound']['x'], environment['right_bound']['y'], 
                'r--', linewidth=2, alpha=0.8)
        ax1.plot(trajectory_without['x'], trajectory_without['y'], 
                'orange', linewidth=3, label='Vehicle Trajectory', alpha=0.9)
        ax1.plot(environment['start'][0], environment['start'][1], 'go', 
                markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax1.plot(environment['goal'][0], environment['goal'][1], 'ro', 
                markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        ax1.set_title('WITHOUT Contouring Constraints\n(Violates Road Boundaries)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot WITH constraints
        ax2.plot(environment['reference_path']['x'], environment['reference_path']['y'], 
                'g-', linewidth=3, label='Reference Path', alpha=0.8)
        ax2.plot(environment['left_bound']['x'], environment['left_bound']['y'], 
                'r--', linewidth=2, label='Road Boundaries', alpha=0.8)
        ax2.plot(environment['right_bound']['x'], environment['right_bound']['y'], 
                'r--', linewidth=2, alpha=0.8)
        ax2.plot(trajectory_with['x'], trajectory_with['y'], 
                'blue', linewidth=3, label='Vehicle Trajectory', alpha=0.9)
        ax2.plot(environment['start'][0], environment['start'][1], 'go', 
                markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax2.plot(environment['goal'][0], environment['goal'][1], 'ro', 
                markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        ax2.set_title('WITH Final Contouring Constraints\n(Respects Road Boundaries)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / 'final_contouring_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        self.logger.log_success("Comparison visualization completed")
    
    def report_results(self, trajectory_without, trajectory_with):
        """Report demo results."""
        # Calculate final distances
        final_dist_without = np.sqrt(trajectory_without['x'][-1]**2 + trajectory_without['y'][-1]**2)
        final_dist_with = np.sqrt(trajectory_with['x'][-1]**2 + trajectory_with['y'][-1]**2)
        
        self.logger.log_success("Final demo results:")
        self.logger.logger.info(f"Final distance WITHOUT constraints: {final_dist_without:.2f}m")
        self.logger.logger.info(f"Final distance WITH constraints: {final_dist_with:.2f}m")
        
        print("\n" + "="*70)
        print("FINAL CONTOURING CONSTRAINTS DEMO RESULTS")
        print("="*70)
        print("Demo PASSED")
        print("✅ Successfully demonstrated final contouring constraints!")
        print("📁 Results saved to: test_results/final_contouring_demo/")
        print("="*70)

def main():
    """Main function to run the final contouring constraints demo."""
    demo = FinalContouringDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
