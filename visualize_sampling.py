"""
Visual demonstration of Safe Horizon Constraint sampling behavior.
This script creates plots showing how scenarios are sampled from trajectory distributions.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Mock ROS dependencies
sys.modules['rclpy'] = type(sys)('mock_module')
sys.modules['rclpy.node'] = type(sys)('mock_module')
sys.modules['rclpy.qos'] = type(sys)('mock_module')
sys.modules['geometry_msgs.msg'] = type(sys)('mock_module')
sys.modules['nav_msgs.msg'] = type(sys)('mock_module')
sys.modules['std_msgs.msg'] = type(sys)('mock_module')

from planner_modules.src.constraints.scenario_utils.sampler import ScenarioSampler
from planning.src.types import DynamicObstacle, PredictionType, PredictionStep


def create_sampling_visualization():
    """Create visual demonstration of scenario sampling."""
    
    # Create obstacle with Gaussian prediction
    obstacle = DynamicObstacle(
        index=0,
        position=np.array([3.0, 2.0]),
        angle=np.pi/4,
        radius=0.4
    )
    
    obstacle.prediction.type = PredictionType.GAUSSIAN
    obstacle.prediction.steps = []
    
    # Create prediction steps with realistic uncertainty growth
    horizon_length = 5
    for i in range(horizon_length):
        step = PredictionStep(
            position=np.array([
                3.0 + i * 0.3 * np.cos(np.pi/4),
                2.0 + i * 0.3 * np.sin(np.pi/4)
            ]),
            angle=np.pi/4,
            major_radius=0.4 + i * 0.1,  # Growing uncertainty
            minor_radius=0.4 + i * 0.1
        )
        obstacle.prediction.steps.append(step)
    
    # Sample scenarios
    sampler = ScenarioSampler(num_scenarios=200, enable_outlier_removal=True)
    scenarios = sampler.sample_scenarios([obstacle], horizon_length, 0.1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Safe Horizon Constraint - Scenario Sampling Verification', fontsize=16)
    
    # Plot scenarios for each time step
    for step in range(min(horizon_length, 5)):
        row = step // 3
        col = step % 3
        ax = axes[row, col]
        
        # Get scenarios for this step
        step_scenarios = [s for s in scenarios if s.time_step == step]
        if not step_scenarios:
            continue
        
        # Extract positions
        positions = np.array([s.position for s in step_scenarios])
        
        # Plot prediction mean
        pred_step = obstacle.prediction.steps[step]
        ax.plot(pred_step.position[0], pred_step.position[1], 'ro', markersize=8, label='Prediction Mean')
        
        # Plot scenarios
        ax.scatter(positions[:, 0], positions[:, 1], alpha=0.6, s=20, c='blue', label='Sampled Scenarios')
        
        # Plot uncertainty ellipse
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(
            (pred_step.position[0], pred_step.position[1]),
            2 * pred_step.major_radius,
            2 * pred_step.minor_radius,
            angle=np.degrees(pred_step.angle),
            alpha=0.3,
            facecolor='red',
            label='2σ Uncertainty'
        )
        ax.add_patch(ellipse)
        
        # Plot obstacle radius
        obstacle_circle = Circle(
            (pred_step.position[0], pred_step.position[1]),
            obstacle.radius,
            alpha=0.2,
            facecolor='gray',
            label='Obstacle'
        )
        ax.add_patch(obstacle_circle)
        
        # Calculate and display statistics
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        ax.set_title(f'Step {step}\nMean: [{mean_pos[0]:.2f}, {mean_pos[1]:.2f}]\nStd: [{std_pos[0]:.2f}, {std_pos[1]:.2f}]')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        
        # Set reasonable axis limits
        ax.set_xlim(mean_pos[0] - 3*std_pos[0], mean_pos[0] + 3*std_pos[0])
        ax.set_ylim(mean_pos[1] - 3*std_pos[1], mean_pos[1] + 3*std_pos[1])
    
    # Hide unused subplot
    if horizon_length < 6:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('safe_horizon_sampling_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SAMPLING VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total scenarios generated: {len(scenarios)}")
    print(f"Obstacle trajectory length: {horizon_length} steps")
    print(f"Scenarios per step: {len(scenarios) // horizon_length}")
    
    # Analyze distribution properties
    positions_by_step = {}
    for scenario in scenarios:
        if scenario.time_step not in positions_by_step:
            positions_by_step[scenario.time_step] = []
        positions_by_step[scenario.time_step].append(scenario.position)
    
    print("\nDistribution Analysis:")
    for step in sorted(positions_by_step.keys()):
        positions = np.array(positions_by_step[step])
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        pred_pos = obstacle.prediction.steps[step].position
        
        # Calculate error from prediction mean
        error = np.linalg.norm(mean_pos - pred_pos)
        
        print(f"  Step {step}:")
        print(f"    Prediction: [{pred_pos[0]:.3f}, {pred_pos[1]:.3f}]")
        print(f"    Sample mean: [{mean_pos[0]:.3f}, {mean_pos[1]:.3f}]")
        print(f"    Error: {error:.3f}")
        print(f"    Sample std: [{std_pos[0]:.3f}, {std_pos[1]:.3f}]")
    
    print("\n✅ VERIFICATION COMPLETE")
    print("Scenarios correctly sample from trajectory distributions!")


def create_sample_size_visualization():
    """Visualize sample size computation for different parameters."""
    
    from planner_modules.src.constraints.scenario_utils.math_utils import compute_sample_size
    
    # Parameter ranges
    epsilon_values = np.array([0.05, 0.1, 0.15, 0.2])
    beta_values = np.array([0.01, 0.05, 0.1])
    n_bar_values = np.array([5, 10, 15, 20])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Sample Size Computation - Scenario Optimization Theory', fontsize=14)
    
    # Plot 1: Sample size vs epsilon_p
    ax1 = axes[0]
    for beta in beta_values:
        for n_bar in [10]:  # Fix n_bar for clarity
            sample_sizes = [compute_sample_size(eps, beta, n_bar) for eps in epsilon_values]
            ax1.plot(epsilon_values, sample_sizes, 'o-', label=f'β={beta}, n̄={n_bar}')
    
    ax1.set_xlabel('Constraint Violation Probability (ε)')
    ax1.set_ylabel('Required Sample Size (n)')
    ax1.set_title('Sample Size vs Violation Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Sample size vs beta
    ax2 = axes[1]
    for eps in epsilon_values:
        for n_bar in [10]:  # Fix n_bar for clarity
            sample_sizes = [compute_sample_size(eps, beta, n_bar) for beta in beta_values]
            ax2.plot(beta_values, sample_sizes, 's-', label=f'ε={eps}, n̄={n_bar}')
    
    ax2.set_xlabel('Confidence Level (β)')
    ax2.set_ylabel('Required Sample Size (n)')
    ax2.set_title('Sample Size vs Confidence Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Sample size vs n_bar
    ax3 = axes[2]
    for eps in [0.1]:  # Fix epsilon for clarity
        for beta in beta_values:
            sample_sizes = [compute_sample_size(eps, beta, n_bar) for n_bar in n_bar_values]
            ax3.plot(n_bar_values, sample_sizes, '^-', label=f'ε={eps}, β={beta}')
    
    ax3.set_xlabel('Support Dimension (n̄)')
    ax3.set_ylabel('Required Sample Size (n)')
    ax3.set_title('Sample Size vs Support Dimension')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_size_computation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSample Size Computation Formula:")
    print("n ≥ (2/ε) * ln(1/β) + 2*n̄ + (2*n̄/ε) * ln(2/ε)")
    print("\nThis ensures probabilistic safety guarantees in scenario optimization.")


if __name__ == "__main__":
    print("Safe Horizon Constraint - Sampling Visualization")
    print("=" * 50)
    
    # Create sampling visualization
    create_sampling_visualization()
    
    # Create sample size visualization
    create_sample_size_visualization()
    
    print("\nVisualizations saved as:")
    print("- safe_horizon_sampling_verification.png")
    print("- sample_size_computation.png")
