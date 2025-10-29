#!/usr/bin/env python3
"""
Test script for the Obstacle Manager

This script tests the obstacle manager functionality including:
- Obstacle creation with different dynamics
- State integration
- State tracking
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from obstacle_manager import (
    ObstacleManager, ObstacleConfig, 
    create_unicycle_obstacle, create_bicycle_obstacle, create_point_mass_obstacle
)


def test_obstacle_manager():
    """Test the obstacle manager functionality."""
    print("Testing Obstacle Manager...")
    
    # Create obstacle manager
    config = {
        "horizon": 10,
        "timestep": 0.1,
        "obstacle_radius": 0.35
    }
    
    manager = ObstacleManager(config)
    
    # Test 1: Create unicycle obstacle
    print("\n1. Testing unicycle obstacle creation...")
    unicycle_config = create_unicycle_obstacle(
        obstacle_id=0,
        position=np.array([5.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        angle=0.0
    )
    
    unicycle_obstacle = manager.create_obstacle(unicycle_config)
    print(f"Created unicycle obstacle: {unicycle_obstacle.index}")
    print(f"Initial position: {unicycle_obstacle.position}")
    print(f"Prediction steps: {len(unicycle_obstacle.prediction.steps)}")
    
    # Test 2: Create bicycle obstacle
    print("\n2. Testing bicycle obstacle creation...")
    bicycle_config = create_bicycle_obstacle(
        obstacle_id=1,
        position=np.array([10.0, 2.0]),
        velocity=np.array([0.5, 0.5]),
        angle=np.pi/4
    )
    
    bicycle_obstacle = manager.create_obstacle(bicycle_config)
    print(f"Created bicycle obstacle: {bicycle_obstacle.index}")
    print(f"Initial position: {bicycle_obstacle.position}")
    print(f"Prediction steps: {len(bicycle_obstacle.prediction.steps)}")
    
    # Test 3: Create point mass obstacle
    print("\n3. Testing point mass obstacle creation...")
    point_mass_config = create_point_mass_obstacle(
        obstacle_id=2,
        position=np.array([15.0, -1.0]),
        velocity=np.array([-0.5, 0.3])
    )
    
    point_mass_obstacle = manager.create_obstacle(point_mass_config)
    print(f"Created point mass obstacle: {point_mass_obstacle.index}")
    print(f"Initial position: {point_mass_obstacle.position}")
    print(f"Prediction steps: {len(point_mass_obstacle.prediction.steps)}")
    
    # Test 4: State integration
    print("\n4. Testing state integration...")
    initial_states = manager.get_all_obstacle_states()
    print(f"Initial states for {len(initial_states)} obstacles:")
    
    for i, states in enumerate(initial_states):
        print(f"  Obstacle {i}: {len(states)} state(s)")
        if len(states) > 0:
            print(f"    Initial state: {states[0]}")
    
    # Test 5: Update obstacle states
    print("\n5. Testing obstacle state updates...")
    manager.update_obstacle_states(0.1)
    
    updated_states = manager.get_all_obstacle_states()
    print(f"Updated states for {len(updated_states)} obstacles:")
    
    for i, states in enumerate(updated_states):
        print(f"  Obstacle {i}: {len(states)} state(s)")
        if len(states) > 1:
            print(f"    Previous state: {states[-2]}")
            print(f"    Current state: {states[-1]}")
    
    # Test 6: Get obstacle info
    print("\n6. Testing obstacle info...")
    obstacle_info = manager.get_obstacle_info()
    print(f"Obstacle info: {obstacle_info}")
    
    # Test 7: Create random obstacles
    print("\n7. Testing random obstacle creation...")
    random_obstacles = manager.create_random_obstacles(
        num_obstacles=3,
        dynamics_types=["unicycle", "bicycle", "point_mass"]
    )
    
    print(f"Created {len(random_obstacles)} random obstacles")
    for obstacle in random_obstacles:
        print(f"  Obstacle {obstacle.index}: position {obstacle.position}")
    
    print("\n✅ Obstacle Manager tests completed successfully!")


if __name__ == "__main__":
    test_obstacle_manager()
