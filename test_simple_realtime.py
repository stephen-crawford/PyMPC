#!/usr/bin/env python3
"""
Simple test for real-time visualization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
from utils.realtime_visualizer import RealtimeVisualizer

def test_simple_visualization():
    """Test simple visualization without MPC."""
    print("Testing simple real-time visualization...")
    
    try:
        # Create visualizer
        visualizer = RealtimeVisualizer(figsize=(12, 8), fps=2)
        print("✅ Visualizer created")
        
        # Create simple reference path
        reference_path = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
        obstacles = [
            {'center': np.array([2, 2]), 'shape': np.eye(2), 'safety_margin': 0.5}
        ]
        
        # Initialize plot
        visualizer.initialize_plot(reference_path, obstacles)
        print("✅ Plot initialized")
        
        # Simulate vehicle movement
        trajectory = []
        for i in range(10):
            # Simple straight line movement
            x = i * 0.4
            y = i * 0.4
            trajectory.append([x, y])
            
            # Create vehicle state (x, y, theta, v, delta)
            vehicle_state = np.array([x, y, 0.0, 1.0, 0.0])
            control_input = np.array([0.1, 0.0])
            
            # Update visualizer
            visualizer.update_frame(
                vehicle_state=vehicle_state,
                control_input=control_input,
                trajectory=np.array(trajectory),
                objective_value=1.0 + i * 0.1,
                solve_time=0.1 + i * 0.01
            )
        
        print("✅ Frames updated")
        
        # Create animation
        gif_path = visualizer.start_animation(
            total_frames=10,
            save_gif=True,
            gif_filename="simple_test.gif"
        )
        print(f"✅ Animation created: {gif_path}")
        
        # Export data
        data_path = visualizer.export_data("simple_test_data.json")
        print(f"✅ Data exported: {data_path}")
        
        # Clean up
        visualizer.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_visualization()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
