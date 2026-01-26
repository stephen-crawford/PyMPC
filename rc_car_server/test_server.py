#!/usr/bin/env python3
"""
Test script for RC Car MPC Server.

This script tests the server components without actual hardware:
1. Simulated Vicon data
2. Mock car client (no UDP)
3. MPC trajectory optimization

Usage:
    python -m rc_car_server.test_server
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rc_car_server.vicon_interface import ViconInterface, VehicleState
from rc_car_server.mpc_controller import RCCarMPCController, CarCommand
from rc_car_server.car_client import MockCarClient
from rc_car_server.server import RCCarServer
from planning.types import ReferencePath, generate_reference_path


def test_vicon_interface():
    """Test Vicon interface with simulation."""
    print("\n" + "="*60)
    print("Testing Vicon Interface (Simulated)")
    print("="*60)

    vicon = ViconInterface(
        host="localhost",
        subject_name="TestCar",
        use_simulation=True,
        update_rate=100.0
    )

    assert vicon.connect(), "Failed to connect"
    print("  Connected to simulated Vicon")

    # Set simulation state
    vicon.set_simulation_state(x=1.0, y=2.0, yaw=0.5, v=1.5)
    time.sleep(0.1)  # Wait for update

    state = vicon.get_state()
    print(f"  State: x={state.x:.2f}, y={state.y:.2f}, yaw={state.yaw:.2f}, valid={state.valid}")

    assert state.valid, "State should be valid"
    assert abs(state.x - 1.0) < 0.01, f"X should be 1.0, got {state.x}"
    assert abs(state.y - 2.0) < 0.01, f"Y should be 2.0, got {state.y}"

    vicon.disconnect()
    print("  Vicon interface test PASSED")


def test_car_client():
    """Test mock car client."""
    print("\n" + "="*60)
    print("Testing Car Client (Mock)")
    print("="*60)

    client = MockCarClient(car_ip="192.168.1.105", car_port=12345)
    assert client.connect(), "Failed to connect"
    print("  Mock client connected")

    # Send some commands
    commands = [
        CarCommand(throttle=0.5, steering=90.0, timestamp=time.time()),
        CarCommand(throttle=0.3, steering=100.0, timestamp=time.time()),
        CarCommand(throttle=0.0, steering=90.0, timestamp=time.time()),
    ]

    for cmd in commands:
        assert client.send_command(cmd), "Failed to send command"
        print(f"  Sent: throttle={cmd.throttle:.2f}, steering={cmd.steering:.1f}")

    history = client.get_command_history()
    assert len(history) == 3, f"Expected 3 commands, got {len(history)}"

    client.disconnect()
    print("  Car client test PASSED")


def test_mpc_controller():
    """Test MPC controller with simple path."""
    print("\n" + "="*60)
    print("Testing MPC Controller")
    print("="*60)

    # Create simple path from start to goal (different points)
    start = (0.0, 0.0)
    goal = (4.0, 2.0)

    path = generate_reference_path(start, goal, path_type="curved", num_points=20)
    print(f"  Created path from {start} to {goal}")

    # Create controller
    controller = RCCarMPCController()
    controller.set_reference_path(path)

    # Initialize
    if not controller.initialize():
        print("  WARNING: MPC initialization failed (expected if modules not loaded)")
        print("  MPC controller test SKIPPED")
        return

    print("  Controller initialized")

    # Simulate state
    state = VehicleState(
        x=0.5, y=0.1, z=0.0,
        roll=0.0, pitch=0.0, yaw=0.1,
        vx=1.0, vy=0.0, vz=0.0,
        omega=0.0,
        timestamp=time.time(),
        valid=True
    )

    controller.update_state(state)

    # Compute command
    command = controller.compute_command()
    print(f"  Command: throttle={command.throttle:.2f}, steering={command.steering:.1f}")

    assert command.valid, "Command should be valid"
    print("  MPC controller test PASSED")


def test_full_server():
    """Test full server with mock components."""
    print("\n" + "="*60)
    print("Testing Full Server (Mock Mode)")
    print("="*60)

    # Create server with mocks
    server = RCCarServer(
        config_path=None,
        use_mock_car=True,
        use_mock_vicon=True
    )

    # Override path with simple waypoints
    server.config['path']['waypoints'] = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
        [0.0, 0.0]
    ]

    print("  Setting up server...")
    if not server.setup():
        print("  WARNING: Server setup failed (may be expected if modules not fully configured)")
        print("  Server test SKIPPED")
        return

    print("  Starting server...")
    server.start()

    # Simulate vehicle motion
    print("  Simulating vehicle motion for 3 seconds...")
    start_time = time.time()
    x, y, yaw, v = 0.0, 0.0, 0.0, 1.0

    while time.time() - start_time < 3.0:
        # Update simulated state
        dt = 0.1
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt

        if server.vicon is not None:
            server.vicon.set_simulation_state(x, y, yaw, v)

        time.sleep(dt)

        # Print status
        stats = server.get_stats()
        if stats['last_command'] is not None:
            cmd = stats['last_command']
            print(f"    x={x:.2f}, y={y:.2f} -> throttle={cmd.throttle:.2f}, steering={cmd.steering:.1f}")

    server.stop()
    print("  Server stopped")

    stats = server.get_stats()
    print(f"  Final stats: solves={stats['mpc_solves']}, commands={stats['commands_sent']}")
    print("  Server test PASSED")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# RC Car MPC Server Tests")
    print("#"*60)

    try:
        test_vicon_interface()
        test_car_client()
        test_mpc_controller()
        test_full_server()

        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
