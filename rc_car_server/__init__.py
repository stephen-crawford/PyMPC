"""
RC Car MPC Server Package.

This package provides real-time MPC control for an RC car using:
- Vicon motion capture for state estimation
- PyMPC framework for trajectory optimization
- UDP communication with the car

Quick Start:
    from rc_car_server import RCCarServer

    server = RCCarServer(config_path='config.yaml')
    server.setup()
    server.start()

    # ... server runs in background ...

    server.stop()

For testing without hardware:
    server = RCCarServer(
        config_path='config.yaml',
        use_mock_car=True,
        use_mock_vicon=True
    )
"""

from rc_car_server.vicon_interface import (
    ViconInterface,
    VehicleState,
    ViconObstacleTracker,
    ViconConnectionState
)

from rc_car_server.mpc_controller import (
    RCCarMPCController,
    CarCommand
)

from rc_car_server.car_client import (
    RCCarClient,
    MockCarClient,
    CarClientConfig
)

from rc_car_server.server import (
    RCCarServer
)

__all__ = [
    # Vicon interface
    'ViconInterface',
    'VehicleState',
    'ViconObstacleTracker',
    'ViconConnectionState',

    # MPC controller
    'RCCarMPCController',
    'CarCommand',

    # Car client
    'RCCarClient',
    'MockCarClient',
    'CarClientConfig',

    # Main server
    'RCCarServer',
]

__version__ = '0.1.0'
