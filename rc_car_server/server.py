"""
RC Car MPC Server.

Main server that integrates:
- Vicon motion capture for state estimation
- MPC trajectory optimization
- UDP command sending to RC car

Usage:
    python -m rc_car_server.server --config config.yaml

Or programmatically:
    server = RCCarServer(config)
    server.start()
"""

import argparse
import time
import signal
import threading
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import yaml
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rc_car_server.vicon_interface import ViconInterface, VehicleState, ViconObstacleTracker
from rc_car_server.mpc_controller import RCCarMPCController, CarCommand
from rc_car_server.car_client import RCCarClient, MockCarClient
from planning.types import ReferencePath, generate_reference_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class RCCarServer:
    """
    Main server for RC car MPC control.

    This server:
    1. Connects to Vicon for state estimation
    2. Runs MPC to compute optimal trajectories
    3. Sends commands to the RC car
    4. Provides status monitoring and visualization hooks
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_mock_car: bool = False,
        use_mock_vicon: bool = False
    ):
        """
        Initialize the server.

        Args:
            config_path: Path to YAML configuration file
            use_mock_car: If True, don't send actual UDP commands
            use_mock_vicon: If True, simulate Vicon data
        """
        self.config = self._load_config(config_path)
        self.use_mock_car = use_mock_car
        self.use_mock_vicon = use_mock_vicon

        # Components
        self.vicon: Optional[ViconInterface] = None
        self.obstacle_tracker: Optional[ViconObstacleTracker] = None
        self.controller: Optional[RCCarMPCController] = None
        self.car_client: Optional[RCCarClient] = None

        # State
        self._running = False
        self._control_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            'mpc_solves': 0,
            'mpc_failures': 0,
            'commands_sent': 0,
            'start_time': None,
            'last_state': None,
            'last_command': None
        }

        # Callbacks for external monitoring
        self._state_callbacks: List[callable] = []
        self._command_callbacks: List[callable] = []

        logger.info("RCCarServer initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            # Vicon settings
            'vicon': {
                'host': 'localhost',
                'port': 801,
                'subject_name': 'RCCar',
                'update_rate': 100.0
            },
            # Car settings
            'car': {
                'ip': '192.168.1.105',
                'port': 12345,
                'send_rate': 50.0
            },
            # MPC settings
            'mpc': {
                'horizon': 10,
                'timestep': 0.1,
                'dynamics_model': 'unicycle',
                'objective': 'contouring',
                'constraints': ['contouring']
            },
            # Vehicle parameters
            'vehicle': {
                'wheelbase': 0.26,
                'max_velocity': 3.0,
                'max_acceleration': 2.0,
                'max_steering_angle': 30.0,
                'robot_radius': 0.15
            },
            # Control settings
            'control': {
                'rate': 20.0,  # Hz
                'timeout': 0.5
            },
            # Reference path (waypoints)
            'path': {
                'waypoints': [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [4.0, 1.0],
                    [4.0, 3.0],
                    [2.0, 4.0],
                    [0.0, 3.0],
                    [0.0, 0.0]
                ]
            },
            # Obstacles (Vicon subject names)
            'obstacles': []
        }

        if config_path is not None:
            try:
                with open(config_path, 'r') as f:
                    loaded = yaml.safe_load(f)
                # Deep merge
                self._deep_merge(default_config, loaded)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        return default_config

    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def setup(self) -> bool:
        """
        Set up all components.

        Returns:
            True if setup successful
        """
        try:
            # Set up Vicon
            vicon_cfg = self.config['vicon']
            self.vicon = ViconInterface(
                host=vicon_cfg['host'],
                port=vicon_cfg['port'],
                subject_name=vicon_cfg['subject_name'],
                use_simulation=self.use_mock_vicon,
                update_rate=vicon_cfg['update_rate']
            )

            if not self.vicon.connect():
                logger.error("Failed to connect to Vicon")
                return False

            # Set up obstacle tracker if obstacles configured
            obstacle_names = self.config.get('obstacles', [])
            if obstacle_names:
                self.obstacle_tracker = ViconObstacleTracker(self.vicon, obstacle_names)
                logger.info(f"Tracking {len(obstacle_names)} obstacles: {obstacle_names}")

            # Set up MPC controller
            mpc_cfg = self.config['mpc']
            vehicle_cfg = self.config['vehicle']

            controller_config = {
                **mpc_cfg,
                **vehicle_cfg,
                'control_rate': self.config['control']['rate']
            }

            self.controller = RCCarMPCController(config_path=None)
            # Override config
            self.controller.config.update(controller_config)
            self.controller.horizon = mpc_cfg['horizon']
            self.controller.timestep = mpc_cfg['timestep']
            self.controller.wheelbase = vehicle_cfg['wheelbase']
            self.controller.max_velocity = vehicle_cfg['max_velocity']

            # Create reference path from waypoints
            waypoints = self.config['path']['waypoints']
            if len(waypoints) >= 2:
                start = tuple(waypoints[0])
                # For closed-loop paths (start == end), use the second-to-last waypoint as goal
                goal = tuple(waypoints[-1])
                if len(waypoints) > 2 and start == goal:
                    # Closed loop - use second-to-last point as goal to avoid spline issues
                    goal = tuple(waypoints[-2])
                    logger.info("Detected closed-loop path, using second-to-last waypoint as goal")
                path = generate_reference_path(start, goal, path_type="curved", num_points=max(20, len(waypoints) * 3))
                self.controller.set_reference_path(path)
            else:
                logger.error("Need at least 2 waypoints for reference path")
                return False

            # Initialize controller
            if not self.controller.initialize():
                logger.error("Failed to initialize MPC controller")
                return False

            # Set up car client
            car_cfg = self.config['car']
            if self.use_mock_car:
                self.car_client = MockCarClient(
                    car_ip=car_cfg['ip'],
                    car_port=car_cfg['port'],
                    send_rate=car_cfg['send_rate']
                )
            else:
                self.car_client = RCCarClient(
                    car_ip=car_cfg['ip'],
                    car_port=car_cfg['port'],
                    send_rate=car_cfg['send_rate']
                )

            if not self.car_client.connect():
                logger.error("Failed to connect car client")
                return False

            logger.info("Server setup complete")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self):
        """Start the control loop."""
        if self._running:
            logger.warning("Server already running")
            return

        self._running = True
        self.stats['start_time'] = time.time()

        # Start control thread
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

        # Start status thread
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
        self._status_thread.start()

        logger.info("Server started")

    def stop(self):
        """Stop the server and send stop command to car."""
        logger.info("Stopping server...")
        self._running = False

        # Send stop command
        if self.car_client is not None:
            self.car_client.send_stop()

        # Wait for threads
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
        if self._status_thread is not None:
            self._status_thread.join(timeout=1.0)

        # Disconnect components
        if self.car_client is not None:
            self.car_client.disconnect()
        if self.vicon is not None:
            self.vicon.disconnect()

        logger.info("Server stopped")

    def _control_loop(self):
        """Main control loop running at fixed rate."""
        rate = self.config['control']['rate']
        interval = 1.0 / rate

        logger.info(f"Control loop started at {rate}Hz")

        while self._running:
            loop_start = time.time()

            try:
                # Get state from Vicon
                state = self.vicon.get_state()

                if not state.valid:
                    logger.warning("Invalid Vicon state, sending stop")
                    command = self.controller.stop()
                else:
                    # Update controller state
                    self.controller.update_state(state)

                    # Update obstacles if tracking
                    if self.obstacle_tracker is not None:
                        self.obstacle_tracker.update()
                        obstacles = []
                        for name, obs_state in self.obstacle_tracker.get_obstacles().items():
                            if obs_state.valid:
                                obstacles.append({
                                    'x': obs_state.x,
                                    'y': obs_state.y,
                                    'vx': obs_state.vx,
                                    'vy': obs_state.vy,
                                    'radius': 0.2
                                })
                        self.controller.update_obstacles(obstacles)

                    # Compute MPC command
                    command = self.controller.compute_command()
                    self.stats['mpc_solves'] += 1

                # Send command to car
                if self.car_client.send_command(command):
                    self.stats['commands_sent'] += 1
                    self.stats['last_command'] = command

                self.stats['last_state'] = state

                # Notify callbacks
                for cb in self._state_callbacks:
                    try:
                        cb(state, command)
                    except Exception as e:
                        logger.error(f"State callback error: {e}")

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                self.stats['mpc_failures'] += 1

                # Send stop on error
                try:
                    self.car_client.send_command(
                        CarCommand(throttle=0.0, steering=90.0, timestamp=time.time())
                    )
                except:
                    pass

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > interval * 1.5:
                logger.warning(f"Control loop overrun: {elapsed*1000:.1f}ms > {interval*1000:.1f}ms")

    def _status_loop(self):
        """Status reporting loop."""
        while self._running:
            try:
                state = self.stats.get('last_state')
                command = self.stats.get('last_command')

                if state is not None and command is not None:
                    logger.info(
                        f"State: x={state.x:.2f}, y={state.y:.2f}, "
                        f"yaw={np.degrees(state.yaw):.1f}deg, "
                        f"v={np.sqrt(state.vx**2 + state.vy**2):.2f}m/s | "
                        f"Cmd: throttle={command.throttle:.2f}, steering={command.steering:.1f}"
                    )

                # Log statistics every 10 seconds
                if self.stats['start_time'] is not None:
                    elapsed = time.time() - self.stats['start_time']
                    if int(elapsed) % 10 == 0:
                        logger.info(
                            f"Stats: solves={self.stats['mpc_solves']}, "
                            f"failures={self.stats['mpc_failures']}, "
                            f"commands={self.stats['commands_sent']}"
                        )

            except Exception as e:
                logger.error(f"Status loop error: {e}")

            time.sleep(1.0)

    def add_state_callback(self, callback: callable):
        """Add callback for state/command updates."""
        self._state_callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get current statistics."""
        return self.stats.copy()

    def set_reference_path(self, waypoints: List[Tuple[float, float]]):
        """Update the reference path."""
        if self.controller is not None:
            self.controller.set_reference_path_from_waypoints(waypoints)
            logger.info(f"Reference path updated with {len(waypoints)} waypoints")

    def emergency_stop(self):
        """Emergency stop the car."""
        logger.warning("EMERGENCY STOP")
        if self.car_client is not None:
            self.car_client.send_stop()
        if self.controller is not None:
            self.controller.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='RC Car MPC Server')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--mock-car', action='store_true',
                       help='Use mock car client (no UDP)')
    parser.add_argument('--mock-vicon', action='store_true',
                       help='Use simulated Vicon data')
    parser.add_argument('--vicon-host', type=str, default=None,
                       help='Vicon server IP address')
    parser.add_argument('--car-ip', type=str, default=None,
                       help='RC car IP address')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create server
    server = RCCarServer(
        config_path=args.config,
        use_mock_car=args.mock_car,
        use_mock_vicon=args.mock_vicon
    )

    # Override config from command line
    if args.vicon_host:
        server.config['vicon']['host'] = args.vicon_host
    if args.car_ip:
        server.config['car']['ip'] = args.car_ip

    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup and start
    if not server.setup():
        logger.error("Server setup failed")
        sys.exit(1)

    server.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
