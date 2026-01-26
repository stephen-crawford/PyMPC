"""
Vicon Motion Capture Interface for RC Car Position Tracking.

This module provides real-time position and orientation data from a Vicon
motion capture system. It supports both the official Vicon DataStream SDK
and a simulated mode for testing without hardware.

Reference: https://docs.vicon.com/display/DSSDK/Vicon+DataStream+SDK
"""

import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ViconConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class VehicleState:
    """
    Vehicle state from Vicon motion capture.

    All units are in SI (meters, radians, seconds).
    """
    x: float           # Position X [m]
    y: float           # Position Y [m]
    z: float           # Position Z [m] (height)
    roll: float        # Roll angle [rad]
    pitch: float       # Pitch angle [rad]
    yaw: float         # Yaw/heading angle [rad]
    vx: float          # Velocity X [m/s]
    vy: float          # Velocity Y [m/s]
    vz: float          # Velocity Z [m/s]
    omega: float       # Angular velocity (yaw rate) [rad/s]
    timestamp: float   # Unix timestamp [s]
    valid: bool        # Whether data is valid/occluded

    def to_mpc_state(self) -> Dict[str, float]:
        """Convert to MPC state format."""
        # Compute planar velocity magnitude
        v = np.sqrt(self.vx**2 + self.vy**2)

        return {
            'x': self.x,
            'y': self.y,
            'psi': self.yaw,
            'v': v
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, psi, v]."""
        v = np.sqrt(self.vx**2 + self.vy**2)
        return np.array([self.x, self.y, self.yaw, v])


class ViconInterface:
    """
    Interface to Vicon DataStream SDK for real-time motion capture.

    Usage:
        vicon = ViconInterface(host="192.168.1.100", subject_name="RCCar")
        vicon.connect()

        while running:
            state = vicon.get_state()
            if state.valid:
                # Use state for MPC
                pass

        vicon.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 801,
        subject_name: str = "RCCar",
        use_simulation: bool = False,
        update_rate: float = 100.0  # Hz
    ):
        """
        Initialize Vicon interface.

        Args:
            host: Vicon Tracker PC IP address
            port: Vicon DataStream port (default 801)
            subject_name: Name of the tracked subject in Vicon Tracker
            use_simulation: If True, simulate Vicon data for testing
            update_rate: Expected update rate in Hz
        """
        self.host = host
        self.port = port
        self.subject_name = subject_name
        self.use_simulation = use_simulation
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate

        # State
        self.state = VehicleState(
            x=0.0, y=0.0, z=0.0,
            roll=0.0, pitch=0.0, yaw=0.0,
            vx=0.0, vy=0.0, vz=0.0,
            omega=0.0,
            timestamp=time.time(),
            valid=False
        )
        self._prev_state: Optional[VehicleState] = None
        self._lock = threading.Lock()

        # Connection state
        self.connection_state = ViconConnectionState.DISCONNECTED
        self._client = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self._state_callbacks: list[Callable[[VehicleState], None]] = []

        # Velocity filter (simple exponential moving average)
        self._velocity_alpha = 0.3  # Filter coefficient
        self._filtered_vx = 0.0
        self._filtered_vy = 0.0
        self._filtered_omega = 0.0

        logger.info(f"ViconInterface initialized: host={host}:{port}, subject={subject_name}")

    def connect(self) -> bool:
        """
        Connect to Vicon DataStream server.

        Returns:
            True if connection successful
        """
        if self.use_simulation:
            logger.info("Using simulated Vicon data")
            self.connection_state = ViconConnectionState.CONNECTED
            self._start_update_thread()
            return True

        try:
            self.connection_state = ViconConnectionState.CONNECTING

            # Try to import Vicon DataStream SDK
            try:
                from vicon_dssdk import ViconDataStream
                self._client = ViconDataStream.Client()
            except ImportError:
                logger.warning("Vicon DataStream SDK not installed. Install with: pip install vicon-dssdk")
                logger.warning("Falling back to simulation mode")
                self.use_simulation = True
                self.connection_state = ViconConnectionState.CONNECTED
                self._start_update_thread()
                return True

            # Connect to Vicon server
            result = self._client.Connect(f"{self.host}:{self.port}")
            if result != ViconDataStream.Result.Success:
                logger.error(f"Failed to connect to Vicon: {result}")
                self.connection_state = ViconConnectionState.ERROR
                return False

            # Configure stream
            self._client.EnableSegmentData()
            self._client.SetStreamMode(ViconDataStream.StreamMode.ClientPull)

            # Set axis mapping (Vicon default is Z-up)
            self._client.SetAxisMapping(
                ViconDataStream.Direction.Forward,
                ViconDataStream.Direction.Left,
                ViconDataStream.Direction.Up
            )

            self.connection_state = ViconConnectionState.CONNECTED
            logger.info(f"Connected to Vicon at {self.host}:{self.port}")

            self._start_update_thread()
            return True

        except Exception as e:
            logger.error(f"Vicon connection error: {e}")
            self.connection_state = ViconConnectionState.ERROR
            return False

    def disconnect(self):
        """Disconnect from Vicon server."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self._client is not None and not self.use_simulation:
            try:
                self._client.Disconnect()
            except:
                pass

        self.connection_state = ViconConnectionState.DISCONNECTED
        logger.info("Disconnected from Vicon")

    def _start_update_thread(self):
        """Start the background update thread."""
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        """Background thread that continuously fetches Vicon data."""
        while self._running:
            try:
                if self.use_simulation:
                    self._update_simulation()
                else:
                    self._update_from_vicon()

                # Notify callbacks
                with self._lock:
                    state_copy = self.state
                for callback in self._state_callbacks:
                    try:
                        callback(state_copy)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Vicon update error: {e}")

            time.sleep(self.dt)

    def _update_from_vicon(self):
        """Fetch latest data from Vicon server."""
        if self._client is None:
            return

        # Get frame
        result = self._client.GetFrame()
        if result != ViconDataStream.Result.Success:
            return

        # Get subject data
        try:
            # Get global translation (position in mm, convert to m)
            trans = self._client.GetSegmentGlobalTranslation(
                self.subject_name, self.subject_name
            )
            if trans[1]:  # Occluded
                with self._lock:
                    self.state.valid = False
                return

            x = trans[0][0] / 1000.0  # mm to m
            y = trans[0][1] / 1000.0
            z = trans[0][2] / 1000.0

            # Get global rotation (Euler XYZ in radians)
            rot = self._client.GetSegmentGlobalRotationEulerXYZ(
                self.subject_name, self.subject_name
            )
            if rot[1]:  # Occluded
                with self._lock:
                    self.state.valid = False
                return

            roll = rot[0][0]
            pitch = rot[0][1]
            yaw = rot[0][2]

            # Compute velocities from position difference
            timestamp = time.time()

            with self._lock:
                if self._prev_state is not None and self._prev_state.valid:
                    dt = timestamp - self._prev_state.timestamp
                    if dt > 0:
                        # Raw velocities
                        raw_vx = (x - self._prev_state.x) / dt
                        raw_vy = (y - self._prev_state.y) / dt
                        raw_vz = (z - self._prev_state.z) / dt

                        # Angular velocity (yaw rate)
                        dyaw = yaw - self._prev_state.yaw
                        # Handle wrap-around
                        if dyaw > np.pi:
                            dyaw -= 2 * np.pi
                        elif dyaw < -np.pi:
                            dyaw += 2 * np.pi
                        raw_omega = dyaw / dt

                        # Apply exponential filter
                        self._filtered_vx = (self._velocity_alpha * raw_vx +
                                           (1 - self._velocity_alpha) * self._filtered_vx)
                        self._filtered_vy = (self._velocity_alpha * raw_vy +
                                           (1 - self._velocity_alpha) * self._filtered_vy)
                        self._filtered_omega = (self._velocity_alpha * raw_omega +
                                              (1 - self._velocity_alpha) * self._filtered_omega)

                # Update state
                self._prev_state = self.state
                self.state = VehicleState(
                    x=x, y=y, z=z,
                    roll=roll, pitch=pitch, yaw=yaw,
                    vx=self._filtered_vx,
                    vy=self._filtered_vy,
                    vz=0.0,
                    omega=self._filtered_omega,
                    timestamp=timestamp,
                    valid=True
                )

        except Exception as e:
            logger.error(f"Error reading Vicon data: {e}")
            with self._lock:
                self.state.valid = False

    def _update_simulation(self):
        """Generate simulated Vicon data for testing."""
        timestamp = time.time()

        with self._lock:
            # Simple circular motion simulation
            if not hasattr(self, '_sim_t'):
                self._sim_t = 0.0
                self._sim_x = 0.0
                self._sim_y = 0.0
                self._sim_yaw = 0.0
                self._sim_v = 0.0

            # Update simulation time
            self._sim_t += self.dt

            # Simulate some motion (can be updated externally via set_simulation_state)
            self.state = VehicleState(
                x=self._sim_x,
                y=self._sim_y,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=self._sim_yaw,
                vx=self._sim_v * np.cos(self._sim_yaw),
                vy=self._sim_v * np.sin(self._sim_yaw),
                vz=0.0,
                omega=0.0,
                timestamp=timestamp,
                valid=True
            )

    def set_simulation_state(self, x: float, y: float, yaw: float, v: float):
        """
        Set the simulated vehicle state (for testing).

        Args:
            x: Position X [m]
            y: Position Y [m]
            yaw: Heading [rad]
            v: Velocity magnitude [m/s]
        """
        if self.use_simulation:
            with self._lock:
                self._sim_x = x
                self._sim_y = y
                self._sim_yaw = yaw
                self._sim_v = v

    def get_state(self) -> VehicleState:
        """
        Get the latest vehicle state.

        Returns:
            Current VehicleState (check .valid before using)
        """
        with self._lock:
            return self.state

    def get_mpc_state(self) -> Tuple[Dict[str, float], bool]:
        """
        Get state in MPC-compatible format.

        Returns:
            Tuple of (state_dict, valid)
        """
        with self._lock:
            return self.state.to_mpc_state(), self.state.valid

    def add_state_callback(self, callback: Callable[[VehicleState], None]):
        """
        Add a callback to be called when new state data arrives.

        Args:
            callback: Function that takes a VehicleState
        """
        self._state_callbacks.append(callback)

    def remove_state_callback(self, callback: Callable[[VehicleState], None]):
        """Remove a previously added callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)

    def is_connected(self) -> bool:
        """Check if connected to Vicon."""
        return self.connection_state == ViconConnectionState.CONNECTED

    def get_latency(self) -> float:
        """Get estimated data latency in seconds."""
        with self._lock:
            return time.time() - self.state.timestamp


class ViconObstacleTracker:
    """
    Track multiple objects (obstacles) via Vicon.

    Each tracked object should be a separate subject in Vicon Tracker.
    """

    def __init__(self, vicon: ViconInterface, obstacle_names: list[str]):
        """
        Initialize obstacle tracker.

        Args:
            vicon: ViconInterface instance (already connected)
            obstacle_names: List of subject names to track as obstacles
        """
        self.vicon = vicon
        self.obstacle_names = obstacle_names
        self.obstacles: Dict[str, VehicleState] = {}

        for name in obstacle_names:
            self.obstacles[name] = VehicleState(
                x=0.0, y=0.0, z=0.0,
                roll=0.0, pitch=0.0, yaw=0.0,
                vx=0.0, vy=0.0, vz=0.0,
                omega=0.0,
                timestamp=time.time(),
                valid=False
            )

    def update(self):
        """Update obstacle positions from Vicon."""
        if self.vicon._client is None or self.vicon.use_simulation:
            return

        for name in self.obstacle_names:
            try:
                from vicon_dssdk import ViconDataStream

                trans = self.vicon._client.GetSegmentGlobalTranslation(name, name)
                if not trans[1]:  # Not occluded
                    x = trans[0][0] / 1000.0
                    y = trans[0][1] / 1000.0
                    z = trans[0][2] / 1000.0

                    rot = self.vicon._client.GetSegmentGlobalRotationEulerXYZ(name, name)
                    yaw = rot[0][2] if not rot[1] else 0.0

                    prev = self.obstacles.get(name)
                    timestamp = time.time()

                    # Compute velocity
                    vx, vy = 0.0, 0.0
                    if prev and prev.valid:
                        dt = timestamp - prev.timestamp
                        if dt > 0:
                            vx = (x - prev.x) / dt
                            vy = (y - prev.y) / dt

                    self.obstacles[name] = VehicleState(
                        x=x, y=y, z=z,
                        roll=0.0, pitch=0.0, yaw=yaw,
                        vx=vx, vy=vy, vz=0.0,
                        omega=0.0,
                        timestamp=timestamp,
                        valid=True
                    )
            except Exception as e:
                logger.error(f"Error tracking obstacle {name}: {e}")

    def get_obstacles(self) -> Dict[str, VehicleState]:
        """Get all tracked obstacles."""
        return self.obstacles.copy()
