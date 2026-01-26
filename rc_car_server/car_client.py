"""
UDP Client for sending commands to the RC Car.

This module handles communication with the RC car's receiver
running on the Raspberry Pi.
"""

import socket
import json
import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass
import logging

from rc_car_server.mpc_controller import CarCommand

logger = logging.getLogger(__name__)


@dataclass
class CarClientConfig:
    """Configuration for the car client."""
    car_ip: str = "192.168.1.105"  # RC car Raspberry Pi IP
    car_port: int = 12345          # UDP port
    send_rate: float = 50.0        # Commands per second
    timeout: float = 0.5           # Socket timeout
    heartbeat_interval: float = 0.1  # Keep-alive interval


class RCCarClient:
    """
    UDP client for sending commands to the RC car.

    The client sends JSON-encoded commands to the car's receiver.
    It supports both manual command sending and automatic rate-limited sending.

    Usage:
        client = RCCarClient(car_ip="192.168.1.105")
        client.connect()

        # Send a single command
        client.send_command(CarCommand(throttle=0.5, steering=90))

        # Or use the command callback for continuous sending
        client.start_sending(command_callback=controller.compute_command)

        client.stop()
    """

    def __init__(
        self,
        car_ip: str = "192.168.1.105",
        car_port: int = 12345,
        send_rate: float = 50.0
    ):
        """
        Initialize the car client.

        Args:
            car_ip: IP address of the Raspberry Pi
            car_port: UDP port (default 12345)
            send_rate: Command send rate in Hz
        """
        self.config = CarClientConfig(
            car_ip=car_ip,
            car_port=car_port,
            send_rate=send_rate
        )

        self.socket: Optional[socket.socket] = None
        self._running = False
        self._send_thread: Optional[threading.Thread] = None
        self._command_callback: Optional[Callable[[], CarCommand]] = None
        self._last_command: Optional[CarCommand] = None
        self._lock = threading.Lock()

        # Statistics
        self.commands_sent = 0
        self.send_errors = 0
        self.last_send_time = 0.0

        logger.info(f"RCCarClient initialized: {car_ip}:{car_port} @ {send_rate}Hz")

    def connect(self) -> bool:
        """
        Create UDP socket for sending commands.

        Returns:
            True if socket created successfully
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.config.timeout)
            logger.info(f"UDP socket created for {self.config.car_ip}:{self.config.car_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to create socket: {e}")
            return False

    def disconnect(self):
        """Close the UDP socket."""
        self.stop()
        if self.socket is not None:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        logger.info("Car client disconnected")

    def send_command(self, command: CarCommand) -> bool:
        """
        Send a command to the RC car.

        Args:
            command: CarCommand with throttle and steering

        Returns:
            True if command sent successfully
        """
        if self.socket is None:
            logger.error("Socket not connected")
            return False

        try:
            # Create JSON message
            message = json.dumps(command.to_dict())

            # Send via UDP
            self.socket.sendto(
                message.encode('utf-8'),
                (self.config.car_ip, self.config.car_port)
            )

            self.commands_sent += 1
            self.last_send_time = time.time()

            with self._lock:
                self._last_command = command

            logger.debug(f"Sent: throttle={command.throttle:.2f}, steering={command.steering:.1f}")
            return True

        except Exception as e:
            self.send_errors += 1
            logger.error(f"Failed to send command: {e}")
            return False

    def start_sending(self, command_callback: Callable[[], CarCommand]):
        """
        Start continuous command sending in background thread.

        Args:
            command_callback: Function that returns the next CarCommand
        """
        if self._running:
            logger.warning("Already sending")
            return

        self._command_callback = command_callback
        self._running = True
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()
        logger.info(f"Started sending commands at {self.config.send_rate}Hz")

    def stop(self):
        """Stop continuous command sending."""
        self._running = False
        if self._send_thread is not None:
            self._send_thread.join(timeout=1.0)
            self._send_thread = None

        # Send stop command
        if self.socket is not None:
            try:
                stop_cmd = CarCommand(throttle=0.0, steering=90.0, timestamp=time.time())
                self.send_command(stop_cmd)
            except:
                pass

        logger.info("Stopped sending commands")

    def _send_loop(self):
        """Background thread that sends commands at fixed rate."""
        interval = 1.0 / self.config.send_rate

        while self._running:
            loop_start = time.time()

            try:
                if self._command_callback is not None:
                    command = self._command_callback()
                    self.send_command(command)
                elif self._last_command is not None:
                    # Resend last command as heartbeat
                    with self._lock:
                        self.send_command(self._last_command)

            except Exception as e:
                logger.error(f"Send loop error: {e}")

            # Sleep to maintain rate
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def send_stop(self):
        """Send emergency stop command."""
        stop_cmd = CarCommand(throttle=0.0, steering=90.0, timestamp=time.time())
        for _ in range(5):  # Send multiple times for reliability
            self.send_command(stop_cmd)
            time.sleep(0.01)
        logger.info("Emergency stop sent")

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            'commands_sent': self.commands_sent,
            'send_errors': self.send_errors,
            'last_send_time': self.last_send_time,
            'is_running': self._running
        }


class MockCarClient(RCCarClient):
    """
    Mock car client for testing without actual hardware.

    Records commands instead of sending them over network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_history: list[CarCommand] = []
        self.max_history = 1000

    def connect(self) -> bool:
        """Mock connect always succeeds."""
        logger.info("Mock car client connected (no actual network)")
        return True

    def send_command(self, command: CarCommand) -> bool:
        """Record command instead of sending."""
        self.command_history.append(command)
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]

        self.commands_sent += 1
        self.last_send_time = time.time()

        with self._lock:
            self._last_command = command

        logger.debug(f"Mock sent: throttle={command.throttle:.2f}, steering={command.steering:.1f}")
        return True

    def get_command_history(self) -> list[CarCommand]:
        """Get recorded command history."""
        return self.command_history.copy()
