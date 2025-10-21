"""
Vehicle dynamics models for MPC.

This module provides various vehicle dynamics models that can be used
in the MPC framework.
"""

import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseDynamics(ABC):
    """
    Abstract base class for vehicle dynamics models.
    """

    def __init__(self, dt: float = 0.1):
        """
        Initialize the dynamics model.

        Args:
            dt: Time step
        """
        self.dt = dt

    @abstractmethod
    def get_state_dimension(self) -> int:
        """
        Get the dimension of the state vector.

        Returns:
            State dimension
        """
        pass

    @abstractmethod
    def get_control_dimension(self) -> int:
        """
        Get the dimension of the control vector.

        Returns:
            Control dimension
        """
        pass

    @abstractmethod
    def get_jacobian(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Jacobian matrices A and B for linearization.

        Args:
            x: State vector
            u: Control vector

        Returns:
            Tuple of (A, B) matrices
        """
        pass

    def discretize(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize the continuous-time system.

        Args:
            A: Continuous-time state matrix
            B: Continuous-time input matrix

        Returns:
            Tuple of (A_d, B_d) discrete-time matrices
        """
        n = A.shape[0]
        I = np.eye(n)
        
        # Matrix exponential for discretization
        # A_d = exp(A * dt)
        # B_d = A^(-1) * (exp(A * dt) - I) * B
        A_d = np.linalg.matrix_power(I + A * self.dt, 1)  # First-order approximation
        B_d = B * self.dt  # First-order approximation
        
        return A_d, B_d

    def simulate(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Simulate the system forward in time.

        Args:
            x0: Initial state
            u: Control sequence

        Returns:
            State trajectory
        """
        x = x0.copy()
        trajectory = [x.copy()]
        
        for u_k in u.T:
            x = self.step(x, u_k)
            trajectory.append(x.copy())
        
        return np.array(trajectory).T

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Single time step simulation.

        Args:
            x: Current state
            u: Current control

        Returns:
            Next state
        """
        # Get linearized dynamics
        A, B = self.get_jacobian(x, u)
        A_d, B_d = self.discretize(A, B)
        
        return A_d @ x + B_d @ u


class BicycleModel(BaseDynamics):
    """
    Bicycle model for vehicle dynamics.

    State: [x, y, psi, v]
    Control: [a, delta]
    """

    def __init__(self, dt: float = 0.1, wheelbase: float = 2.5):
        """
        Initialize the bicycle model.

        Args:
            dt: Time step
            wheelbase: Vehicle wheelbase
        """
        super().__init__(dt)
        self.wheelbase = wheelbase

    def get_state_dimension(self) -> int:
        """
        Get the dimension of the state vector.

        Returns:
            State dimension (4)
        """
        return 4

    def get_control_dimension(self) -> int:
        """
        Get the dimension of the control vector.

        Returns:
            Control dimension (2)
        """
        return 2

    def get_jacobian(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Jacobian matrices for linearization.

        Args:
            x: State vector [x, y, psi, v]
            u: Control vector [a, delta]

        Returns:
            Tuple of (A, B) matrices
        """
        x_pos, y_pos, psi, v = x
        a, delta = u

        # State matrix A
        A = np.array([
            [0, 0, -v * np.sin(psi), np.cos(psi)],
            [0, 0, v * np.cos(psi), np.sin(psi)],
            [0, 0, 0, np.tan(delta) / self.wheelbase],
            [0, 0, 0, 0]
        ])

        # Input matrix B
        B = np.array([
            [0, 0],
            [0, 0],
            [0, v / (self.wheelbase * np.cos(delta)**2)],
            [1, 0]
        ])

        return A, B

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Single time step simulation using bicycle model.

        Args:
            x: Current state [x, y, psi, v]
            u: Current control [a, delta]

        Returns:
            Next state
        """
        x_pos, y_pos, psi, v = x
        a, delta = u

        # Bicycle model equations
        x_next = x_pos + v * np.cos(psi) * self.dt
        y_next = y_pos + v * np.sin(psi) * self.dt
        psi_next = psi + v * np.tan(delta) / self.wheelbase * self.dt
        v_next = v + a * self.dt

        return np.array([x_next, y_next, psi_next, v_next])


class KinematicModel(BaseDynamics):
    """
    Simple kinematic model for vehicle dynamics.

    State: [x, y, psi]
    Control: [v, omega]
    """

    def __init__(self, dt: float = 0.1):
        """
        Initialize the kinematic model.

        Args:
            dt: Time step
        """
        super().__init__(dt)

    def get_state_dimension(self) -> int:
        """
        Get the dimension of the state vector.

        Returns:
            State dimension (3)
        """
        return 3

    def get_control_dimension(self) -> int:
        """
        Get the dimension of the control vector.

        Returns:
            Control dimension (2)
        """
        return 2

    def get_jacobian(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Jacobian matrices for linearization.

        Args:
            x: State vector [x, y, psi]
            u: Control vector [v, omega]

        Returns:
            Tuple of (A, B) matrices
        """
        x_pos, y_pos, psi = x
        v, omega = u

        # State matrix A
        A = np.array([
            [0, 0, -v * np.sin(psi)],
            [0, 0, v * np.cos(psi)],
            [0, 0, 0]
        ])

        # Input matrix B
        B = np.array([
            [np.cos(psi), 0],
            [np.sin(psi), 0],
            [0, 1]
        ])

        return A, B

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Single time step simulation using kinematic model.

        Args:
            x: Current state [x, y, psi]
            u: Current control [v, omega]

        Returns:
            Next state
        """
        x_pos, y_pos, psi = x
        v, omega = u

        # Kinematic model equations
        x_next = x_pos + v * np.cos(psi) * self.dt
        y_next = y_pos + v * np.sin(psi) * self.dt
        psi_next = psi + omega * self.dt

        return np.array([x_next, y_next, psi_next])
