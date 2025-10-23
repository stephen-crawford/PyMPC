"""
Dynamics models for MPC planning.

This module contains various vehicle dynamics models including:
- Overactuated models for exact control
- Car dynamics (bicycle model)
- Kinematic models
"""

import numpy as np
import casadi as cs
from typing import List, Tuple
from abc import ABC, abstractmethod


class BaseDynamics(ABC):
    """Abstract base class for vehicle dynamics models."""
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize dynamics model.
        
        Args:
            dt: Time step
        """
        self.dt = dt
        self.nx = 0  # Number of states
        self.nu = 0  # Number of control inputs
        self.state_names: List[str] = []
        self.input_names: List[str] = []
        
        # Bounds
        self.x_lb: List[float] = []
        self.x_ub: List[float] = []
        self.u_lb: List[float] = []
        self.u_ub: List[float] = []
    
    @abstractmethod
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous time dynamics: dx/dt = f(x, u).
        
        Args:
            x: State vector
            u: Control input vector
            
        Returns:
            State derivative
        """
    
    def discrete_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Discrete time dynamics using RK4 integration.
        
        Args:
            x: Current state
            u: Control input
            
        Returns:
            Next state
        """
        # RK4 integration
        k1 = self.continuous_dynamics(x, u)
        k2 = self.continuous_dynamics(x + self.dt/2 * k1, u)
        k3 = self.continuous_dynamics(x + self.dt/2 * k2, u)
        k4 = self.continuous_dynamics(x + self.dt * k3, u)
        
        return x + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def get_state_bounds(self) -> Tuple[List[float], List[float]]:
        """Get state bounds."""
        return self.x_lb, self.x_ub
    
    def get_input_bounds(self) -> Tuple[List[float], List[float]]:
        """Get input bounds."""
        return self.u_lb, self.u_ub
    
    def get_state_index(self, state_name: str) -> int:
        """Get index of state variable."""
        if state_name not in self.state_names:
            raise ValueError(f"State {state_name} not found. Available: {self.state_names}")
        return self.state_names.index(state_name)
    
    def get_input_index(self, input_name: str) -> int:
        """Get index of input variable."""
        if input_name not in self.input_names:
            raise ValueError(f"Input {input_name} not found. Available: {self.input_names}")
        return self.input_names.index(input_name)


class OveractuatedPointMass(BaseDynamics):
    """
    Overactuated point mass model for exact control.
    
    States: [x, y, vx, vy]
    Inputs: [fx, fy] (direct force control)
    """
    
    def __init__(self, dt: float = 0.1, mass: float = 1.0):
        """
        Initialize overactuated point mass model.
        
        Args:
            dt: Time step
            mass: Vehicle mass
        """
        super().__init__(dt)
        self.mass = mass
        
        # State: [x, y, vx, vy]
        self.nx = 4
        self.state_names = ["x", "y", "vx", "vy"]
        
        # Input: [fx, fy] (direct force control)
        self.nu = 2
        self.input_names = ["fx", "fy"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, -20.0, -20.0]
        self.x_ub = [1000.0, 1000.0, 20.0, 20.0]
        self.u_lb = [-50.0, -50.0]
        self.u_ub = [50.0, 50.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for overactuated point mass.
        
        dx/dt = vx
        dy/dt = vy
        dvx/dt = fx/m
        dvy/dt = fy/m
        """
        vx, vy = x[2], x[3]
        fx, fy = u[0], u[1]
        
        return cs.vertcat(
            vx,                    # dx/dt
            vy,                    # dy/dt
            fx / self.mass,        # dvx/dt
            fy / self.mass         # dvy/dt
        )


class OveractuatedUnicycle(BaseDynamics):
    """
    Overactuated unicycle model for exact control.
    
    States: [x, y, psi, v, omega]
    Inputs: [ax, ay] (direct acceleration control)
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize overactuated unicycle model.
        
        Args:
            dt: Time step
        """
        super().__init__(dt)
        
        # State: [x, y, psi, v, omega]
        self.nx = 5
        self.state_names = ["x", "y", "psi", "v", "omega"]
        
        # Input: [ax, ay] (direct acceleration control)
        self.nu = 2
        self.input_names = ["ax", "ay"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, -np.pi*4, 0.0, -5.0]
        self.x_ub = [1000.0, 1000.0, np.pi*4, 20.0, 5.0]
        self.u_lb = [-10.0, -10.0]
        self.u_ub = [10.0, 10.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for overactuated unicycle.
        
        dx/dt = v*cos(psi)
        dy/dt = v*sin(psi)
        dpsi/dt = omega
        dv/dt = ax*cos(psi) + ay*sin(psi)
        domega/dt = (-ax*sin(psi) + ay*cos(psi))/v (if v > 0)
        """
        psi, v, omega = x[2], x[3], x[4]
        ax, ay = u[0], u[1]
        
        # Avoid division by zero
        v_safe = cs.fmax(v, 0.01)
        
        return cs.vertcat(
            v * cs.cos(psi),                                    # dx/dt
            v * cs.sin(psi),                                    # dy/dt
            omega,                                              # dpsi/dt
            ax * cs.cos(psi) + ay * cs.sin(psi),               # dv/dt
            (-ax * cs.sin(psi) + ay * cs.cos(psi)) / v_safe    # domega/dt
        )


class BicycleModel(BaseDynamics):
    """
    Bicycle model for car dynamics.
    
    States: [x, y, psi, v, delta]
    Inputs: [a, delta_dot] (acceleration, steering rate)
    """
    
    def __init__(self, dt: float = 0.1, wheelbase: float = 2.79, 
                 max_steering_angle: float = 0.5):
        """
        Initialize bicycle model.
        
        Args:
            dt: Time step
            wheelbase: Vehicle wheelbase
            max_steering_angle: Maximum steering angle
        """
        super().__init__(dt)
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
        
        # State: [x, y, psi, v, delta]
        self.nx = 5
        self.state_names = ["x", "y", "psi", "v", "delta"]
        
        # Input: [a, delta_dot]
        self.nu = 2
        self.input_names = ["a", "delta_dot"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, -np.pi*4, 0.0, -max_steering_angle]
        self.x_ub = [1000.0, 1000.0, np.pi*4, 20.0, max_steering_angle]
        self.u_lb = [-5.0, -2.0]
        self.u_ub = [5.0, 2.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for bicycle model.
        
        dx/dt = v*cos(psi + beta)
        dy/dt = v*sin(psi + beta)
        dpsi/dt = (v/L)*sin(beta)
        dv/dt = a
        ddelta/dt = delta_dot
        
        where beta = atan(tan(delta)/2) for rear-wheel steering
        """
        psi, v, delta = x[2], x[3], x[4]
        a, delta_dot = u[0], u[1]
        
        # Slip angle
        beta = cs.atan(cs.tan(delta) / 2.0)
        
        return cs.vertcat(
            v * cs.cos(psi + beta),     # dx/dt
            v * cs.sin(psi + beta),     # dy/dt
            (v / self.wheelbase) * cs.sin(beta),  # dpsi/dt
            a,                          # dv/dt
            delta_dot                   # ddelta/dt
        )


class KinematicModel(BaseDynamics):
    """
    Kinematic model (simplified bicycle model).
    
    States: [x, y, psi, v]
    Inputs: [a, omega] (acceleration, angular velocity)
    """
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize kinematic model.
        
        Args:
            dt: Time step
        """
        super().__init__(dt)
        
        # State: [x, y, psi, v]
        self.nx = 4
        self.state_names = ["x", "y", "psi", "v"]
        
        # Input: [a, omega]
        self.nu = 2
        self.input_names = ["a", "omega"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, -np.pi*4, 0.0]
        self.x_ub = [1000.0, 1000.0, np.pi*4, 20.0]
        self.u_lb = [-5.0, -2.0]
        self.u_ub = [5.0, 2.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for kinematic model.
        
        dx/dt = v*cos(psi)
        dy/dt = v*sin(psi)
        dpsi/dt = omega
        dv/dt = a
        """
        psi, v = x[2], x[3]
        a, omega = u[0], u[1]
        
        return cs.vertcat(
            v * cs.cos(psi),    # dx/dt
            v * cs.sin(psi),    # dy/dt
            omega,              # dpsi/dt
            a                   # dv/dt
        )


class ContouringBicycleModel(BaseDynamics):
    """
    Bicycle model with path parameter for contouring control.
    
    States: [x, y, psi, v, delta, s]
    Inputs: [a, delta_dot] (acceleration, steering rate)
    """
    
    def __init__(self, dt: float = 0.1, wheelbase: float = 2.79, 
                 max_steering_angle: float = 0.5):
        """
        Initialize contouring bicycle model.
        
        Args:
            dt: Time step
            wheelbase: Vehicle wheelbase
            max_steering_angle: Maximum steering angle
        """
        super().__init__(dt)
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
        
        # State: [x, y, psi, v, delta, s]
        self.nx = 6
        self.state_names = ["x", "y", "psi", "v", "delta", "s"]
        
        # Input: [a, delta_dot]
        self.nu = 2
        self.input_names = ["a", "delta_dot"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, -np.pi*4, 0.0, -max_steering_angle, 0.0]
        self.x_ub = [1000.0, 1000.0, np.pi*4, 20.0, max_steering_angle, 10000.0]
        self.u_lb = [-5.0, -2.0]
        self.u_ub = [5.0, 2.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for contouring bicycle model.
        
        dx/dt = v*cos(psi + beta)
        dy/dt = v*sin(psi + beta)
        dpsi/dt = (v/L)*sin(beta)
        dv/dt = a
        ddelta/dt = delta_dot
        ds/dt = v
        """
        psi, v, delta = x[2], x[3], x[4]
        a, delta_dot = u[0], u[1]
        
        # Slip angle
        beta = cs.atan(cs.tan(delta) / 2.0)
        
        return cs.vertcat(
            v * cs.cos(psi + beta),     # dx/dt
            v * cs.sin(psi + beta),     # dy/dt
            (v / self.wheelbase) * cs.sin(beta),  # dpsi/dt
            a,                          # dv/dt
            delta_dot,                  # ddelta/dt
            v                           # ds/dt (path parameter)
        )


class QuadrotorModel(BaseDynamics):
    """
    Quadrotor dynamics model for 3D flight.
    
    States: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    Inputs: [thrust, tau_phi, tau_theta, tau_psi]
    """
    
    def __init__(self, dt: float = 0.1, mass: float = 1.0, 
                 inertia: np.ndarray = None):
        """
        Initialize quadrotor model.
        
        Args:
            dt: Time step
            mass: Quadrotor mass
            inertia: Inertia matrix (3x3)
        """
        super().__init__(dt)
        self.mass = mass
        self.inertia = inertia if inertia is not None else np.eye(3)
        
        # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.nx = 12
        self.state_names = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]
        
        # Input: [thrust, tau_phi, tau_theta, tau_psi]
        self.nu = 4
        self.input_names = ["thrust", "tau_phi", "tau_theta", "tau_psi"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, 0.0, -20.0, -20.0, -20.0, -np.pi, -np.pi, -np.pi, -10.0, -10.0, -10.0]
        self.x_ub = [1000.0, 1000.0, 100.0, 20.0, 20.0, 20.0, np.pi, np.pi, np.pi, 10.0, 10.0, 10.0]
        self.u_lb = [0.0, -5.0, -5.0, -5.0]
        self.u_ub = [20.0, 5.0, 5.0, 5.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for quadrotor model.
        
        Args:
            x: State vector [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
            u: Control input [thrust, tau_phi, tau_theta, tau_psi]
            
        Returns:
            State derivative
        """
        # Extract states
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]
        
        # Extract inputs
        thrust = u[0]
        tau_phi, tau_theta, tau_psi = u[1], u[2], u[3]
        
        # Rotation matrix from body to world frame
        R = self._rotation_matrix(phi, theta, psi)
        
        # Gravity vector
        g = 9.81
        
        # Dynamics
        dx_dt = vx
        dy_dt = vy
        dz_dt = vz
        
        # Linear acceleration in world frame
        accel_world = R @ cs.vertcat(0, 0, thrust/self.mass) - cs.vertcat(0, 0, g)
        dvx_dt = accel_world[0]
        dvy_dt = accel_world[1]
        dvz_dt = accel_world[2]
        
        # Angular velocity dynamics
        dphi_dt = p + q*cs.sin(phi)*cs.tan(theta) + r*cs.cos(phi)*cs.tan(theta)
        dtheta_dt = q*cs.cos(phi) - r*cs.sin(phi)
        dpsi_dt = q*cs.sin(phi)/cs.cos(theta) + r*cs.cos(phi)/cs.cos(theta)
        
        # Angular acceleration (simplified)
        dp_dt = tau_phi / self.inertia[0, 0]
        dq_dt = tau_theta / self.inertia[1, 1]
        dr_dt = tau_psi / self.inertia[2, 2]
        
        return cs.vertcat(
            dx_dt, dy_dt, dz_dt,
            dvx_dt, dvy_dt, dvz_dt,
            dphi_dt, dtheta_dt, dpsi_dt,
            dp_dt, dq_dt, dr_dt
        )
    
    def _rotation_matrix(self, phi: cs.SX, theta: cs.SX, psi: cs.SX) -> cs.SX:
        """Compute rotation matrix from Euler angles."""
        c_phi, s_phi = cs.cos(phi), cs.sin(phi)
        c_theta, s_theta = cs.cos(theta), cs.sin(theta)
        c_psi, s_psi = cs.cos(psi), cs.sin(psi)
        
        R = cs.vertcat(
            cs.horzcat(c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi),
            cs.horzcat(c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi),
            cs.horzcat(-s_theta, s_phi*c_theta, c_phi*c_theta)
        )
        
        return R


class AckermannModel(BaseDynamics):
    """
    Ackermann steering model for car-like vehicles.
    
    States: [x, y, psi, v, delta]
    Inputs: [a, delta_dot]
    """
    
    def __init__(self, dt: float = 0.1, wheelbase: float = 2.79, 
                 max_steering_angle: float = 0.5):
        """
        Initialize Ackermann model.
        
        Args:
            dt: Time step
            wheelbase: Vehicle wheelbase
            max_steering_angle: Maximum steering angle
        """
        super().__init__(dt)
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
        
        # State: [x, y, psi, v, delta]
        self.nx = 5
        self.state_names = ["x", "y", "psi", "v", "delta"]
        
        # Input: [a, delta_dot]
        self.nu = 2
        self.input_names = ["a", "delta_dot"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, -np.pi*4, 0.0, -max_steering_angle]
        self.x_ub = [1000.0, 1000.0, np.pi*4, 20.0, max_steering_angle]
        self.u_lb = [-5.0, -2.0]
        self.u_ub = [5.0, 2.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for Ackermann model.
        
        Args:
            x: State vector [x, y, psi, v, delta]
            u: Control input [a, delta_dot]
            
        Returns:
            State derivative
        """
        psi, v, delta = x[2], x[3], x[4]
        a, delta_dot = u[0], u[1]
        
        # Ackermann steering geometry
        beta = cs.atan(cs.tan(delta) / 2.0)
        
        return cs.vertcat(
            v * cs.cos(psi + beta),     # dx/dt
            v * cs.sin(psi + beta),     # dy/dt
            (v / self.wheelbase) * cs.sin(beta),  # dpsi/dt
            a,                          # dv/dt
            delta_dot                   # ddelta/dt
        )


class DifferentialDriveModel(BaseDynamics):
    """
    Differential drive model for mobile robots.
    
    States: [x, y, theta, v_left, v_right]
    Inputs: [a_left, a_right]
    """
    
    def __init__(self, dt: float = 0.1, wheel_radius: float = 0.1, 
                 wheel_base: float = 0.5):
        """
        Initialize differential drive model.
        
        Args:
            dt: Time step
            wheel_radius: Wheel radius
            wheel_base: Distance between wheels
        """
        super().__init__(dt)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        
        # State: [x, y, theta, v_left, v_right]
        self.nx = 5
        self.state_names = ["x", "y", "theta", "v_left", "v_right"]
        
        # Input: [a_left, a_right]
        self.nu = 2
        self.input_names = ["a_left", "a_right"]
        
        # Bounds
        self.x_lb = [-1000.0, -1000.0, -np.pi*4, -10.0, -10.0]
        self.x_ub = [1000.0, 1000.0, np.pi*4, 10.0, 10.0]
        self.u_lb = [-5.0, -5.0]
        self.u_ub = [5.0, 5.0]
    
    def continuous_dynamics(self, x: cs.SX, u: cs.SX) -> cs.SX:
        """
        Continuous dynamics for differential drive model.
        
        Args:
            x: State vector [x, y, theta, v_left, v_right]
            u: Control input [a_left, a_right]
            
        Returns:
            State derivative
        """
        theta = x[2]
        v_left, v_right = x[3], x[4]
        a_left, a_right = u[0], u[1]
        
        # Linear and angular velocities
        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.wheel_base
        
        return cs.vertcat(
            v * cs.cos(theta),          # dx/dt
            v * cs.sin(theta),           # dy/dt
            omega,                       # dtheta/dt
            a_left,                      # dv_left/dt
            a_right                      # dv_right/dt
        )


def create_dynamics_model(model_type: str, **kwargs) -> BaseDynamics:
    """
    Factory function to create dynamics models.
    
    Args:
        model_type: Type of dynamics model
        **kwargs: Additional parameters
        
    Returns:
        Dynamics model instance
    """
    models = {
        "overactuated_point_mass": OveractuatedPointMass,
        "overactuated_unicycle": OveractuatedUnicycle,
        "bicycle": BicycleModel,
        "kinematic": KinematicModel,
        "contouring_bicycle": ContouringBicycleModel,
        "quadrotor": QuadrotorModel,
        "ackermann": AckermannModel,
        "differential_drive": DifferentialDriveModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)
