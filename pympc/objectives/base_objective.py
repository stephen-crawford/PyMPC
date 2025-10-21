"""
Base objective class for MPC.

This module provides the abstract base class for all objective functions
used in the MPC framework.
"""

import casadi as ca
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np

from pympc.core.modules_manager import BaseModule
from pympc.utils.const import OBJECTIVE


class BaseObjective(BaseModule):
    """
    Abstract base class for MPC objectives.

    All objective types should inherit from this class and implement
    the required methods for objective computation.
    """

    def __init__(self, solver=None, name: str = None, weight: float = 1.0):
        """
        Initialize the objective.

        Args:
            solver: Solver instance (optional)
            name: Name of the objective
            weight: Weight of the objective
        """
        super().__init__(solver)
        self.name = name or self.__class__.__name__.lower()
        self.module_type = OBJECTIVE
        self.weight = weight
        self.active = True

    @abstractmethod
    def compute_casadi(self, X: ca.MX, U: ca.MX, opti: ca.Opti) -> ca.MX:
        """
        Compute the objective function using CasADi.

        Args:
            X: State variables
            U: Control variables
            opti: CasADi Opti object

        Returns:
            Objective expression
        """
        pass

    def is_active(self) -> bool:
        """
        Check if the objective is active.

        Returns:
            True if active, False otherwise
        """
        return self.active

    def set_active(self, active: bool) -> None:
        """
        Set the objective active status.

        Args:
            active: Whether the objective is active
        """
        self.active = active

    def set_weight(self, weight: float) -> None:
        """
        Set the objective weight.

        Args:
            weight: New weight value
        """
        self.weight = weight

    def get_weight(self) -> float:
        """
        Get the objective weight.

        Returns:
            Current weight value
        """
        return self.weight

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update objective parameters.

        Args:
            params: Dictionary of parameters
        """
        pass

    def is_data_ready(self, data: Dict[str, Any]) -> bool:
        """
        Check if required data is available.

        Args:
            data: Dictionary of available data

        Returns:
            True if data is ready, False otherwise
        """
        return True
