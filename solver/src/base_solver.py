import copy
from abc import ABC, abstractmethod

from planning.src.types import Data
from solver.src.modules_manager import ModuleManager
from solver.src.parameter_manager import ParameterManager
from utils.const import CONSTRAINT, OBJECTIVE
from utils.utils import LOG_DEBUG


class BaseSolver(ABC):
    def __init__(self, timestep=0.1, horizon=1):
        self.timestep = timestep
        self.horizon = horizon
        self.parameter_manager = ParameterManager()
        self.module_manager = ModuleManager()
        # Register default parameters
        self.define_parameters()
        self.parameter_manager.add("solver_timeout")
        self.parameter_manager.set_parameter("solver_timeout", 0.1)

        self.dynamics_model = None

    def get_module_manager(self):
        return self.module_manager

    def get_parameter_manager(self):
        return self.parameter_manager

    def get_ego_prediction(self, k, var):
        pass

    def get_reference_trajectory(self):
        pass

    def initialize(self, data):
        self.define_parameters()
        self.module_manager.set_parameters_all(self.parameter_manager, data, self.horizon)

    def initialize_rollout(self, state, shift_forward=True):
        pass

    def _initialize_base_rollout(self, state):
        pass

    def set_initial_state(self, state):
        pass

    def set_dynamics_model(self, dynamics_model):
        self.dynamics_model = dynamics_model

    def define_parameters(self):

        # Define parameters for objectives and constraints (in order)
        for module in self.module_manager.modules:
            if module.module_type == OBJECTIVE:
                module.define_parameters(self.parameter_manager)

        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                module.define_parameters(self.parameter_manager)

    def get_objective_cost(self, state, stage_idx):
        cost = self.module_manager.objective(state, self.parameter_manager, stage_idx)
        return cost

    def get_constraint_list(self, stage_idx):
        constraints = []

        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                constraints += module.get_constraints(self.dynamics_model, self.parameter_manager, stage_idx)

        return constraints

    def get_constraint_upper_bounds_list(self):
        ub = []
        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                ub += module.get_upper_bound()
        return ub


    def get_constraint_lower_bounds_list(self):
        lb = []
        for module in self.module_manager.modules:
            if module.module_type == CONSTRAINT:
                lb += module.get_lower_bound()
        return lb

    def on_data_received(self, data, data_name):
        for module in self.module_manager.modules:
            module.on_data_received(data, data_name)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_output(self, k, var_name):
        pass

    @abstractmethod
    def explain_exit_flag(self, code):
        pass


##### SOLVER MANAGEMENT
def deep_copy_solver(solver):
    """
    Create a proper deep copy of the solver object.

    This function handles special cases for solver objects that might
    have complex internal states or custom copying requirements.

    Args:
        solver: The solver object to copy

    Returns:
        A deep copy of the solver
    """
    try:
        if hasattr(solver, 'copy'):
            # Use the solver's custom copy method if available
            LOG_DEBUG("Using solver's custom copy method")
            return solver.copy()
        elif hasattr(solver, 'clone'):
            # Some solvers use 'clone' instead of 'copy'
            LOG_DEBUG("Using solver's clone method")
            return solver.clone()
        else:
            # Fall back to standard deep copy
            LOG_DEBUG("Using standard deep copy for solver")
            solver_copy = copy.deepcopy(solver)

            # Ensure the copy has all required attributes
            _verify_solver_copy(solver, solver_copy)
            return solver_copy
    except Exception as e:
        LOG_DEBUG(f"Error copying solver: {str(e)}")
        # Create a minimal working copy manually
        return _create_minimal_solver_copy(solver)


def _verify_solver_copy(original_solver, solver_copy):
    """
    Verify that the solver copy has all necessary attributes.

    Args:
        original_solver: The original solver
        solver_copy: The copied solver
    """
    # List of critical attributes that must be present
    critical_attributes = [
        'output', 'info', 'tmp_config', 'horizon', 'constraints'
    ]

    for attr in critical_attributes:
        if hasattr(original_solver, attr) and not hasattr(solver_copy, attr):
            LOG_DEBUG(f"WARNING: Copied solver missing attribute: {attr}")
            # Copy the attribute directly if missing
            setattr(solver_copy, attr, copy.deepcopy(getattr(original_solver, attr)))

    return solver_copy


def _create_minimal_solver_copy(solver):
    """
    Create a minimal working copy of the solver when deep copy fails.

    Args:
        solver: The original solver

    Returns:
        A minimal working copy of the solver
    """
    LOG_DEBUG("Creating minimal solver copy")

    # Create a new empty solver of the same class
    solver_class = solver.__class__
    minimal_solver = solver_class.__new__(solver_class)

    # Copy essential attributes
    essential_attrs = [
        'output', 'info', 'tmp_config', 'horizon', 'timestep',
        'constraints', 'cost', 'parameters'
    ]

    for attr in essential_attrs:
        if hasattr(solver, attr):
            try:
                setattr(minimal_solver, attr, copy.deepcopy(getattr(solver, attr)))
            except Exception as e:
                LOG_DEBUG(f"Error copying attribute {attr}: {str(e)}")
                # Try a shallow copy if deep copy fails
                setattr(minimal_solver, attr, getattr(solver, attr))

    return minimal_solver
