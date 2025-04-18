import sys
import os
import numpy as np

from solver_generator.solver_definition import define_parameters, objective, constraints, constraint_lower_bounds, \
    constraint_upper_bounds, constraint_number
from solver_generator.util.files import load_settings, write_to_yaml
from solver_generator.util.files import solver_path, solver_settings_path
from solver_generator.util.logging import print_success, print_header, print_path, print_warning
from solver_generator.solver_config import generate_rqt_reconfigure
from solver_generator.util.parameters import Parameters
from solver.casadisolver import CasADiSolver
from solver.osqpsolver import OSQPSolver

def generate_casadisolver(modules, settings, model, skipsolver_generator=False):
    """
    Generate a CasADi solver instance with the given settings and model.

    Args:
        modules: List of modules for the solver
        settings: Dictionary containing solver settings
        model: Model object with dynamics and constraints
        skipsolver_generator: Flag to skip generation if True

    Returns:
        solver: CasADi solver instance
        simulator: Simulator instance (may be the same as solver)
    """
    if skipsolver_generator:
        print_header("Output")
        print_warning("Solver generation was disabled by the command line option. Skipped.", no_tab=True)
        return None, None

    print_header(f"Creating CasADi Solver: {settings['name']}solver")

    # Create parameters for the solver
    params = Parameters()
    define_parameters(modules, params, settings)
    settings["params"] = params

    # Create solver with settings
    dt = settings.get("integrator_step", 0.1)
    N = settings.get("N", 20)

    # Initialize the solver with the appropriate parameters
    solver = CasADiSolver(dt=dt, N=N)

    # Set up model dimensions
    solver.nx = model.nx
    solver.nu = model.nu
    solver.nvar = model.get_nvar()
    solver.npar = params.length()

    # Set up bounds
    solver.lb = model.lower_bound
    solver.ub = model.upper_bound

    # Configure the objective and constraints for each stage
    for i in range(0, N):
        def objective_with_stage_index(stage_idx):
            return lambda z, p: objective(modules, z, p, model, settings, stage_idx)

        def constraints_with_stage_index(stage_idx):
            return lambda z, p: constraints(modules, z, p, model, settings, stage_idx)

        solver.set_stage_objective(i, objective_with_stage_index(i))

        # For all stages after the initial stage (k = 0)
        if i > 0:
            solver.set_stage_constraints(i, constraints_with_stage_index(i))
            solver.set_constraint_bounds(i,
                                         constraint_lower_bounds(modules),
                                         constraint_upper_bounds(modules),
                                         constraint_number(modules))
        else:
            solver.set_constraint_number(i, 0)  # No constraints for initial stage

    # Set up dynamics function
    solver.set_dynamics(lambda z, p: model.discrete_dynamics(z, p, settings))

    # Set up initial state indices
    solver.set_initial_state_indices(model.get_xinit())

    # Finalize the problem setup
    solver.finalize_problem()

    # For simulation, use the same solver
    simulator = solver

    print_header("Output")
    print_path("Solver", solver_path(settings), tab=True, end="")
    print_success(" -> generated")

    return solver, simulator


def generate_osqpsolver(modules, settings, model, skipsolver_generator=False):
    """
    Generate an OSQP solver instance with the given settings and model.

    Args:
        modules: List of modules for the solver
        settings: Dictionary containing solver settings
        model: Model object with dynamics and constraints
        skipsolver_generator: Flag to skip generation if True

    Returns:
        solver: OSQP solver instance
        simulator: Simulator instance (may be the same as solver)
    """
    if skipsolver_generator:
        print_header("Output")
        print_warning("Solver generation was disabled by the command line option. Skipped.", no_tab=True)
        return None, None

    print_header(f"Creating OSQP Solver: {settings['name']}solver")

    # Create parameters for the solver
    params = Parameters()
    define_parameters(modules, params, settings)
    settings["params"] = params

    # Create solver with settings
    dt = settings.get("integrator_step", 0.1)
    N = settings.get("N", 20)

    # Initialize the solver with the appropriate parameters
    solver = OSQPSolver(dt=dt, N=N)

    # Set up model dimensions
    solver.nx = model.nx
    solver.nu = model.nu
    solver.nvar = model.get_nvar()
    solver.npar = params.length()

    # Set up bounds
    solver.lb = model.lower_bound
    solver.ub = model.upper_bound

    # Configure the objective and constraints for each stage
    for i in range(0, N):
        def objective_with_stage_index(stage_idx):
            return lambda z, p: objective(modules, z, p, model, settings, stage_idx)

        def constraints_with_stage_index(stage_idx):
            return lambda z, p: constraints(modules, z, p, model, settings, stage_idx)

        solver.set_stage_objective(i, objective_with_stage_index(i))

        # For all stages after the initial stage (k = 0)
        if i > 0:
            solver.set_stage_constraints(i, constraints_with_stage_index(i))
            solver.set_constraint_bounds(i,
                                         constraint_lower_bounds(modules),
                                         constraint_upper_bounds(modules),
                                         constraint_number(modules))
        else:
            solver.set_constraint_number(i, 0)  # No constraints for initial stage

    # Set up dynamics function
    solver.set_dynamics(lambda z, p: model.discrete_dynamics(z, p, settings))

    # Set up initial state indices
    solver.set_initial_state_indices(model.get_xinit())

    # Finalize the problem setup
    solver.finalize_problem()

    # For simulation, use the same solver
    simulator = solver

    print_header("Output")
    print_path("Solver", solver_path(settings), tab=True, end="")
    print_success(" -> generated")

    return solver, simulator


def generatesolver(modules, model, settings=None):
    """
    Generate a solver based on settings.

    Args:
        modules: List of modules for the solver
        model: Model object with dynamics and constraints
        settings: Dictionary containing solver settings

    Returns:
        solver: Solver instance
        simulator: Simulator instance
    """
    # Parse command line arguments for skipping solver generation
    skipsolver_generator = len(sys.argv) > 1 and sys.argv[1].lower() == "false"
    print("Skip solver gen set to: " + str(skipsolver_generator))

    # Load settings if not provided
    if settings is None:
        settings = load_settings()

    # Ensure we have a valid solver type
    if settings["solver_settings"]["solver"] not in ["casadi", "osqp"]:
        raise IOError("Unknown solver specified in settings.yml (should be 'casadi' or 'osqp')")

    print_header(f"Creating {settings['solver_settings']['solver'].capitalize()} "
                 f"Solver: {settings['name']}solver")

    # Generate the appropriate solver
    solver = None
    simulator = None

    if settings["solver_settings"]["solver"] == "osqp":
        solver, simulator = generate_osqpsolver(modules, settings, model, skipsolver_generator)
    elif settings["solver_settings"]["solver"] == "casadi":
        solver, simulator = generate_casadisolver(modules, settings, model, skipsolver_generator)

    # Save parameter and model maps if solver was generated
    if solver and simulator:
        settings["params"].save_map()
        model.save_map()

        # Save solver settings
        solver_settings = {
            "N": settings["N"],
            "nx": model.nx,
            "nu": model.nu,
            "nvar": model.get_nvar(),
            "npar": settings["params"].length()
        }

        path = solver_settings_path()
        write_to_yaml(path, solver_settings)

        # Generate RQT reconfigure configuration
        generate_rqt_reconfigure(settings)

    return solver, simulator

