import time
import logging
from solver import Solver
from rostools import Timer, DataSaver
from trajectory import Trajectory
from planner_output import PlannerOutput
from utils import CONFIG, BENCHMARKERS

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, modules):
        self._solver = Solver()
        self._solver.reset()
        self._modules = modules
        self._init_modules()
        self._startup_timer = Timer(1.0)
        self._was_reset = False
        self._experiment_util = None  # Assign appropriately
        self._output = None
        self._warmstart = None

    def _init_modules(self):
        for module in self._modules:
            module.initialize(self._solver)

    def solve_mpc(self, state, data):
        logger.info("planner.solve_mpc")
        was_feasible = self._output.success if self._output else False
        self._output = PlannerOutput(self._solver.dt, self._solver.N)
        module_data = {}

        is_data_ready = all(module.is_data_ready(data) for module in self._modules)
        if not is_data_ready:
            if self._startup_timer.has_finished():
                logger.warning("Data is not ready")
            self._output.success = False
            return self._output

        if self._was_reset:
            self._experiment_util.set_start_experiment()
            self._was_reset = False

        planning_benchmarker = BENCHMARKERS.get_benchmarker("planning")
        if planning_benchmarker.is_running():
            planning_benchmarker.cancel()
        planning_benchmarker.start()

        shift_forward = CONFIG["shift_previous_solution_forward"] and CONFIG["enable_output"]
        if was_feasible:
            self._solver.initialize_warmstart(self, state, shift_forward)
        else:
            self._solver.initialize_with_braking(self, state)

        self._solver.set_xinit(self, state)

        for module in self._modules:
            module.update(self, state, data, module_data)

        for k in range(self._solver.N):
            for module in self._modules:
                module.set_parameters(data, module_data, k)

        self._warmstart = Trajectory()
        for k in range(self._solver.N):
            self._warmstart.add(self._solver.get_ego_prediction(k, "x"),
                                self._solver.get_ego_prediction(k, "y"))
        self._solver.load_warmstart()

        used_time = time.time() - data.planning_start_time
        self._solver.params.solver_timeout = 1.0 / CONFIG["control_frequency"] - used_time - 0.006

        exit_flag = -1
        for module in self._modules:
            exit_flag = module.optimize(self, state, data, module_data)
            if exit_flag != -1:
                break

        if exit_flag == -1:
            exit_flag = self._solver.solve()

        planning_benchmarker.stop()

        if exit_flag != 1:
            self._output.success = False
            logger.warning(f"MPC failed: {self._solver.explain_exit_flag(exit_flag)}")
            return self._output

        self._output.success = True
        for k in range(1, self._solver.N):
            self._output.trajectory.add(self._solver.get_output(k, "x"),
                                        self._solver.get_output(k, "y"))

        if self._output.success and CONFIG["debug_limits"]:
            self._solver.print_if_bound_limited()

        logger.info("Planner::solveMPC done")
        return self._output

    def get_solution(self, k, var_name):
        return self._solver.get_output(k, var_name)

    def get_data_saver(self):
        return self._experiment_util.get_data_saver()

    def on_data_received(self, data, data_name):
        for module in self._modules:
            module.on_data_received(data, data_name)

    def visualize(self, state, data):
        logger.info("Planner::visualize")
        for module in self._modules:
            module.visualize(data)

        # Visualization methods need to be implemented
        # visualize_trajectory(self._output.trajectory, "planned_trajectory", True, 0.2)
        # Additional visualization calls...
        logger.info("Planner::visualize Done")

    def save_data(self, state, data):
        if not self._is_data_ready:
            return

        data_saver = self._experiment_util.get_data_saver()
        planning_time = BENCHMARKERS.get_benchmarker("planning").get_last()
        data_saver.add_data("runtime_control_loop", planning_time)
        if planning_time > 1.0 / CONFIG["control_frequency"]:
            logger.warning(f"Planning took too long: {planning_time} ms")
        data_saver.add_data("runtime_optimization", BENCHMARKERS.get_benchmarker("optimization").get_last())

        data_saver.add_data("status", 2. if self._output.success else 3.)
        for module in self._modules:
            module.save_data(data_saver)
        self._experiment_util.update(self, state, self._solver, data)

    def reset(self, state, data, success):
        if CONFIG["recording"]["enable"]:
            self._experiment_util.on_task_complete(success)

        self._solver.reset()
        for module in self._modules:
            module.reset()

        state.reset()
        data.reset()
        self._was_reset = True
        self._startup_timer.start()

    def is_objective_reached(self, state, data):
        return all(module.is_objective_reached(self, state, data) for module in self._modules)
