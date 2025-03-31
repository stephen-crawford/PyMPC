import logging
from utils.utils import read_config_file

# Initialize logger
logger = logging.getLogger(__name__)

# Read configuration
CONFIG = read_config_file()

_save_folder = CONFIG["recording"]["folder"]
_save_file = CONFIG["recording"]["file"]

class DataSaver:
    def __init__(self):
        self.data = {}
        self.add_timestamp = False

    def set_add_timestamp(self, value):
        self.add_timestamp = value

    def add_data(self, key, value):
        self.data[key] = value

    def save_data(self, folder, file):
        with open(f"{folder}/{file}", "w") as f:
            f.write(str(self.data))  # Example: Save as a string dictionary

    def get_file_path(self, folder, file, flag):
        return f"{folder}/{file}"


# Initialize DataSaver
_data_saver = DataSaver()
_data_saver.set_add_timestamp(CONFIG["recording"]["timestamp"])

if CONFIG["recording"]["enable"]:
    logger.info(f"Planner Save File: {_data_saver.get_file_path(_save_folder, _save_file, False)}")

_control_iteration = 0
_iteration_at_last_reset = 0
_experiment_counter = 0


def update(self, state, solver, data):
    global _control_iteration

    logger.info("planner.util.save_data()")

    if len(data.dynamic_obstacles) == 0:
        logger.info("Not exporting data: Obstacles not yet received.")
        return

    # Save vehicle data
    _data_saver.add_data("vehicle_pose", state.getPos())
    _data_saver.add_data("vehicle_orientation", state.get("psi"))

    # Save planned trajectory
    for k in range(CONFIG["N"]):
        _data_saver.add_data(f"vehicle_plan_{k}", solver.get_ego_prediction_position(k))

    # Save obstacle data
    for v, obstacle in enumerate(data.dynamic_obstacles):

        if obstacle.index is not None:
            _data_saver.add_data(f"obstacle_map_{v}", obstacle.index)
            _data_saver.add_data(f"obstacle_{v}_pose", obstacle.position)
            _data_saver.add_data(f"obstacle_{v}_orientation", obstacle.angle)

        # Save disc obstacle (assume only one disc)
        _data_saver.add_data("disc_0_pose", obstacle.position)
        _data_saver.add_data("disc_0_radius", obstacle.radius)
        _data_saver.add_data("disc_0_obstacle", v)

    _data_saver.add_data("max_intrusion", data.intrusion)
    _data_saver.add_data("metric_collisions", int(data.intrusion > 0.0))

    # Time keeping
    _data_saver.add_data("iteration", _control_iteration)
    _control_iteration += 1


def export_data():
    _data_saver.save_data(_save_folder, _save_file)


def on_task_complete(objective_reached):
    global _iteration_at_last_reset, _experiment_counter

    _data_saver.add_data("reset", _control_iteration)
    _data_saver.add_data(
        "metric_duration",
        (_control_iteration - _iteration_at_last_reset) * (1.0 / float(CONFIG["control_frequency"]))
    )
    _data_saver.add_data("metric_completed", int(objective_reached))

    _iteration_at_last_reset = _control_iteration
    _experiment_counter += 1

    num_experiments = int(CONFIG["recording"]["num_experiments"])
    if _experiment_counter % num_experiments == 0 and _experiment_counter > 0:
        export_data()

    if _experiment_counter >= num_experiments:
        logger.info(f"Completed {num_experiments} experiments.")
    else:
        logger.info(f"Starting experiment {_experiment_counter + 1} / {num_experiments}")

    assert _experiment_counter < num_experiments, "Stopping the planner."


def set_start_experiment():
    global _iteration_at_last_reset
    _iteration_at_last_reset = _control_iteration
