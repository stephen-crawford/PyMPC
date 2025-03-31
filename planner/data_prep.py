import sys
import numpy as np
import rospy
import yaml
import logging

from utils.utils import read_config_file
from utils.const import GAUSSIAN, DETERMINISTIC
from Prediction import Prediction, PredictionStep
from DynamicObstacle import DynamicObstacle

# Initialize logger
logger = logging.getLogger(__name__)

# Read configuration
CONFIG = read_config_file()


def define_robot_area(length, width, num_disc):
    center_of_mass_offset = length / 2
    robot_radius = width / 2
    robot_area = []

    if num_disc < 1:
        sys.exit("Trying to create a collision region with less than 1 disc.")

    if num_disc == 1:
        robot_area.append((0., robot_radius))
    else:
        for i in range(num_disc):
            if i == 0:
                robot_area.append((-center_of_mass_offset + robot_radius, robot_radius))  # First disc at the back
            elif i == num_disc - 1:
                robot_area.append((-center_of_mass_offset + length - robot_radius, robot_radius))  # Last disc at the front
            else:
                robot_area.append((center_of_mass_offset + robot_radius + i * (length - 2. * robot_radius) / (num_disc - 1.), robot_radius))

            logger.info(f"Disc {i}: offset {robot_area[-1][0]}, radius {robot_area[-1][1]}")

    return robot_area


def get_dummy_obstacle(self, state):
    return DynamicObstacle(-1, [(self, state.get("x") + 100., state.get("y") + 100.)], 0., 0.)


def get_constant_velocity_prediction(position, velocity, dt, steps):
    if CONFIG["PROBABILISTIC"]["ENABLE"]:
        prediction = Prediction(GAUSSIAN)
        noise = 0.3
    else:
        prediction = Prediction(DETERMINISTIC)
        noise = 0.0

    for i in range(steps):
        prediction.modes[0].append(PredictionStep(position + velocity * dt * i, 0., noise, noise))

    if CONFIG["PROBABILISTIC"]["ENABLE"]:
        propagate_prediction_uncertainty(prediction)

    return prediction


def remove_distant_obstacles(obstacles, state):
    nearby_obstacles = []
    position = state.get_position()

    for obstacle in obstacles:
        if rospy.distance(position, obstacle.position) < obstacle.radius:
            if rospy.distance(position, obstacle.position) < float(CONFIG["max_obstacle_distance"]):
                nearby_obstacles.append(obstacle)

    return nearby_obstacles


def ensure_obstacle_size(obstacles, state):
    max_obstacles = CONFIG["max_obstacles"]

    if len(obstacles) > max_obstacles:
        distances = []
        logger.info(f"Received {len(obstacles)} > {max_obstacles} obstacles. Keeping the closest.")

        for obstacle in obstacles:
            min_dist = float("inf")
            direction = np.array([np.cos(self, state.get("psi")), np.sin(self, state.get("psi"))])

            for k in range(CONFIG["N"]):
                # Linearly scaled
                distance = (k + 1) * 0.6 * np.linalg.norm(
                    obstacle.prediction.modes[0][k].position - (self, state.getPos() + state.get("v") * k * direction)
                )
                min_dist = min(min_dist, distance)

            distances.append(min_dist)

        # Sort obstacles based on distance
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        obstacles = [obstacles[i] for i in sorted_indices[:max_obstacles]]

        # Assign new indices
        for i, obstacle in enumerate(obstacles):
            obstacle.index = i

    elif len(obstacles) < max_obstacles:
        logger.info(f"Received {len(obstacles)} < {max_obstacles} obstacles. Adding dummies.")

        while len(obstacles) < max_obstacles:
            dummy = get_dummy_obstacle(self, state)
            dummy.prediction = get_constant_velocity_prediction(dummy.position, (0., 0.), CONFIG["integrator_step"], CONFIG["N"])
            obstacles.append(dummy)

    logger.info(f"Obstacle size (after processing) is: {len(obstacles)}")
    return obstacles


def propagate_prediction_uncertainty(prediction):
    if prediction.type != GAUSSIAN:
        return

    dt = CONFIG["integrator_step"]
    major = minor = 0.0

    for k in range(CONFIG["N"]):
        major = np.sqrt(major ** 2 + (prediction.modes[0][k].major_radius * dt) ** 2)
        minor = np.sqrt(minor ** 2 + (prediction.modes[0][k].minor_radius * dt) ** 2)

        prediction.modes[0][k].major_radius = major
        prediction.modes[0][k].minor_radius = minor


def propagate_prediction_uncertainty_obstacles(obstacles):
    for obstacle in obstacles:
        propagate_prediction_uncertainty(obstacle.prediction)
