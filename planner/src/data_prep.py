from typing import Any

import numpy as np
from enum import Enum
import logging

from numpy import floating

from planner.src.types import Disc, DynamicObstacle, Prediction, PredictionType, PredictionStep
from utils.utils import CONFIG, MOCKED_CONFIG


# Assuming these are already defined elsewhere based on your code snippet:
# - State class
# - CONFIG dictionary

def define_robot_area(length: float, width: float, n_discs: int) -> list[Disc]:
    """Define the robot area using discs."""
    # Where is the center w.r.t. the back of the vehicle
    center_offset = length / 2.  # Could become a parameter
    radius = width / 2.

    robot_area = []
    assert n_discs > 0, "Trying to create a collision region with less than a disc"

    if n_discs == 1:
        robot_area.append(Disc(0., radius))
    else:
        for i in range(n_discs):
            if i == 0:
                # First disc at the back of the car
                robot_area.append(Disc(-center_offset + radius, radius))
            elif i == n_discs - 1:
                # Last disc at the front of the car
                robot_area.append(Disc(-center_offset + length - radius, radius))
            else:
                # Other discs in between
                offset = -center_offset + radius + (i * (length - 2. * radius) / (n_discs - 1.))
                robot_area.append(Disc(offset, radius))

            logging.debug(f"Disc {i}: offset {robot_area[-1].offset}, radius {robot_area[-1].radius}")

    return robot_area


def get_dummy_obstacle(state: 'State') -> DynamicObstacle:
    """Create a dummy obstacle far from the current state."""
    return DynamicObstacle(
        -1,
        np.array([state.get("x") + 100., state.get("y") + 100.]),
        0.,
        0.
    )


def get_constant_velocity_prediction(position: np.ndarray, velocity: np.ndarray, dt: float, steps: int) -> Prediction:
    """Generate prediction based on constant velocity model."""
    if CONFIG["probabilistic"]["enable"]:
        prediction = Prediction(PredictionType.GAUSSIAN)
        noise = 0.3
    else:
        prediction = Prediction(PredictionType.DETERMINISTIC)
        noise = 0.

    # Initialize the modes list if it doesn't exist
    if not prediction.modes:
        prediction.modes.append([])

    for i in range(steps):
        prediction.modes[0].append(PredictionStep(
            position + velocity * dt * i,
            0.,
            noise,
            noise
        ))

    if CONFIG["probabilistic"]["enable"]:
        propagate_prediction_uncertainty(prediction)

    return prediction


def distance(a: np.ndarray, b: np.ndarray) -> floating[Any]:
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(a - b)


def remove_distant_obstacles(obstacles: list[DynamicObstacle], state: 'State') -> None:
    """Remove obstacles that are far from the current state."""
    nearby_obstacles = []

    pos = state.get_pos()
    for obstacle in obstacles:
        if distance(pos, obstacle.position) < CONFIG["max_obstacle_distance"]:
            nearby_obstacles.append(obstacle)

    obstacles.clear()
    obstacles.extend(nearby_obstacles)


def ensure_obstacle_size(obstacles: list[DynamicObstacle], state: 'State') -> None:
    """Ensure that the number of obstacles matches the configured maximum."""
    max_obstacles = CONFIG["max_obstacles"]
    # Create an index list
    indices = list(range(len(obstacles)))

    # If more, we sort and retrieve the closest obstacles
    if len(obstacles) > max_obstacles:
        distances = []
        logging.debug(f"Received {len(obstacles)} > {max_obstacles} obstacles. Keeping the closest.")

        for obstacle in obstacles:
            min_dist = 1e5
            direction = np.array([np.cos(state.get("psi")), np.sin(state.get("psi"))])

            for k in range(CONFIG["N"]):
                # Linearly scaled
                dist = (k + 1) * 0.6 * distance(
                    obstacle.prediction.modes[0][k].position,
                    state.get_pos() + state.get("v") * k * direction
                )

                if dist < min_dist:
                    min_dist = dist

            distances.append(min_dist)

        # Sort obstacles on distance
        indices.sort(key=lambda i: distances[i])

        # Keep the closest obstacles
        processed_obstacles = []

        for v in range(max_obstacles):
            processed_obstacles.append(obstacles[indices[v]])

        # Assign sequential IDs
        for i in range(len(processed_obstacles)):
            processed_obstacles[i].index = i

        obstacles.clear()
        obstacles.extend(processed_obstacles)

    elif len(obstacles) < max_obstacles:
        logging.debug(f"Received {len(obstacles)} < {max_obstacles} obstacles. Adding dummies.")

        for cur_size in range(len(obstacles), max_obstacles):
            dummy = get_dummy_obstacle(state)
            dummy.prediction = get_constant_velocity_prediction(
                dummy.position,
                np.array([0., 0.]),
                CONFIG["integrator_step"],
                CONFIG["N"]
            )
            obstacles.append(dummy)

    logging.debug(f"Obstacle size (after processing) is: {len(obstacles)}")


def propagate_prediction_uncertainty(prediction: Prediction) -> None:
    """Propagate uncertainty through the prediction steps."""
    if prediction.type != PredictionType.GAUSSIAN:
        return

    dt = CONFIG["integrator_step"]
    major = 0.
    minor = 0.

    for k in range(CONFIG["N"]):
        major = np.sqrt(major ** 2 + (prediction.modes[0][k].major_radius * dt) ** 2)
        minor = np.sqrt(minor ** 2 + (prediction.modes[0][k].minor_radius * dt) ** 2)
        prediction.modes[0][k].major_radius += major # This was originally straight assignment not addition
        prediction.modes[0][k].minor_radius += minor

def propagate_prediction_uncertainty_for_obstacles(obstacles: list[DynamicObstacle]) -> None:
    """Propagate uncertainty for all obstacles."""
    for obstacle in obstacles:
        propagate_prediction_uncertainty(obstacle.prediction)