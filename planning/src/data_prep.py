import logging

import numpy as np

from planning.src.types import Disc, DynamicObstacle, Prediction, PredictionType, PredictionStep, State, \
    generate_reference_path
from utils.math_utils import distance
from utils.utils import CONFIG

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


def get_dummy_obstacle(state: State) -> DynamicObstacle:
    """Create a dummy obstacle far from the current state."""
    return DynamicObstacle(
        -1,
        np.array([state.get("x") + 100., state.get("y") + 100.]),
        0.,
        0.
    )


def propagate_obstacles(data, dt=0.1, horizon=10, speed=0, sigma_pos=0.2):
  if not data.dynamic_obstacles:
      return

  for obstacle in data.dynamic_obstacles:
      pred = obstacle.prediction
      path = getattr(pred, "path", None)

      # Fallback: constant velocity if no path
      if path is None:
          velocity = np.array([np.cos(obstacle.angle), np.sin(obstacle.angle)]) * speed
          obstacle.prediction = get_constant_velocity_prediction(obstacle.position, velocity, dt, horizon)
          continue

      total_length = path.s[-1]

      # Initialize progress in arc length
      if not hasattr(obstacle, "s"):
          obstacle.s = 0.0
      obstacle.s += speed * dt

      # If reached end of current path → generate new path
      if obstacle.s >= total_length:
          start = [path.x[-1], path.y[-1], path.z[-1] if hasattr(path, "z") else 0.0]

          # Choose new goal
          if hasattr(obstacle, "road_to_follow"):
              road = obstacle.road_to_follow
              s_offset = np.random.uniform(0, road.length)
              x_center = road.x_spline(s_offset)
              y_center = road.y_spline(s_offset)

              # Tangent & lateral offset to simulate lane variation
              dx = road.x_spline(min(s_offset + 0.1, road.length)) - x_center
              dy = road.y_spline(min(s_offset + 0.1, road.length)) - y_center
              tangent = np.array([dx, dy]) / np.linalg.norm([dx, dy])
              normal = np.array([-tangent[1], tangent[0]])
              lateral_offset = np.random.uniform(-3.0, 3.0)
              goal = [x_center + lateral_offset * normal[0], y_center + lateral_offset * normal[1], 0.0]
          else:
              goal = [
                  np.random.uniform(start[0] - 20, start[0] + 20),
                  np.random.uniform(start[1] - 20, start[1] + 20),
                  0.0
              ]

          # Generate new reference path
          new_path = generate_reference_path(
              start, goal, path_type=np.random.choice(["straight", "curved", "s-turn", "circle"]),
              num_points=10
          )

          obstacle.prediction.path = new_path
          obstacle.s = 0.0
          path = new_path
          total_length = path.s[-1]

      # ✅ Compute position using arc-length splines
      s_now = min(obstacle.s, total_length)
      x = path.x_spline(s_now)
      y = path.y_spline(s_now)
      z = path.z_spline(s_now) if path.z_spline else 0.0
      obstacle.position = np.array([x, y, z])

      # ✅ Compute heading
      ds = 0.1
      s_next = min(s_now + ds, total_length)
      dx = path.x_spline(s_next) - x
      dy = path.y_spline(s_next) - y
      obstacle.angle = np.arctan2(dy, dx)

      # ✅ Build prediction horizon
      pred_steps = []
      s_future = s_now
      for _ in range(horizon):
          s_future = min(s_future + speed * dt, total_length)
          px = path.x_spline(s_future)
          py = path.y_spline(s_future)
          pz = path.z_spline(s_future) if path.z_spline else 0.0

          s_next = min(s_future + ds, total_length)
          dx_f = path.x_spline(s_next) - px
          dy_f = path.y_spline(s_next) - py
          angle = np.arctan2(dy_f, dx_f)

          pos = np.array([px, py, pz])

          # Add noise & uncertainty
          if pred.type == PredictionType.GAUSSIAN:
              pos += np.random.normal(0, sigma_pos, size=pos.shape)
              major_r, minor_r = sigma_pos * 2, sigma_pos
          elif pred.type == PredictionType.NONGAUSSIAN:
              pos += np.random.standard_t(df=3, size=pos.shape) * sigma_pos
              major_r, minor_r = sigma_pos * 3, sigma_pos * 1.5
          else:
              major_r, minor_r = 0.1, 0.1

          pred_steps.append(PredictionStep(pos, angle, major_r, minor_r))

      pred.steps = pred_steps

def get_constant_velocity_prediction(position: np.ndarray, velocity: np.ndarray, dt: float, steps: int) -> Prediction:
    """Generate prediction based on constant velocity model."""
    if CONFIG["probabilistic"]["enable"]:
        prediction = Prediction(PredictionType.GAUSSIAN)
        noise = 0.3
    else:
        prediction = Prediction(PredictionType.DETERMINISTIC)
        noise = 0.

    # Initialize the modes list if it doesn't exist
    if not prediction.steps:
        prediction.steps = []

    for i in range(steps):
        prediction.steps.append(PredictionStep(
            position + velocity * dt * i,
            0.,
            noise,
            noise
        ))

    if CONFIG["probabilistic"]["enable"]:
        propagate_prediction_uncertainty(prediction)

    return prediction

def remove_distant_obstacles(obstacles: list[DynamicObstacle], state: 'State') -> None:
    """Remove obstacles that are far from the current state."""
    nearby_obstacles = []

    pos = state.get_position()
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
                    obstacle.prediction.steps[k].position,
                    state.get_position() + state.get("v") * k * direction
                )

                if dist < min_dist:
                    min_dist = dist

            distances.append(min_dist)

        # Sort obstacles on distance
        indices.sort(key=lambda j: distances[j])

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

    for k in range(10):
        major = np.sqrt(major ** 2 + (prediction.steps[k].major_radius * dt) ** 2)
        minor = np.sqrt(minor ** 2 + (prediction.steps[k].minor_radius * dt) ** 2)
        prediction.steps[k].major_radius += major # This was originally straight assignment not addition
        prediction.steps[k].minor_radius += minor

def propagate_prediction_uncertainty_for_obstacles(obstacles: list[DynamicObstacle]) -> None:
    """Propagate uncertainty for all obstacles."""
    for obstacle in obstacles:
        propagate_prediction_uncertainty(obstacle.prediction)