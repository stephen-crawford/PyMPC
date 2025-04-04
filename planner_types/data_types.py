import numpy as np
from typing import List, Optional


class Disc:
  def __init__(self, offset_: float, radius_: float):
    self.offset = offset_
    self.radius = radius_

  def get_position(self, robot_position: np.ndarray, angle: float) -> np.ndarray:
    return robot_position + np.array([self.offset * np.cos(angle), self.offset * np.sin(angle)])

  def to_robot_center(self, disc_position: np.ndarray, angle: float) -> np.ndarray:
    return disc_position - np.array([self.offset * np.cos(angle), self.offset * np.sin(angle)])


class Halfspace:
  def __init__(self, A: np.ndarray, b: float):
    self.A = A
    self.b = b


class PredictionStep:
  def __init__(self, position: np.ndarray, angle: float, major_radius: float, minor_radius: float):
    self.position = position
    self.angle = angle
    self.major_radius = major_radius
    self.minor_radius = minor_radius


class PredictionType:
  NONE = 0
  DETERMINISTIC = 1
  GAUSSIAN = 2


class Prediction:
  def __init__(self, type=PredictionType.NONE):
    self.type = type
    self.modes = []
    self.probabilities = []

    if type == PredictionType.DETERMINISTIC or type == PredictionType.GAUSSIAN:
      self.modes.append([])
      self.probabilities.append(1.0)

  def empty(self) -> bool:
    return len(self.modes) == 0 or (len(self.modes) > 0 and len(self.modes[0]) == 0)


class ObstacleType:
  # Define obstacle types as needed - the original code doesn't specify values
  STATIC = 0
  DYNAMIC = 1


class DynamicObstacle:
  def __init__(self, _index: int, _position: np.ndarray, _angle: float, _radius: float, _type: int):
    self.index = _index
    self.position = _position
    self.angle = _angle
    self.radius = _radius
    self.type = _type


class ReferencePath:
  def __init__(self, length: int):
    self.x = []
    self.y = []
    self.psi = []
    self.v = []
    self.s = []

  def clear(self):
    self.x.clear()
    self.y.clear()
    self.psi.clear()
    self.v.clear()
    self.s.clear()

  def point_in_path(self, point_num: int, other_x: float, other_y: float) -> bool:
    return self.x[point_num] == other_x and self.y[point_num] == other_y


class Trajectory:
  def __init__(self, dt: float, length: int):
    self.dt = dt
    self.positions = []

  def add(self, p_or_x, y=None):
    if y is None:
      # p is a vector
      self.positions.append(p_or_x)
    else:
      # x and y are coordinates
      self.positions.append(np.array([p_or_x, y]))


class FixedSizeTrajectory:
  def __init__(self, size: int):
    self._size = size
    self.positions = []

  def add(self, p: np.ndarray):
    # On jump, erase the trajectory
    if len(self.positions) > 0:
      distance = np.sqrt(np.sum((p - self.positions[-1]) ** 2))
      if distance > 5.0:
        self.positions.clear()
        self.positions.append(p)
        return

    if len(self.positions) < self._size:
      self.positions.append(p)
    else:
      self.positions.pop(0)
      self.positions.append(p)