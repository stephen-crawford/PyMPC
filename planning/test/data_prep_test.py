import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import logging
import math

from planning.src.data_prep import propagate_prediction_uncertainty_for_obstacles, propagate_prediction_uncertainty, \
    ensure_obstacle_size, remove_distant_obstacles, get_constant_velocity_prediction, get_dummy_obstacle, \
    define_robot_area
from planning.src.types import Prediction, PredictionStep, PredictionType, DynamicObstacle
from utils.math_utils import distance

class MockState:
    def __init__(self, x=0.0, y=0.0, psi=0.0, v=0.0):
        self._values = {"x": x, "y": y, "psi": psi, "v": v}
        self._pos = np.array([x, y])

    def get(self, key):
        return self._values.get(key, 0.0)

    def get_position(self):
        return self._pos

# Mock CONFIG for testing
mock_config = {
    "max_obstacle_distance": 50.0,
    "max_obstacles": 3,
    "N": 5,
    "integrator_step": 0.1,
    "probabilistic": {"enable": False}
}

class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        # Set up global CONFIG mock
        global CONFIG
        CONFIG = mock_config

        self.patcher = patch('planning.src.data_prep.CONFIG', mock_config)
        self.mock_config = self.patcher.start()

        # Set up basic test objects
        self.state = MockState(10.0, 20.0, 0.0, 5.0)


    def tearDown(self):
        self.patcher.stop()

    def test_define_robot_area_single_disc(self):
        """Test defining robot area with a single disc."""
        length = 4.0
        width = 2.0
        n_discs = 1

        result = define_robot_area(length, width, n_discs)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].offset, 0.0)
        self.assertAlmostEqual(result[0].radius, 1.0)

    def test_define_robot_area_multiple_discs(self):
        """Test defining robot area with multiple discs."""
        length = 4.0
        width = 2.0
        n_discs = 3

        result = define_robot_area(length, width, n_discs)

        self.assertEqual(len(result), 3)
        # Check first disc (back of car)
        self.assertAlmostEqual(result[0].offset, -1.0)
        self.assertAlmostEqual(result[0].radius, 1.0)
        # Check last disc (front of car)
        self.assertAlmostEqual(result[2].offset, 1.0)
        self.assertAlmostEqual(result[2].radius, 1.0)
        # Check middle disc
        self.assertAlmostEqual(result[1].offset, 0.0)
        self.assertAlmostEqual(result[1].radius, 1.0)

    def test_define_robot_area_zero_discs(self):
        """Test that defining robot area with zero discs raises an assertion error."""
        length = 4.0
        width = 2.0
        n_discs = 0

        with self.assertRaises(AssertionError):
            define_robot_area(length, width, n_discs)

    def test_get_dummy_obstacle(self):
        """Test creating a dummy obstacle."""
        state = MockState(10.0, 20.0)

        result = get_dummy_obstacle(state)

        self.assertEqual(result.index, -1)
        self.assertTrue(np.array_equal(result.position, np.array([110.0, 120.0])))
        self.assertEqual(result.angle, 0.0)
        self.assertEqual(result.radius, 0.0)

    def test_get_constant_velocity_prediction_deterministic(self):
        """Test constant velocity prediction with deterministic setting."""
        position = np.array([10.0, 20.0])
        velocity = np.array([1.0, 2.0])
        dt = 0.1
        steps = 3

        # Ensure probabilistic is disabled
        CONFIG["probabilistic"]["enable"] = False

        result = get_constant_velocity_prediction(position, velocity, dt, steps)

        self.assertEqual(result.type, PredictionType.DETERMINISTIC)
        self.assertEqual(len(result.modes), 1)
        self.assertEqual(len(result.modes[0]), steps)

        # Check positions at each step
        np.testing.assert_allclose(result.modes[0][0].position, np.array([10.0, 20.0]))
        np.testing.assert_allclose(result.modes[0][1].position, np.array([10.1, 20.2]))
        np.testing.assert_allclose(result.modes[0][2].position, np.array([10.2, 20.4]))

        # Check noise parameters
        self.assertEqual(result.modes[0][0].major_radius, 0.0)
        self.assertEqual(result.modes[0][0].minor_radius, 0.0)

    def test_get_constant_velocity_prediction_probabilistic(self):
        """Test constant velocity prediction with probabilistic setting."""
        position = np.array([10.0, 20.0])
        velocity = np.array([1.0, 2.0])
        dt = 0.1
        steps = 3

        # Enable probabilistic
        CONFIG["probabilistic"]["enable"] = True

        result = get_constant_velocity_prediction(position, velocity, dt, steps)

        self.assertEqual(result.type, PredictionType.GAUSSIAN)
        self.assertEqual(len(result.modes), 1)
        self.assertEqual(len(result.modes[0]), steps)

        # Check positions at each step
        np.testing.assert_allclose(result.modes[0][0].position, np.array([10.0, 20.0]))
        np.testing.assert_allclose(result.modes[0][1].position, np.array([10.1, 20.2]))
        np.testing.assert_allclose(result.modes[0][2].position, np.array([10.2, 20.4]))

        # Check that uncertainty propagation has been applied
        self.assertGreater(result.modes[0][0].major_radius, 0.0)
        self.assertGreater(result.modes[0][0].minor_radius, 0.0)

        # Reset probabilistic to default
        CONFIG["probabilistic"]["enable"] = False

    def test_remove_distant_obstacles(self):
        """Test removing distant obstacles."""
        state = MockState(10.0, 10.0)

        # Create test obstacles
        obstacles = [
            DynamicObstacle(0, np.array([15.0, 15.0]), 0.0, 1.0),  # Close (distance ~7.07)
            DynamicObstacle(1, np.array([40.0, 40.0]), 0.0, 1.0),  # Far (distance ~42.43)
            DynamicObstacle(2, np.array([12.0, 12.0]), 0.0, 1.0),  # Close (distance ~2.83)
        ]

        CONFIG["max_obstacle_distance"] = 10.0

        remove_distant_obstacles(obstacles, state)

        # Should keep only the two close obstacles
        self.assertEqual(len(obstacles), 2)
        self.assertTrue(np.array_equal(obstacles[0].position, np.array([15.0, 15.0])))
        self.assertTrue(np.array_equal(obstacles[1].position, np.array([12.0, 12.0])))

    def test_ensure_obstacle_size_add_dummies(self):
        """Test ensuring obstacle size by adding dummies when there are too few."""
        state = MockState(10.0, 10.0)

        # Create fewer obstacles than max
        obstacles = [
            DynamicObstacle(0, np.array([15.0, 15.0]), 0.0, 1.0),
            DynamicObstacle(1, np.array([12.0, 12.0]), 0.0, 1.0),
        ]

        CONFIG["max_obstacles"] = 4
        CONFIG["N"] = 3

        ensure_obstacle_size(obstacles, state)

        # Should have 4 obstacles now (2 original + 2 dummies)
        self.assertEqual(len(obstacles), 4)

        # Check that prediction was added to dummies
        self.assertEqual(len(obstacles[2].prediction.modes), 1)
        self.assertEqual(len(obstacles[2].prediction.modes[0]), 3)  # N = 3
        self.assertTrue(np.array_equal(obstacles[2].position, np.array([110.0, 110.0])))

    def test_ensure_obstacle_size_remove_excess(self):
        """Test ensuring obstacle size by removing excess obstacles."""
        state = MockState(10.0, 10.0, psi=0.0, v=1.0)

        # Mock predictions for testing distance sorting
        def create_mock_prediction(positions):
            pred = Prediction(PredictionType.DETERMINISTIC)
            pred.modes = [[]]
            for pos in positions:
                pred.modes[0].append(PredictionStep(np.array(pos), 0.0, 0.0, 0.0))
            return pred

        # Create more obstacles than max with predictions
        obstacles = [
            DynamicObstacle(0, np.array([15.0, 15.0]), 0.0, 1.0),
            DynamicObstacle(1, np.array([40.0, 40.0]), 0.0, 1.0),  # Should be removed (farthest)
            DynamicObstacle(2, np.array([11.0, 11.0]), 0.0, 1.0),  # Should be kept (closest)
            DynamicObstacle(3, np.array([20.0, 20.0]), 0.0, 1.0),
        ]

        # Add mock predictions based on initial positions
        for i, obs in enumerate(obstacles):
            positions = [(obs.position[0] + j, obs.position[1] + j) for j in range(CONFIG["N"])]
            obs.prediction = create_mock_prediction(positions)

        CONFIG["max_obstacles"] = 3
        print("Obstacles now includes: " + str(obstacles))
        ensure_obstacle_size(obstacles, state)
        print("Obstacles now includes: " + str(obstacles))
        # Should have 3 obstacles now
        self.assertEqual(len(obstacles), 3)

        # Check that closest obstacles were kept
        self.assertEqual(obstacles[0].index, 0)  # IDs should be reassigned
        self.assertEqual(obstacles[1].index, 1)
        self.assertEqual(obstacles[2].index, 2)

        # The closest obstacle should be in the list
        found_closest = False
        for obs in obstacles:
            if np.array_equal(obs.position, np.array([11.0, 11.0])):
                found_closest = True
                break
        self.assertTrue(found_closest, "The closest obstacle should be kept")

        # The farthest obstacle should not be in the list
        found_farthest = False
        for obs in obstacles:
            if np.array_equal(obs.position, np.array([40.0, 40.0])):
                found_farthest = True
                break
        self.assertFalse(found_farthest, "The farthest obstacle should be removed")

    def test_propagate_prediction_uncertainty(self):
        """Test propagating uncertainty through prediction steps."""
        # Create a prediction with initial uncertainty
        prediction = Prediction(PredictionType.GAUSSIAN)
        prediction.modes = [[]]

        for i in range(3):
            prediction.modes[0].append(PredictionStep(
                np.array([10.0 + i, 20.0 + i]),
                0.0,
                0.3,  # Initial major radius
                0.3  # Initial minor radius
            ))

        CONFIG["N"] = 3
        CONFIG["integrator_step"] = 0.1

        # Save original values for comparison
        original_major = prediction.modes[0][0].major_radius
        original_minor = prediction.modes[0][0].minor_radius

        propagate_prediction_uncertainty(prediction)

        # Check that uncertainty grows with steps
        for i in range(3):
            self.assertGreater(prediction.modes[0][i].major_radius, original_major)
            self.assertGreater(prediction.modes[0][i].minor_radius, original_minor)

        # Check that uncertainty increases with each step
        self.assertLess(prediction.modes[0][0].major_radius, prediction.modes[0][1].major_radius)
        self.assertLess(prediction.modes[0][1].major_radius, prediction.modes[0][2].major_radius)

    def test_propagate_prediction_uncertainty_for_obstacles(self):
        """Test propagating uncertainty for all obstacles."""
        # Create obstacles with predictions
        obstacles = []
        for i in range(2):
            obs = DynamicObstacle(i, np.array([10.0 * i, 20.0 * i]), 0.0, 1.0)

            # Add prediction
            pred = Prediction(PredictionType.GAUSSIAN)
            pred.modes = [[]]
            for j in range(3):
                pred.modes[0].append(PredictionStep(
                    np.array([10.0 + j, 20.0 + j]),
                    0.0,
                    0.3,  # Initial major radius
                    0.3  # Initial minor radius
                ))
            obs.prediction = pred
            obstacles.append(obs)

        CONFIG["N"] = 3
        CONFIG["integrator_step"] = 0.1

        # Save original values for comparison
        original_major = obstacles[0].prediction.modes[0][0].major_radius

        propagate_prediction_uncertainty_for_obstacles(obstacles)

        # Check that uncertainty has propagated in all obstacles
        for obs in obstacles:
            for i in range(3):
                self.assertGreater(obs.prediction.modes[0][i].major_radius, original_major)

        # Check that uncertainty increases with each step for each obstacle
        for obs in obstacles:
            self.assertLess(obs.prediction.modes[0][0].major_radius,
                            obs.prediction.modes[0][2].major_radius)

    def test_distance_function(self):
        """Test the distance calculation function."""
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])

        result = distance(a, b)
        expected = 5.0  # sqrt((4-1)^2 + (6-2)^2) = sqrt(25) = 5

        self.assertAlmostEqual(result, expected)


# For running the tests directly
if __name__ == "__main__":
    unittest.main()