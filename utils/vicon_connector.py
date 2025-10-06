# state_provider.py
import time
import numpy as np


# This is a placeholder class. You must replace it with your actual Vicon client.
class ViconStateProvider:
	"""Provides the current state of the vehicle from the Vicon system."""

	def __init__(self):
		self._last_state = None
		self._last_time = None
		# You would initialize your Vicon connection here.
		print("Vicon Client Initialized (Placeholder).")

	def get_current_state(self):
		"""
		Fetches the latest data from Vicon and estimates velocity.
		:return: A dictionary {'x', 'y', 'psi', 'v'}
		"""
		# --- THIS IS MOCK DATA - REPLACE WITH REAL VICON SDK CALLS ---
		# For example: pose = vicon_client.get_pose("car_name")
		# x, y, psi = pose.x, pose.y, pose.yaw
		x, y, psi = self._get_mock_pose()
		current_time = time.time()

		# --- Estimate Velocity ---
		v = 0.0
		if self._last_state and self._last_time:
			dt = current_time - self._last_time
			if dt > 1e-6:  # Avoid division by zero
				dist = np.sqrt((x - self._last_state['x']) ** 2 + (y - self._last_state['y']) ** 2)
				v = dist / dt

		current_state = {'x': x, 'y': y, 'psi': psi, 'v': v}

		self._last_state = current_state
		self._last_time = current_time

		return current_state

	def _get_mock_pose(self):
		# Replace this with your actual data fetching
		t = time.time()
		# Simulate a simple circular path for demonstration
		x = 5 * np.cos(t * 0.1)
		y = 5 * np.sin(t * 0.1)
		psi = 0.1 * t + np.pi / 2
		return x, y, psi