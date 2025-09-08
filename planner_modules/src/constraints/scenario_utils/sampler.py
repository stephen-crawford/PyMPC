import json
import threading
import math
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import casadi as cd

from planning.src.types import trajectory_sample, Partition
from utils.math_utils import RandomGenerator, distance, rotation_matrix_from_heading
from utils.utils import LOG_INFO, LOG_WARN, LOG_DEBUG, read_config_file, get_config_dotted

PARTITION_READ_THREADS = 2


class PartitionReader:
	"""Reads trajectory partitions from JSON files."""

	def __init__(self):
		self.n_samples_ = 0
		self.data_size_ = 0
		self.file_path_ = ""
		self.batches_ = None
		self.small = 0
		self.batch_o_: List[float] = []
		self.batch_x_: List[List[float]] = []
		self.batch_y_: List[List[float]] = []

		# Load configuration
		self.config = read_config_file()
		self.horizon = self.get_config_value("horizon", 10)
		self.sample_size = self.get_config_value("scenario_constraints.sample_size", 50)

	def get_config_value(self, key, default=None):
		"""Get configuration value with proper fallback"""
		try:
			if '.' in key:
				parts = key.split('.')
				value = self.config
				for part in parts:
					if isinstance(value, dict) and part in value:
						value = value[part]
					else:
						return default
				return value
			else:
				return self.config.get(key, default)
		except Exception:
			return default

	def init(self, index: int) -> bool:
		"""Initialize partition reader with given index"""
		self.batch_x_.clear()
		self.batch_y_.clear()
		self.batch_o_.clear()

		LOG_INFO(f"[INFO] Initializing JSON Trajectory Sampler [{index}]")
		self.file_path_ = str(Path("scenario_replay/partitions") / f"partition-{index}.json")

		return self.read_samples()

	def read_samples(self) -> bool:
		"""Read samples from JSON file"""
		try:
			with open(self.file_path_, "r") as f:
				j = json.load(f)
		except (OSError, json.JSONDecodeError) as e:
			LOG_WARN(f"[ERROR] Error reading partitions from {self.file_path_}: {e}")
			return False

		v2_x = [0.0] * self.horizon
		v2_y = [0.0] * self.horizon

		for s in range(len(j)):
			try:
				obs_val = round(float(j[str(s)]["Observable"]) * 10) / 10
				self.batch_o_.append(obs_val)

				for k in range(self.horizon):
					v2_x[k] = float(j[str(s)]["Trajectory X"][f"x{k}"])
					v2_y[k] = float(j[str(s)]["Trajectory Y"][f"y{k}"])

				self.batch_x_.append(v2_x.copy())
				self.batch_y_.append(v2_y.copy())

			except (KeyError, ValueError) as e:
				LOG_WARN(f"Error parsing sample {s}: {e}")
				continue

		self.n_samples_ = len(self.batch_x_)
		LOG_INFO(f"\tSamples: {self.n_samples_}")

		# Replicate samples if we don't have enough
		if self.n_samples_ < self.sample_size and self.n_samples_ > 0:
			start_size = self.n_samples_
			for i in range(start_size, self.sample_size):
				self.batch_x_.append(self.batch_x_[i % start_size])
				self.batch_y_.append(self.batch_y_[i % start_size])
				self.batch_o_.append(self.batch_o_[i % start_size])
			self.n_samples_ = self.sample_size

		return self.n_samples_ > 0

	def get_sample_batch_x(self):
		return self.batch_x_

	def get_sample_batch_y(self):
		return self.batch_y_

	def get_sample_batch_o(self):
		return self.batch_o_


class PartitionSampler:
	"""Manages multiple partition readers for scenario sampling"""

	def __init__(self):
		self.partitions_: List[PartitionReader] = []
		self.largest_sample_size_ = 0
		self.mutex_ = threading.Lock()
		self.thread_pool_ = ThreadPoolExecutor(max_workers=PARTITION_READ_THREADS)
		self.partition_future_ = None
		self.partition_read_id_ = 0
		self.online_partition_x_: List[List[float]] = []
		self.online_partition_y_: List[List[float]] = []
		self.online_partition_obs_: List[float] = []

		self.config = read_config_file()
		self.horizon = self.get_config_value("horizon", 10)
		self.sample_size = self.get_config_value("scenario_constraints.sample_size", 50)
		self.debug_output = self.get_config_value("scenario_constraints.debug_output", False)

	def get_config_value(self, key, default=None):
		"""Get configuration value with proper fallback"""
		try:
			if '.' in key:
				parts = key.split('.')
				value = self.config
				for part in parts:
					if isinstance(value, dict) and part in value:
						value = value[part]
					else:
						return default
				return value
			else:
				return self.config.get(key, default)
		except Exception:
			return default

	def update_partition_data(self, partition_size: int):
		"""Update partition data with new size"""
		LOG_DEBUG("[INFO] Updating partitions...")
		LOG_DEBUG(f"[INFO] Number of Partitions: {partition_size}")

		new_partitions = [PartitionReader() for _ in range(partition_size)]
		success = [False] * partition_size

		def init_partition(i):
			success[i] = new_partitions[i].init(i)

		# Parallel initialization
		with ThreadPoolExecutor(max_workers=PARTITION_READ_THREADS) as pool:
			pool.map(init_partition, range(partition_size))

		if not all(success):
			LOG_WARN("Failed to load some partitions")
			return

		if new_partitions:
			min_size = min(p.n_samples_ for p in new_partitions)
			max_size = max(p.n_samples_ for p in new_partitions)
			LOG_INFO(f"[INFO] Min partition size: {min_size}, max partition size: {max_size}")

		with self.mutex_:
			self.partitions_ = new_partitions
			self.largest_sample_size_ = self.sample_size

		LOG_INFO("[INFO] New partitions loaded!")

	def sample_partitioned_scenarios(self, obstacles, partitions: Dict[int, Partition],
									 partition_id: int, num_partitions: int, dt: float,
									 output_scenarios: trajectory_sample):
		"""Sample scenarios from partitioned database"""
		LOG_DEBUG("[INFO] Using partitioned database to generate scenarios")

		# Check if partition reading thread is ready
		thread_ready = True
		if self.partition_read_id_ > 0 and self.partition_future_:
			thread_ready = self.partition_future_.done()

		# Start new partition reading if needed
		if partition_id != self.partition_read_id_ and thread_ready:
			self.partition_read_id_ = partition_id
			self.partition_future_ = self.thread_pool_.submit(
				self.update_partition_data, num_partitions
			)

		with self.mutex_:
			if not self.partitions_:
				LOG_WARN("No partitions available for sampling")
				return

			for v, obstacle in enumerate(obstacles):
				angle = getattr(obstacle, 'angle', 0.0)
				partition = partitions.get(getattr(obstacle, 'index', 0), Partition(0, 0.0))

				if self.debug_output and getattr(obstacle, 'index', 0) not in partitions:
					LOG_WARN(f"[WARN] Sample Partition not found! (ID = {getattr(obstacle, 'index', 0)})")

				# Get partition data safely
				safe_id = min(partition.id, len(self.partitions_) - 1)
				batch_x = self.partitions_[safe_id].get_sample_batch_x()
				batch_y = self.partitions_[safe_id].get_sample_batch_y()
				batch_o = self.partitions_[safe_id].get_sample_batch_o()

				# Clear online partition data
				self.online_partition_x_.clear()
				self.online_partition_y_.clear()
				self.online_partition_obs_.clear()

				# Sample trajectories
				self.sample_trajectories(
					batch_x, batch_y, batch_o,
					self.online_partition_x_, self.online_partition_y_, self.online_partition_obs_,
					self.sample_size + self.sample_size % 2,
					float(getattr(partition, 'velocity', 1.0))
				)

				# Set initial positions
				initial_pos = getattr(obstacle, 'position', [0.0, 0.0])
				output_scenarios[0][v][0] = np.ones(self.sample_size) * initial_pos[0]
				output_scenarios[0][v][1] = np.ones(self.sample_size) * initial_pos[1]

				# Generate trajectory samples
				for k in range(self.horizon):
					prev_k = 0 if k == 0 else k - 1
					for s in range(self.sample_size):
						if s < len(self.online_partition_x_) and k < len(self.online_partition_x_[s]):
							output_scenarios[k][v][0][s] = (
									output_scenarios[prev_k][v][0][s] +
									(self.online_partition_x_[s][k] * math.cos(angle) -
									 self.online_partition_y_[s][k] * math.sin(angle)) * dt
							)
							output_scenarios[k][v][1][s] = (
									output_scenarios[prev_k][v][1][s] +
									(self.online_partition_x_[s][k] * math.sin(angle) +
									 self.online_partition_y_[s][k] * math.cos(angle)) * dt
							)

		LOG_DEBUG(f"[INFO] Sampler: Real scenarios ready, format: [k = {len(output_scenarios)} | "
				  f"v = {len(output_scenarios[0])} | s = {output_scenarios[0][0][0].shape[0]}]")

	def sample_trajectories(self, a, b, c, ac, bc, cc, samples: int, observable: float):
		"""Sample trajectories based on observable value"""
		if not c or len(c) < samples:
			# If not enough samples, replicate existing ones
			if a and b and c:
				ac.extend(a)
				bc.extend(b)
				cc.extend(c)
		else:
			# Find samples around the observable value
			idx = bisect_left(c, observable)
			start_avail = idx
			end_avail = len(c) - idx

			if end_avail >= samples // 2 and start_avail >= samples // 2:
				start_idx = idx - samples // 2
				end_idx = idx + samples // 2
			elif end_avail < samples // 2:
				extra_before = samples // 2 - end_avail
				start_idx = max(0, idx - (samples // 2 + extra_before))
				end_idx = len(c)
			else:
				extra_after = samples // 2 - start_avail
				start_idx = 0
				end_idx = min(len(c), idx + (samples // 2 + extra_after))

			ac.extend(a[start_idx:end_idx])
			bc.extend(b[start_idx:end_idx])
			cc.extend(c[start_idx:end_idx])


class ScenarioSampler:
	"""Main scenario sampler class"""

	def __init__(self):
		self.partition_sampler = PartitionSampler()
		self._samples_ready = False
		self.standard_samples_ready = False

		# Load configuration
		self.config = read_config_file()
		self.horizon = self.get_config_value("horizon", 10)
		self.timestep = self.get_config_value("timestep", 0.1)
		self.max_obstacles = self.get_config_value("max_obstacles", 5)
		self.batch_count = self.get_config_value("scenario_constraints.batch_count", 10)
		self.num_scenarios = self.get_config_value("scenario_constraints.num_scenarios", 100)
		self.sample_size = self.get_config_value("scenario_constraints.sample_size", 50)
		self.use_real_samples = self.get_config_value("scenario_constraints.use_real_samples", False)
		self.enable_safe_horizon = self.get_config_value("scenario_constraints.enable_safe_horizon", True)
		self.propagate_covariance = self.get_config_value("scenario_constraints.propagate_covariance", True)

		# Initialize sample storage
		self.samples = []
		self.standard_samples = np.zeros((self.batch_count, self.sample_size, 2))

		# Initialize 3D sample structure: [horizon][obstacles][x/y][sample_id]
		self._initialize_samples()

		# Load or generate standard samples
		if self.use_real_samples:
			self.partition_sampler.update_partition_data(1)

		load_successful = self.load()
		if not load_successful:
			LOG_WARN("Sample database does not exist. Generating new samples...")
			if self.get_config_value("scenario_constraints.sample_truncated", False):
				self.sample_truncated_standard_normal()
			else:
				self.sample_standard_normal()

		if not self.enable_safe_horizon:
			self.prune()

		self.resize_samples()
		self.standard_samples_ready = True
		self._samples_ready = False

		# Initialize probability and transformation matrices
		self._initialize_matrices()

	def _initialize_samples(self):
		"""Initialize the sample data structure"""
		self.samples = []
		for step in range(self.horizon):
			step_samples = []
			for obstacle_id in range(self.max_obstacles):
				obstacle_samples = [
					np.ones(self.sample_size) * 100.0,  # x coordinates
					np.ones(self.sample_size) * 100.5  # y coordinates
				]
				step_samples.append(obstacle_samples)
			self.samples.append(step_samples)

	def _initialize_matrices(self):
		"""Initialize transformation matrices for different modes"""
		if self.get_config_value("scenario_constraints.binomial_distribution", False):
			num_modes = self.horizon + 1
		else:
			num_modes = 1

		# Initialize probability sums
		self.sum_of_probabilities = []
		for obstacle_id in range(self.max_obstacles):
			self.sum_of_probabilities.append([None] * num_modes)

		# Initialize transformation matrices
		self.R = []
		self.SVD = []
		self.Sigma = []
		self.A_ = []

		for obstacle_id in range(self.max_obstacles):
			self.R.append([[None] * num_modes for _ in range(self.horizon)])
			self.SVD.append([[None] * num_modes for _ in range(self.horizon)])
			self.Sigma.append([[None] * num_modes for _ in range(self.horizon)])
			self.A_.append([[None] * num_modes for _ in range(self.horizon)])

	def get_config_value(self, key, default=None):
		"""Get configuration value with proper fallback"""
		try:
			if '.' in key:
				parts = key.split('.')
				value = self.config
				for part in parts:
					if isinstance(value, dict) and part in value:
						value = value[part]
					else:
						return default
				return value
			else:
				return self.config.get(key, default)
		except Exception:
			return default

	def sample_standard_normal(self):
		"""Generate standard normal samples"""
		u = np.random.rand(self.batch_count, self.sample_size, 2)
		# Convert uniform to Gaussian using Box-Muller transform
		u1, u2 = u[:, :, 0], u[:, :, 1]
		z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
		z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
		self.standard_samples = np.stack([z1, z2], axis=-1)

	def sample_truncated_standard_normal(self):
		"""Generate truncated standard normal samples"""
		truncated_radius = self.get_config_value("scenario_constraints.truncated_radius", 3.0)
		truncated_cap = math.exp(-math.pow(truncated_radius, 2) / 2)

		for batch_id in range(self.batch_count):
			for sample_id in range(self.sample_size):
				# Generate uniform samples
				u1 = np.random.random()
				u2 = np.random.random()

				# Scale and shift for truncation
				u1 = u1 * (1.0 - truncated_cap) + truncated_cap

				# Box-Muller transform
				z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
				z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

				self.standard_samples[batch_id][sample_id] = [z1, z2]

	def sort_samples(self):
		"""Sort samples by distance from origin"""
		norms = np.sum(self.standard_samples ** 2, axis=2)
		sort_idx = np.argsort(norms, axis=1)[:, ::-1]  # Descending order
		self.standard_samples = np.take_along_axis(
			self.standard_samples, sort_idx[..., None], axis=1
		)

	def prune(self):
		"""Prune samples to remove outliers"""
		self.sort_samples()
		max_size = 0

		for batch_id in range(self.batch_count):
			distances = np.zeros(self.sample_size)
			used_indices = np.zeros(self.sample_size, dtype=bool)
			max_ellipse_radius = self.fit_maximum_ellipse(self.standard_samples[batch_id])

			# Check points on ellipse boundary
			for point_id in range(50):
				angle = 2.0 * math.pi / 50 * point_id
				point = [
					max_ellipse_radius[0] * math.cos(angle),
					max_ellipse_radius[1] * math.sin(angle)
				]

				# Calculate distances to all samples
				for sample_id in range(self.sample_size):
					sample_point = self.standard_samples[batch_id][sample_id]
					distances[sample_id] = np.linalg.norm(np.array(point) - np.array(sample_point))

				# Sort by distance and mark closest samples as used
				sorted_indices = np.argsort(distances)
				constraints_to_check = (
						self.get_config_value("scenario_constraints.polygon_checked_constraints", 10) +
						self.get_config_value("scenario_constraints.removal_count", 5)
				)

				for i in range(min(constraints_to_check, len(sorted_indices))):
					used_indices[sorted_indices[i]] = True

			# Find the last used index
			last_used = 0
			for sample_id in range(self.sample_size - 1, -1, -1):
				if used_indices[sample_id]:
					last_used = sample_id + 1
					break

			max_size = max(max_size, last_used)

		# Prune all batches to max_size
		pruned_count = self.sample_size - max_size
		LOG_WARN(f"Gaussian samples pruned: {pruned_count} / {self.sample_size} "
				 f"({pruned_count / self.sample_size * 100:.1f}%)")

		for batch_id in range(self.batch_count):
			self.standard_samples[batch_id] = self.standard_samples[batch_id][:max_size]
			np.random.shuffle(self.standard_samples[batch_id])

		self.sample_size = max_size

	def fit_maximum_ellipse(self, samples):
		"""Fit maximum ellipse to samples"""
		max_x = max_y = 0.0
		for sample in samples:
			abs_x = abs(sample[0])
			abs_y = abs(sample[1])
			if abs_x > max_x:
				max_x = abs_x
			if abs_y > max_y:
				max_y = abs_y
		return max_x, max_y

	def used_indices(self, size, default_value):
		"""Generate used indices array"""
		return np.full(size, default_value, dtype=bool)

	def integrate_and_translate_to_mean_and_variance(self, obstacles, timestep):
		"""Main sampling function that integrates obstacle predictions"""
		if not self.standard_samples_ready:
			LOG_WARN("Must call standard sampling before obstacle sampling")
			return self.samples

		if len(obstacles) > self.max_obstacles:
			LOG_WARN(f"Received {len(obstacles)} obstacles, but max is {self.max_obstacles}")
			obstacles = obstacles[:self.max_obstacles]

		random_generator = RandomGenerator()

		# Generate batch picks for each time step
		batch_pick = []
		for step in range(self.horizon):
			batch_pick.append(random_generator.Int(self.batch_count - 1))

		# Generate shuffle indices
		shuffle_indices = []
		for step in range(self.horizon):
			indices = np.arange(self.sample_size)
			np.random.shuffle(indices)
			shuffle_indices.append(indices)

		# Process each obstacle
		for obstacle_id, obstacle in enumerate(obstacles):
			if not hasattr(obstacle, 'prediction') or obstacle.prediction is None:
				LOG_WARN(f"Obstacle {obstacle_id} has no prediction")
				continue

			prediction = obstacle.prediction

			# Handle different prediction types
			if hasattr(prediction, 'modes') and prediction.modes:
				self._process_multimodal_prediction(
					obstacle_id, prediction, batch_pick, shuffle_indices, timestep
				)
			else:
				# Single mode prediction
				self._process_single_mode_prediction(
					obstacle_id, prediction, batch_pick, shuffle_indices, timestep
				)

		self._samples_ready = True
		return self.samples

	def _process_multimodal_prediction(self, obstacle_id, prediction, batch_pick, shuffle_indices, timestep):
		"""Process multimodal obstacle prediction"""
		# Calculate cumulative probabilities
		modes = getattr(prediction, 'modes', [])
		probabilities = getattr(prediction, 'probabilities', [])

		if not modes:
			return

		cumulative_probs = []
		cumulative_sum = 0
		for prob in probabilities:
			cumulative_sum += prob
			cumulative_probs.append(cumulative_sum)

		# Process each sample
		for sample_id in range(self.sample_size):
			# Select mode based on probability
			mode_random = np.random.random()
			selected_mode = 0
			for mode_idx, cum_prob in enumerate(cumulative_probs):
				if mode_random <= cum_prob:
					selected_mode = mode_idx
					break

			if selected_mode >= len(modes):
				selected_mode = len(modes) - 1

			mode = modes[selected_mode]
			self._generate_sample_trajectory(
				obstacle_id, mode, sample_id, batch_pick, shuffle_indices, timestep
			)

	def _process_single_mode_prediction(self, obstacle_id, prediction, batch_pick, shuffle_indices, timestep):
		"""Process single mode obstacle prediction"""
		for sample_id in range(self.sample_size):
			self._generate_sample_trajectory(
				obstacle_id, prediction, sample_id, batch_pick, shuffle_indices, timestep
			)

	def _generate_sample_trajectory(self, obstacle_id, mode_prediction, sample_id,
									batch_pick, shuffle_indices, timestep):
		"""Generate a single sample trajectory"""
		bivariate_gaussian = np.zeros(2)

		for step in range(self.horizon):
			# Get prediction for this time step
			if hasattr(mode_prediction, '__getitem__'):
				step_prediction = mode_prediction[step] if step < len(mode_prediction) else mode_prediction[-1]
			else:
				step_prediction = mode_prediction

			# Get position and uncertainty
			position = getattr(step_prediction, 'position', [0.0, 0.0])
			major_radius = getattr(step_prediction, 'major_radius', 0.1)
			minor_radius = getattr(step_prediction, 'minor_radius', 0.1)
			angle = getattr(step_prediction, 'angle', 0.0)

			# Create rotation matrix
			R = rotation_matrix_from_heading(-angle)

			# Create covariance matrix
			if self.propagate_covariance:
				sigma_major = major_radius * timestep
				sigma_minor = minor_radius * timestep
			else:
				sigma_major = major_radius
				sigma_minor = minor_radius

			SVD = np.array([[sigma_major ** 2, 0], [0, sigma_minor ** 2]])
			Sigma = R @ SVD @ R.T

			try:
				# Cholesky decomposition
				A = np.linalg.cholesky(Sigma)
			except np.linalg.LinAlgError:
				# Fallback to square root if Cholesky fails
				eigenvals, eigenvecs = np.linalg.eigh(Sigma)
				eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
				A = eigenvecs @ np.diag(np.sqrt(eigenvals))

			# Select random sample
			batch_idx = batch_pick[step]
			sample_idx = shuffle_indices[step][sample_id % len(shuffle_indices[step])]

			if (batch_idx < len(self.standard_samples) and
					sample_idx < len(self.standard_samples[batch_idx])):
				standard_sample = self.standard_samples[batch_idx][sample_idx]

				# Transform standard sample
				transformed_sample = A @ standard_sample
				bivariate_gaussian += transformed_sample

				# Set sample position
				self.samples[step][obstacle_id][0][sample_id] = position[0] + bivariate_gaussian[0]
				self.samples[step][obstacle_id][1][sample_id] = position[1] + bivariate_gaussian[1]

	def sample_partitioned_scenarios(self, obstacles, partitions, partition_id, num_partitions, timestep, samples):
		"""Sample scenarios using partitioned database"""
		self.partition_sampler.sample_partitioned_scenarios(
			obstacles, partitions, partition_id, num_partitions, timestep, samples
		)
		self._samples_ready = True

	def load(self):
		"""Load samples from database (placeholder)"""
		# This would load pre-computed samples from a database
		# For now, return False to generate new samples
		return False

	def resize_samples(self):
		"""Resize sample arrays to current configuration"""
		self._initialize_samples()

	def samples_ready(self):
		"""Check if samples are ready for use"""
		return self._samples_ready

	def to_casadi(self):
		"""Convert arrays to CasADi format for optimization"""
		try:
			self.standard_samples = cd.DM(self.standard_samples)
			# Convert other matrices as needed
			for obstacle_id in range(self.max_obstacles):
				for step in range(self.horizon):
					if self.R[obstacle_id][step][0] is not None:
						self.R[obstacle_id][step][0] = cd.DM(self.R[obstacle_id][step][0])
					if self.SVD[obstacle_id][step][0] is not None:
						self.SVD[obstacle_id][step][0] = cd.DM(self.SVD[obstacle_id][step][0])
					if self.Sigma[obstacle_id][step][0] is not None:
						self.Sigma[obstacle_id][step][0] = cd.DM(self.Sigma[obstacle_id][step][0])
					if self.A_[obstacle_id][step][0] is not None:
						self.A_[obstacle_id][step][0] = cd.DM(self.A_[obstacle_id][step][0])
		except Exception as e:
			LOG_WARN(f"Error converting to CasADi format: {e}")


def get_config_value(key, default=None):
	"""Global function to get configuration values"""
	try:
		config = read_config_file()
		if '.' in key:
			parts = key.split('.')
			value = config
			for part in parts:
				if isinstance(value, dict) and part in value:
					value = value[part]
				else:
					return default
			return value
		else:
			return config.get(key, default)
	except Exception:
		return default