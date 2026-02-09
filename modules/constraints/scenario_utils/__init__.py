"""
Scenario utilities package for SMPC constraints.

This package provides:
- scenario_module: SafeHorizonModule implementation
- math_utils: ScenarioConstraint and compute_sample_size
- sampler: ScenarioSampler and MonteCarloValidator
- optimal_transport_predictor: OT-based statistical learning of obstacle dynamics
"""

from modules.constraints.scenario_utils.math_utils import (
    ScenarioConstraint,
    compute_sample_size,
    compute_effective_epsilon,
)
from modules.constraints.scenario_utils.sampler import (
    ScenarioSampler,
    MonteCarloValidator,
    WeightType,
)
from modules.constraints.scenario_utils.scenario_module import SafeHorizonModule
from modules.constraints.scenario_utils.optimal_transport_predictor import (
    OptimalTransportPredictor,
    OTWeightType,
    EmpiricalDistribution,
    TrajectoryBuffer,
    TrajectoryObservation,
    ModeDistribution,
    wasserstein_distance,
    wasserstein_barycenter,
    sinkhorn_algorithm,
    create_ot_predictor_with_standard_modes,
)

__all__ = [
    'ScenarioConstraint',
    'compute_sample_size',
    'compute_effective_epsilon',
    'ScenarioSampler',
    'MonteCarloValidator',
    'WeightType',
    'SafeHorizonModule',
    # Optimal Transport Predictor
    'OptimalTransportPredictor',
    'OTWeightType',
    'EmpiricalDistribution',
    'TrajectoryBuffer',
    'TrajectoryObservation',
    'ModeDistribution',
    'wasserstein_distance',
    'wasserstein_barycenter',
    'sinkhorn_algorithm',
    'create_ot_predictor_with_standard_modes',
]
