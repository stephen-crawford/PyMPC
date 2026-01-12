# Moving Obstacle Handling Improvements

## Current Issues

1. **Prediction Accuracy**: Obstacle predictions may not accurately reflect actual obstacle movement
2. **Uncertainty Growth**: Uncertainty may not grow appropriately over the prediction horizon
3. **Velocity Estimation**: Using average speed doesn't account for acceleration/deceleration
4. **Constraint Timing**: Constraints use predicted states but may not account for obstacle behavior changes
5. **Warmstart Projection**: Warmstart projection may not properly account for moving obstacles

## Recommended Improvements

### 1. Improve Obstacle Velocity Estimation

**Current**: Uses average speed from obstacle speeds
**Improvement**: Use actual obstacle velocity vectors and account for acceleration

```python
# In propagate_obstacles or obstacle_manager
# Instead of:
avg_speed = np.mean(obstacle_speeds)

# Use:
# - Actual velocity vectors from obstacle state
# - Account for acceleration if available
# - Use obstacle dynamics model to predict velocity changes
```

### 2. Increase Uncertainty Growth for Moving Obstacles

**Current**: Uncertainty grows linearly: `uncertainty_std = base_std + k * growth_rate`
**Improvement**: Use quadratic or exponential growth for moving obstacles

```python
# For moving obstacles, increase uncertainty growth
if obstacle_is_moving:
    # Quadratic growth: uncertainty grows faster for further predictions
    uncertainty_std = base_std + k * growth_rate + (k * k) * 0.01
else:
    # Linear growth for static/slow obstacles
    uncertainty_std = base_std + k * growth_rate
```

### 3. Account for Obstacle Acceleration in Predictions

**Current**: Predictions assume constant velocity
**Improvement**: Use obstacle dynamics model to predict acceleration/deceleration

```python
# In propagate_obstacles, when using dynamics model:
# Account for control inputs that change velocity
# Use actual obstacle control inputs if available
# Predict velocity changes based on obstacle behavior
```

### 4. Strengthen Constraints for Fast-Moving Obstacles

**Current**: Same constraint strength for all obstacles
**Improvement**: Increase safety margin for faster-moving obstacles

```python
# In GaussianConstraints.calculate_constraints:
# Calculate obstacle speed
obstacle_speed = np.linalg.norm(obstacle_velocity) if hasattr(obstacle, 'velocity') else 0.0

# Increase safety margin for faster obstacles
if obstacle_speed > 1.0:  # Moving faster than 1 m/s
    speed_factor = 1.0 + (obstacle_speed - 1.0) * 0.2  # 20% increase per m/s above 1 m/s
    safety_margin_factor = 1.5 * speed_factor  # Base 1.5, scaled by speed
else:
    safety_margin_factor = 1.5
```

### 5. Improve Prediction Regeneration Timing

**Current**: Predictions regenerated after obstacle position update
**Improvement**: Regenerate predictions immediately after obstacle state update, before constraint calculation

```python
# In integration_test_framework.py:
# Regenerate predictions immediately after obstacle state update
# Ensure predictions are based on most recent obstacle state
# Use obstacle's actual dynamics model for accurate predictions
```

### 6. Add Obstacle Behavior Prediction

**Current**: Obstacles move based on simple dynamics
**Improvement**: Predict obstacle behavior (turning, stopping, accelerating) based on context

```python
# In obstacle_manager or propagate_obstacles:
# Predict obstacle behavior based on:
# - Obstacle's current state
# - Obstacle's goal/behavior type
# - Obstacle's interaction with vehicle
# - Obstacle's interaction with environment
```

### 7. Improve Warmstart Projection for Moving Obstacles

**Current**: Warmstart projection uses current obstacle position
**Improvement**: Project warmstart considering obstacle movement over prediction horizon

```python
# In GaussianConstraints._project_warmstart_to_gaussian_safety:
# For each stage, use predicted obstacle position at that stage
# Not just current obstacle position
# This ensures warmstart accounts for obstacle movement
```

### 8. Add Adaptive Constraint Horizon

**Current**: Fixed constraint horizon (stages 0-8)
**Improvement**: Extend constraint horizon for fast-moving obstacles

```python
# In GaussianConstraints.calculate_constraints:
# Calculate obstacle speed
obstacle_speed = np.linalg.norm(obstacle_velocity) if hasattr(obstacle, 'velocity') else 0.0

# Extend constraint horizon for faster obstacles
if obstacle_speed > 2.0:  # Moving faster than 2 m/s
    max_stage_for_constraints = 10  # Extended horizon
else:
    max_stage_for_constraints = 8  # Standard horizon
```

## Implementation Priority

1. **High Priority**:
   - Improve velocity estimation (use actual velocity vectors)
   - Strengthen constraints for fast-moving obstacles
   - Improve warmstart projection for moving obstacles

2. **Medium Priority**:
   - Increase uncertainty growth for moving obstacles
   - Account for obstacle acceleration in predictions
   - Add adaptive constraint horizon

3. **Low Priority**:
   - Add obstacle behavior prediction
   - Improve prediction regeneration timing (already done, but can be optimized)

## Reference: C++ mpc_planner

The C++ reference code likely:
- Uses obstacle dynamics models for accurate predictions
- Accounts for obstacle acceleration/deceleration
- Uses adaptive constraint strength based on obstacle speed
- Regenerates predictions each MPC iteration with current obstacle state
- Uses obstacle velocity vectors, not just average speed
