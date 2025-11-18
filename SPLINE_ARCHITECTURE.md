# Spline Architecture Documentation

This document describes how splines are used in PyMPC, matching the reference codebases:
- https://github.com/ttk592/spline (numeric cubic splines)
- https://github.com/tud-amr/mpc_planner (symbolic Spline2D for optimization)
- https://github.com/oscardegroot/ros_tools

## Architecture Overview

The codebase uses **two distinct spline types** based on the computation context:

### 1. Numeric Splines (TKSpline)
**Purpose**: Post-optimization evaluation, visualization, warmstart, numeric computations

**Implementation**: `TKSpline` class in `utils/math_tools.py`
- Based on ttk592/spline library
- Natural cubic splines with C2 continuity
- Pure numeric evaluation (numpy arrays)

**Usage**:
```python
# Stored in ReferencePath
path.x_spline = TKSpline(s_vals, x_vals)
path.y_spline = TKSpline(s_vals, y_vals)

# Evaluation
x = path.x_spline(s)  # or path.x_spline.at(s)
dx = path.x_spline.derivative()(s)  # or path.x_spline.deriv(s)
```

**Where Used**:
- `ReferencePath.x_spline`, `ReferencePath.y_spline` (always numeric)
- Post-optimization path evaluation
- Visualization and plotting
- Warmstart initialization
- Obstacle manager path queries
- Test framework numeric evaluations

### 2. Symbolic Splines (Spline2D and Spline3D)
**Purpose**: CasADi symbolic optimization

**Implementation**: 
- `Spline2D` class in `utils/math_tools.py` - for 2D paths (x, y)
- `Spline3D` class in `utils/math_tools.py` - for 3D paths (x, y, z)
- Based on tud-amr/mpc_planner implementation, extended to 3D
- Uses parameter dictionaries with spline coefficients
- Works with CasADi symbolic variables (MX/SX)

**Usage (2D)**:
```python
# Requires parameter dictionary with coefficients
# path_x_{i}_a, path_x_{i}_b, path_x_{i}_c, path_x_{i}_d
# path_y_{i}_a, path_y_{i}_b, path_y_{i}_c, path_y_{i}_d
# path_{i}_start for each segment i

spline2d = Spline2D(params, num_segments, s)  # s is CasADi MX/SX
x, y = spline2d.at(s)  # Returns symbolic expressions
dx, dy = spline2d.deriv(s)  # Returns symbolic derivatives
```

**Usage (3D)**:
```python
# Requires parameter dictionary with coefficients including z
# path_x_{i}_a, path_x_{i}_b, path_x_{i}_c, path_x_{i}_d
# path_y_{i}_a, path_y_{i}_b, path_y_{i}_c, path_y_{i}_d
# path_z_{i}_a, path_z_{i}_b, path_z_{i}_c, path_z_{i}_d
# path_{i}_start for each segment i

spline3d = Spline3D(params, num_segments, s)  # s is CasADi MX/SX
x, y, z = spline3d.at(s)  # Returns symbolic expressions
dx, dy, dz = spline3d.deriv(s)  # Returns symbolic derivatives
```

**Where Used**:
- `ContouringObjective.get_value()` - uses Spline2D or Spline3D based on `three_dimensional_contouring` flag
- `ContouringConstraints._compute_symbolic_constraints()` - uses Spline2D or Spline3D based on configuration
- Dynamics models updating spline state symbolically

## Automatic Detection

The code automatically detects whether to use numeric or symbolic splines:

### In ContouringObjective:
```python
# Check if s is symbolic
if isinstance(s, (cd.MX, cd.SX)):
    # Symbolic: Use Spline2D with params OR CasADi interpolants
    if has_path_params:
        path = Spline2D(params, num_segments, s)  # Preferred
    else:
        # Fallback: CasADi interpolants sampled from TKSpline
        x_interp = cd.interpolant('x_interp', 'linear', [s_vals], x_vals)
else:
    # Numeric: Use TKSpline directly
    path_x = reference_path.x_spline.at(float(cur_s))
```

### In ContouringConstraints:
```python
# Check if spline_val is symbolic
is_symbolic = isinstance(spline_val, (cd.MX, cd.SX))
if is_symbolic:
    # Use Spline2D with params OR CasADi interpolants
    return self._compute_symbolic_constraints(spline_val, state, data, stage_idx)
else:
    # Numeric fallback (should not happen in normal operation)
    return self._compute_numeric_constraints(cur_s, state, data, stage_idx)
```

## Key Design Decisions

1. **ReferencePath always uses TKSpline**: All `ReferencePath` objects store `TKSpline` instances for numeric evaluation. This ensures consistency for post-optimization use.

2. **Symbolic optimization uses Spline2D**: When path parameters are available, `Spline2D` with parameter dictionaries is used for symbolic computation, matching the C++ reference implementation.

3. **Fallback to CasADi interpolants**: When path parameters aren't available but symbolic computation is needed, the code samples from TKSpline and creates CasADi interpolants.

4. **Compatibility interface**: TKSpline supports scipy `CubicSpline` interface (`__call__` and `derivative()`) to allow existing code to work without changes.

## File Locations

- **Spline implementations**: `utils/math_tools.py`
  - `TKSpline`: Numeric cubic splines (1D, used for x, y, z separately)
  - `Spline2D`: Symbolic 2D splines for optimization (x, y)
  - `Spline3D`: Symbolic 3D splines for optimization (x, y, z)
  - `Spline`: Internal class used by Spline2D and Spline3D
  - `SplineSegment`: Internal class for parameter-based segments
  - `create_numeric_spline2d()`: Helper for 2D numeric splines
  - `create_numeric_spline3d()`: Helper for 3D numeric splines

- **Usage in objectives**: `modules/objectives/contouring_objective.py`
- **Usage in constraints**: `modules/constraints/contouring_constraints.py`
- **Reference path storage**: `planning/types.py` (`ReferencePath` class)

## Migration Notes

All existing code using `reference_path.x_spline(s)` and `reference_path.x_spline.derivative()(s)` continues to work due to TKSpline's compatibility interface. The code automatically uses the correct spline type based on context (symbolic vs numeric).

