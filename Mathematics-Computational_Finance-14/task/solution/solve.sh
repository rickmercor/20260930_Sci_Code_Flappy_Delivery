#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np  # noqa: E402, F811


def generate_brownian_increments(
    seed: int, n_steps: int, n_paths: int, dim_w: int, h: float
) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (("n_steps", n_steps), ("n_paths", n_paths), ("dim_w", dim_w)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(h, (int, float, np.integer, np.floating)) or not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    rng = np.random.default_rng(int(seed))
    return rng.normal(0.0, np.sqrt(float(h)), size=(int(n_steps), int(n_paths), int(dim_w)))

import numpy as np  # noqa: E402, F811


def simulate_gbm_paths(
    x0: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    r: float,
    h: float,
    increments: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    x0 = np.asarray(x0, dtype=float)
    q = np.asarray(q, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    increments = np.asarray(increments, dtype=float)
    if x0.ndim != 1 or x0.size == 0 or np.any(x0 <= 0) or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must be a nonempty positive finite vector")
    d = x0.size
    if q.shape != (d,) or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite vector matching x0")
    if sigma.ndim != 2 or sigma.shape[0] != d or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must have one row per asset and be finite")
    if increments.ndim != 3 or increments.shape[2] != sigma.shape[1] or increments.shape[0] == 0 or increments.shape[1] == 0:
        raise ValueError("increments must have shape (n_steps, n_paths, dim_w)")
    if not np.all(np.isfinite(increments)) or not np.isfinite(r) or not np.isfinite(h) or h <= 0:
        raise ValueError("r and increments must be finite and h must be positive")
    paths = np.empty((increments.shape[0] + 1, increments.shape[1], d), dtype=float)
    paths[0] = x0
    drift = (float(r) - q) * float(h)
    for i in range(increments.shape[0]):
        factor = 1.0 + drift + increments[i] @ sigma.T
        paths[i + 1] = paths[i] * factor
        if np.any(paths[i + 1] <= 0) or not np.all(np.isfinite(paths[i + 1])):
            raise ValueError("Euler update produced a nonpositive or nonfinite asset")
    return paths

import numpy as np  # noqa: E402, F811


def evaluate_value_heads(
    paths: np.ndarray,
    segment_starts: np.ndarray,
    x_scale: np.ndarray,
    feature_matrix: np.ndarray,
    feature_bias: np.ndarray,
    value_weights: np.ndarray,
    value_bias: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    paths = np.asarray(paths, dtype=float)
    segment_starts = np.asarray(segment_starts)
    x_scale = np.asarray(x_scale, dtype=float)
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    feature_bias = np.asarray(feature_bias, dtype=float)
    value_weights = np.asarray(value_weights, dtype=float)
    value_bias = np.asarray(value_bias, dtype=float)
    if paths.ndim != 3 or paths.shape[0] == 0 or paths.shape[1] == 0 or paths.shape[2] == 0 or np.any(paths <= 0):
        raise ValueError("paths must be a nonempty positive three-dimensional array")
    d = paths.shape[2]
    if segment_starts.ndim != 1 or segment_starts.size == 0 or not np.issubdtype(segment_starts.dtype, np.integer):
        raise ValueError("segment_starts must be a nonempty integer vector")
    if np.any(segment_starts < 0) or np.any(segment_starts >= paths.shape[0]) or np.any(np.diff(segment_starts) <= 0):
        raise ValueError("segment_starts must be increasing valid path indices")
    m = segment_starts.size
    if x_scale.shape != (d,) or np.any(x_scale <= 0):
        raise ValueError("x_scale must be positive and match the asset dimension")
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != d:
        raise ValueError("feature_matrix has incompatible shape")
    n_features = feature_matrix.shape[0]
    if feature_bias.shape != (n_features,) or value_weights.shape != (m, n_features) or value_bias.shape != (m,):
        raise ValueError("feature or value-head arrays have incompatible shapes")
    arrays = (paths, x_scale, feature_matrix, feature_bias, value_weights, value_bias)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all numeric inputs must be finite")
    values = np.empty((m, paths.shape[1]), dtype=float)
    for j, start in enumerate(segment_starts.astype(int)):
        features = np.tanh(np.log(paths[start] / x_scale) @ feature_matrix.T + feature_bias)
        values[j] = features @ value_weights[j] + value_bias[j]
    return values

import numpy as np  # noqa: E402, F811


def evaluate_control_heads(
    paths: np.ndarray,
    x_scale: np.ndarray,
    feature_matrix: np.ndarray,
    feature_bias: np.ndarray,
    control_weights: np.ndarray,
    control_bias: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    paths = np.asarray(paths, dtype=float)
    x_scale = np.asarray(x_scale, dtype=float)
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    feature_bias = np.asarray(feature_bias, dtype=float)
    control_weights = np.asarray(control_weights, dtype=float)
    control_bias = np.asarray(control_bias, dtype=float)
    if paths.ndim != 3 or min(paths.shape) <= 0 or np.any(paths <= 0):
        raise ValueError("paths must be a nonempty positive three-dimensional array")
    n_steps, d = paths.shape[0] - 1, paths.shape[2]
    if n_steps <= 0 or x_scale.shape != (d,) or np.any(x_scale <= 0):
        raise ValueError("paths need at least one step and x_scale must match them")
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != d:
        raise ValueError("feature_matrix has incompatible shape")
    n_features = feature_matrix.shape[0]
    if feature_bias.shape != (n_features,) or control_weights.ndim != 3:
        raise ValueError("feature and control arrays have incompatible ranks")
    if control_weights.shape[0] != n_steps or control_weights.shape[2] != n_features:
        raise ValueError("one compatible control head is required per time step")
    dim_w = control_weights.shape[1]
    if control_bias.shape != (n_steps, dim_w):
        raise ValueError("control_bias has incompatible shape")
    arrays = (paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all numeric inputs must be finite")
    controls = np.empty((n_steps, paths.shape[1], dim_w), dtype=float)
    for i in range(n_steps):
        features = np.tanh(np.log(paths[i] / x_scale) @ feature_matrix.T + feature_bias)
        controls[i] = features @ control_weights[i].T + control_bias[i]
    return controls

import numpy as np  # noqa: E402, F811


def propagate_bsde_segments(
    value_starts: np.ndarray,
    controls: np.ndarray,
    increments: np.ndarray,
    segment_steps: np.ndarray,
    r: float,
    h: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    value_starts = np.asarray(value_starts, dtype=float)
    controls = np.asarray(controls, dtype=float)
    increments = np.asarray(increments, dtype=float)
    segment_steps = np.asarray(segment_steps)
    theta = np.asarray(theta, dtype=float)
    if value_starts.ndim != 2 or min(value_starts.shape) <= 0:
        raise ValueError("value_starts must be a nonempty matrix")
    if controls.ndim != 3 or increments.shape != controls.shape or controls.shape[1] != value_starts.shape[1]:
        raise ValueError("controls and increments must match paths and each other")
    if segment_steps.shape != (value_starts.shape[0],) or not np.issubdtype(segment_steps.dtype, np.integer):
        raise ValueError("segment_steps must be one integer count per segment")
    if np.any(segment_steps <= 0) or int(np.sum(segment_steps)) != controls.shape[0]:
        raise ValueError("segment_steps must be positive and cover the full grid")
    if theta.shape != (controls.shape[2],):
        raise ValueError("theta must have one coefficient per Brownian dimension")
    if not np.isfinite(r) or not np.isfinite(h) or h <= 0:
        raise ValueError("r must be finite and h must be finite and positive")
    if not all(np.all(np.isfinite(array)) for array in (value_starts, controls, increments, theta)):
        raise ValueError("all arrays must be finite")
    terminals = np.empty_like(value_starts, dtype=float)
    offset = 0
    for j, count in enumerate(segment_steps.astype(int)):
        y = value_starts[j].copy()
        for i in range(offset, offset + count):
            drift = -(-float(r) * y + controls[i] @ theta)
            y = y + drift * float(h) + np.sum(controls[i] * increments[i], axis=1)
        terminals[j] = y
        offset += count
    return terminals

import numpy as np  # noqa: E402, F811


def construct_bermudan_targets(
    paths: np.ndarray,
    value_starts: np.ndarray,
    segment_steps: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    paths = np.asarray(paths, dtype=float)
    value_starts = np.asarray(value_starts, dtype=float)
    segment_steps = np.asarray(segment_steps)
    strikes = np.asarray(strikes, dtype=float)
    if paths.ndim != 3 or min(paths.shape) <= 0 or np.any(paths <= 0) or not np.all(np.isfinite(paths)):
        raise ValueError("paths must be a nonempty positive finite array")
    if value_starts.ndim != 2 or value_starts.shape[1] != paths.shape[1]:
        raise ValueError("value_starts must match the number of paths")
    m = value_starts.shape[0]
    if segment_steps.shape != (m,) or not np.issubdtype(segment_steps.dtype, np.integer):
        raise ValueError("segment_steps must be one integer count per segment")
    if np.any(segment_steps <= 0) or int(np.sum(segment_steps)) != paths.shape[0] - 1:
        raise ValueError("segment_steps must be positive and cover the path grid")
    if strikes.shape != (m,) or not np.all(np.isfinite(strikes)) or not np.all(np.isfinite(value_starts)):
        raise ValueError("strikes and value_starts must be finite and shape compatible")
    targets = np.empty_like(value_starts, dtype=float)
    for j, end in enumerate(np.cumsum(segment_steps.astype(int))):
        basket = np.exp(np.mean(np.log(paths[end]), axis=1))
        exercise = np.maximum(strikes[j] - basket, 0.0)
        targets[j] = np.maximum(value_starts[j + 1], exercise) if j < m - 1 else exercise
    return targets

import numpy as np  # noqa: E402, F811


def compute_joint_objective(terminals: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    terminals = np.asarray(terminals, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if terminals.ndim != 2 or min(terminals.shape) <= 0 or targets.shape != terminals.shape:
        raise ValueError("terminals and targets must be matching nonempty matrices")
    if not np.all(np.isfinite(terminals)) or not np.all(np.isfinite(targets)):
        raise ValueError("terminals and targets must be finite")
    segment_losses = np.mean((terminals - targets) ** 2, axis=1)
    return np.concatenate((segment_losses, np.array([np.sum(segment_losses)])))

import numpy as np  # noqa: E402, F811


def compute_error_certificate(loss_summary: np.ndarray, h: float) -> float:
    """Reference implementation."""
    loss_summary = np.asarray(loss_summary, dtype=float)
    if loss_summary.ndim != 1 or loss_summary.size < 2:
        raise ValueError("loss_summary must contain component losses and their sum")
    if not np.all(np.isfinite(loss_summary)) or np.any(loss_summary < 0):
        raise ValueError("loss_summary must be finite and nonnegative")
    if not np.isclose(loss_summary[-1], np.sum(loss_summary[:-1]), rtol=1e-12, atol=1e-12):
        raise ValueError("the final loss_summary entry must equal the component sum")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    return float(h + loss_summary[-1])

import numpy as np  # noqa: E402


def run_full_pipeline(seed: int = 260118634) -> float:
    """Reference implementation chaining every earlier step."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    h = 0.125
    segment_steps = np.array([2, 2, 2, 2, 2], dtype=int)
    x0 = np.array([48.0, 52.0])
    x_scale = np.array([50.0, 50.0])
    q = np.array([0.01, 0.015])
    sigma = np.array([[0.24, 0.05], [0.08, 0.20]])
    r = 0.06
    strikes = np.array([51.0, 50.0, 49.0, 48.0, 47.0])
    theta = np.array([0.45, -0.30])
    feature_matrix = np.array([[0.70, -0.30], [-0.40, 0.60], [0.25, 0.50], [-0.55, -0.20]])
    feature_bias = np.array([0.10, -0.15, 0.05, 0.20])
    value_weights = np.array([[1.20, -0.40, 0.70, 0.30], [0.90, -0.60, 0.50, -0.20], [0.60, -0.30, 0.40, 0.10], [0.50, -0.25, 0.35, 0.05], [0.40, -0.20, 0.30, 0.15]])
    value_bias = np.array([3.40, 3.00, 2.60, 2.20, 1.80])
    control_bias = np.array([[-1.80, -1.40], [-1.70, -1.35], [-1.60, -1.25], [-1.50, -1.15], [-1.40, -1.05], [-1.30, -0.95], [-1.20, -0.85], [-1.10, -0.75], [-1.00, -0.65], [-0.90, -0.55]])
    control_weights = np.array([[[0.30, -0.20, 0.15, 0.10], [-0.10, 0.25, 0.20, -0.15]], [[0.25, -0.15, 0.10, 0.12], [-0.08, 0.22, 0.18, -0.12]], [[0.22, -0.18, 0.14, 0.08], [-0.12, 0.20, 0.16, -0.10]], [[0.20, -0.14, 0.12, 0.06], [-0.10, 0.18, 0.14, -0.08]], [[0.18, -0.12, 0.10, 0.05], [-0.08, 0.16, 0.12, -0.06]], [[0.16, -0.10, 0.08, 0.04], [-0.06, 0.14, 0.10, -0.05]], [[0.14, -0.08, 0.06, 0.03], [-0.05, 0.12, 0.08, -0.04]], [[0.12, -0.06, 0.05, 0.02], [-0.04, 0.10, 0.07, -0.03]], [[0.10, -0.05, 0.04, 0.02], [-0.03, 0.09, 0.06, -0.02]], [[0.08, -0.04, 0.03, 0.01], [-0.02, 0.08, 0.05, -0.02]]])
    n_steps = int(np.sum(segment_steps))
    increments = generate_brownian_increments(seed, n_steps, 7, 2, h)  # noqa: F821
    paths = simulate_gbm_paths(x0, q, sigma, r, h, increments)  # noqa: F821
    starts = np.concatenate((np.array([0]), np.cumsum(segment_steps)[:-1]))
    value_starts = evaluate_value_heads(  # noqa: F821
        paths, starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias
    )
    controls = evaluate_control_heads(  # noqa: F821
        paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias
    )
    terminals = propagate_bsde_segments(  # noqa: F821
        value_starts, controls, increments, segment_steps, r, h, theta
    )
    targets = construct_bermudan_targets(  # noqa: F821
        paths, value_starts, segment_steps, strikes
    )
    loss_summary = compute_joint_objective(terminals, targets)  # noqa: F821
    return compute_error_certificate(loss_summary, h)  # noqa: F821
SCICODE_GOLD_EOF
