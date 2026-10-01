"""
Evaluate the frozen martingale-control heads along the path grid.

The martingale control is parameterized separately at every grid point in a

deep BSDE discretization. Each frozen head consumes the shared normalized

log-price features and returns one coefficient per Brownian dimension.

Preserving the distinct time, path, Brownian, and feature axes is essential

when evaluating all heads over the path grid.

Inputs

------

paths: Positive asset paths of shape (n_steps + 1, n_paths, d).

x_scale: Positive log-price scale of shape (d,).

feature_matrix, feature_bias: Shared tanh feature parameters.

control_weights, control_bias: Grid-specific control-head parameters.

Returns

-------

controls: Float array of shape (n_steps, n_paths, dim_w).

Returns
-------
np.ndarray of shape (n_steps, n_paths, dim_w), frozen controls as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_control_heads(
    paths: np.ndarray,
    x_scale: np.ndarray,
    feature_matrix: np.ndarray,
    feature_bias: np.ndarray,
    control_weights: np.ndarray,
    control_bias: np.ndarray,
) -> np.ndarray:
    """Evaluate frozen grid-point control heads.

    Parameters
    ----------
    paths : np.ndarray
        Positive states of shape ``(n_steps + 1, n_paths, d)``.
    x_scale : np.ndarray
        Positive feature scale of shape ``(d,)``.
    feature_matrix : np.ndarray
        Feature coefficients of shape ``(n_features, d)``.
    feature_bias : np.ndarray
        Feature offsets of shape ``(n_features,)``.
    control_weights : np.ndarray
        Head weights of shape ``(n_steps, dim_w, n_features)``.
    control_bias : np.ndarray
        Head offsets of shape ``(n_steps, dim_w)``.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes, states or scales are not positive,
        the number of heads differs from the number of steps, or inputs are nonfinite.

    Returns
    -------
    controls : np.ndarray
        Control values of shape ``(n_steps, n_paths, dim_w)``.
    """
    return controls  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_evaluate_control_heads(
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """paths = np.array([[[10.0, 20.0], [11.0, 19.0]], [[12.0, 18.0], [9.0, 22.0]], [[13.0, 17.0], [8.0, 23.0]]])
x_scale = np.array([10.0, 20.0])
feature_matrix = np.array([[0.5, -0.2], [0.3, 0.4]])
feature_bias = np.array([0.1, -0.1])
control_weights = np.array([[[0.2, -0.1], [0.3, 0.4]], [[-0.2, 0.5], [0.1, -0.3]]])
control_bias = np.array([[-1.0, -0.8], [-0.7, -0.6]])
""",
            "call": "np.round(evaluate_control_heads(paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias), 12).tolist()",
            "gold_call": "np.round(_oracle_evaluate_control_heads(paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias), 12).tolist()",
        },
        {
            "setup": """paths = np.array([[[5.0]], [[5.0]]])
x_scale = np.array([5.0])
feature_matrix = np.array([[1.0]])
feature_bias = np.array([0.0])
control_weights = np.array([[[2.0]]])
control_bias = np.array([[3.0]])
""",
            "call": "np.round(evaluate_control_heads(paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias), 12).tolist()",
            "gold_call": "np.round(_oracle_evaluate_control_heads(paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias), 12).tolist()",
        },
        {
            "setup": """paths = np.ones((3, 1, 1))
x_scale = np.ones(1)
feature_matrix = np.ones((1, 1))
feature_bias = np.zeros(1)
control_weights = np.ones((1, 1, 1))
control_bias = np.ones((1, 1))
def run_model():
    try:
        evaluate_control_heads(paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_evaluate_control_heads(paths, x_scale, feature_matrix, feature_bias, control_weights, control_bias)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
