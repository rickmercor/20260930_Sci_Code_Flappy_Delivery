"""
Evaluate the frozen value head at every segment start.

A fully forward compound-BSDE solver parameterizes the value at the beginning

of every segment as a function of the state there, including random intermediate

states. Each frozen value head is a linear readout of a shared tanh feature map

of normalized log prices.

Inputs

------

paths: Positive asset paths of shape (n_times, n_paths, d).

segment_starts: Increasing grid indices for the segment starts.

x_scale: Positive log-price scale of shape (d,).

feature_matrix, feature_bias: Shared tanh feature parameters.

value_weights, value_bias: Segment-specific value-head parameters.

Returns

-------

values: Float array of shape (n_segments, n_paths).

Returns
-------
np.ndarray of shape (m, n_paths), segment-start values as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_value_heads(
    paths: np.ndarray,
    segment_starts: np.ndarray,
    x_scale: np.ndarray,
    feature_matrix: np.ndarray,
    feature_bias: np.ndarray,
    value_weights: np.ndarray,
    value_bias: np.ndarray,
) -> np.ndarray:
    """Evaluate frozen segment-start value heads on their path states.

    Parameters
    ----------
    paths : np.ndarray
        Positive paths of shape ``(n_times, n_paths, d)``.
    segment_starts : np.ndarray
        Increasing segment-start indices of shape ``(m,)``.
    x_scale : np.ndarray
        Positive feature scale of shape ``(d,)``.
    feature_matrix : np.ndarray
        Feature coefficients of shape ``(n_features, d)``.
    feature_bias : np.ndarray
        Feature offsets of shape ``(n_features,)``.
    value_weights : np.ndarray
        Head weights of shape ``(m, n_features)``.
    value_bias : np.ndarray
        Head offsets of shape ``(m,)``.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes, paths or x_scale are not positive,
        segment starts are invalid, or any input is nonfinite.

    Returns
    -------
    values : np.ndarray
        Segment-start values of shape ``(m, n_paths)``.
    """
    return values  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_evaluate_value_heads(
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """paths = np.array([[[10.0, 20.0], [11.0, 19.0]], [[12.0, 18.0], [9.0, 22.0]], [[13.0, 17.0], [8.0, 23.0]]])
segment_starts = np.array([0, 2])
x_scale = np.array([10.0, 20.0])
feature_matrix = np.array([[0.5, -0.2], [0.3, 0.4]])
feature_bias = np.array([0.1, -0.1])
value_weights = np.array([[1.0, -0.5], [0.7, 0.2]])
value_bias = np.array([2.0, 1.5])
""",
            "call": "np.round(evaluate_value_heads(paths, segment_starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias), 12).tolist()",
            "gold_call": "np.round(_oracle_evaluate_value_heads(paths, segment_starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias), 12).tolist()",
        },
        {
            "setup": """paths = np.array([[[5.0]]])
segment_starts = np.array([0])
x_scale = np.array([5.0])
feature_matrix = np.array([[1.0]])
feature_bias = np.array([0.0])
value_weights = np.array([[2.0]])
value_bias = np.array([3.0])
""",
            "call": "np.round(evaluate_value_heads(paths, segment_starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias), 12).tolist()",
            "gold_call": "np.round(_oracle_evaluate_value_heads(paths, segment_starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias), 12).tolist()",
        },
        {
            "setup": """paths = np.ones((2, 1, 1))
segment_starts = np.array([1, 0])
x_scale = np.ones(1)
feature_matrix = np.ones((1, 1))
feature_bias = np.zeros(1)
value_weights = np.ones((2, 1))
value_bias = np.zeros(2)
def run_model():
    try:
        evaluate_value_heads(paths, segment_starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_evaluate_value_heads(paths, segment_starts, x_scale, feature_matrix, feature_bias, value_weights, value_bias)
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
