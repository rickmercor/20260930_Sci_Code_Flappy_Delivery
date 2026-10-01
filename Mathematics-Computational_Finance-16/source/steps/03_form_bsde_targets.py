"""
A right-endpoint discretization of the deterministic BSDE integral defines the regression target that is later fit against the left-endpoint value-plus-gradient operator at each time step. y_{n+1} = V_{n+1}(x_{n+1}) - h(u, grad V_{n+1}) dt. For affine drift f, h is tr(F) + 0.5 ||s * grad V||^2 + u dot (s * grad V), where s contains the diagonal diffusion entries and u is evaluated at the advanced state.

Inputs
------
continuation: Array of value and gradient data with shape (K, d + 1).
next_states: Advanced states with shape (K, d).
policy_matrix: Affine policy matrix with shape (d, d).
policy_bias: Affine policy bias with shape (d,).
drift_trace: Divergence of the affine drift.
sigma_diag: Positive diffusion diagonal with shape (d,).
dt: Positive time increment.

Returns
-------
targets: Float array of shape (K,).

Returns
-------
np.ndarray of shape (K,), the right-endpoint BSDE targets as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def form_bsde_targets(
    continuation: np.ndarray,
    next_states: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    drift_trace: float,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Form right-endpoint nonlinear BSDE regression targets.

    Parameters
    ----------
    continuation : np.ndarray
        Value in column zero and gradient in the remaining columns.
    next_states : np.ndarray
        Advanced states with shape (K, d).
    policy_matrix : np.ndarray
        Matrix of the affine policy evaluated at `next_states`.
    policy_bias : np.ndarray
        Bias of the affine policy.
    drift_trace : float
        Divergence of the affine drift.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    dt : float
        Positive time increment.

    Raises
    ------
    ValueError
        If shapes are incompatible, an input is non-finite, `sigma_diag` is not
        positive, or `dt` is not finite and positive.

    Returns
    -------
    targets : np.ndarray
        BSDE targets with shape (K,).
    """
    return targets  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_form_bsde_targets(
    continuation: np.ndarray,
    next_states: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    drift_trace: float,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reference implementation."""
    data = np.asarray(continuation, dtype=float)
    points = np.asarray(next_states, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("next_states must be a non-empty two-dimensional array")
    sample_count, dimension = points.shape
    if data.shape != (sample_count, dimension + 1):
        raise ValueError("continuation must have shape (K, d + 1)")
    if np.asarray(policy_matrix).shape != (dimension, dimension):
        raise ValueError("policy_matrix must have shape (d, d)")
    if np.asarray(policy_bias).shape != (dimension,):
        raise ValueError("policy_bias must have shape (d,)")
    if np.asarray(sigma_diag).shape != (dimension,):
        raise ValueError("sigma_diag must have shape (d,)")
    arrays = (
        data,
        points,
        np.asarray(policy_matrix, dtype=float),
        np.asarray(policy_bias, dtype=float),
        np.asarray(sigma_diag, dtype=float),
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all array inputs must be finite")
    if not np.isfinite(drift_trace):
        raise ValueError("drift_trace must be finite")
    if np.any(arrays[4] <= 0.0):
        raise ValueError("sigma_diag must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    value = data[:, 0]
    gradient = data[:, 1:]
    scaled_gradient = gradient * arrays[4]
    policy = points @ arrays[2].T + arrays[3]
    nonlinear_term = (
        float(drift_trace)
        + 0.5 * np.sum(scaled_gradient * scaled_gradient, axis=1)
        + np.sum(policy * scaled_gradient, axis=1)
    )
    return value - nonlinear_term * dt

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
data = np.array([[0.8, 0.2, -0.1, 0.3], [0.4, -0.2, 0.5, 0.1]])
points = np.array([[0.1, -0.3, 0.5], [-0.2, 0.4, 0.7]])
U = np.array([[-0.25, 0.08, -0.04], [0.05, -0.18, 0.06], [-0.03, 0.04, -0.12]])
c = np.array([0.02, -0.01, 0.015])
sigma = np.array([0.35, 0.22, 0.18])
""",
            "call": "np.round(form_bsde_targets(data, points, U, c, -0.16, sigma, 0.15), 12).tolist()",
            "gold_call": "np.round(_oracle_form_bsde_targets(data, points, U, c, -0.16, sigma, 0.15), 12).tolist()",
        },
        {
            "setup": """import numpy as np
data = np.array([[2.0, 0.0]])
points = np.zeros((1, 1))
U = np.zeros((1, 1))
c = np.zeros(1)
sigma = np.ones(1)
""",
            "call": "form_bsde_targets(data, points, U, c, 0.0, sigma, 0.1).tolist()",
            "gold_call": "_oracle_form_bsde_targets(data, points, U, c, 0.0, sigma, 0.1).tolist()",
        },
        {
            "setup": """import numpy as np
data = np.zeros((2, 3))
points = np.zeros((2, 2))
U = np.zeros((2, 2))
c = np.zeros(2)
sigma = np.ones(2)
def run_model():
    try:
        form_bsde_targets(data, points, U, c, 0.0, sigma, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_form_bsde_targets(data, points, U, c, 0.0, sigma, 0.0)
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
