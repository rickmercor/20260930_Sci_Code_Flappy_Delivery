"""
A controlled factor state follows an Euler-Maruyama step of the reverse diffusion. The same noise increment later enters the derivative term of the BSDE regression, coupling the discrete state trajectory to the local regression problem solved at each time slice. For row k, the affine drift is f(x_k) = F x_k + b, the feedback is u(x_k) = U x_k + c, and diagonal diffusion s gives x'_k = x_k + [f(x_k) + s * u(x_k)] dt + s * xi_k sqrt(dt).

Inputs
------
states: Float array of shape (K, d).
noises: Float array of shape (K, d).
drift_matrix: Float array of shape (d, d).
drift_bias: Float array of shape (d,).
sigma_diag: Positive float array of shape (d,).
policy_matrix: Float array of shape (d, d).
policy_bias: Float array of shape (d,).
dt: Positive time increment.

Returns
-------
next_states: Float array of shape (K, d).

Returns
-------
np.ndarray of shape (K, d), the advanced controlled states as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_controlled_diffusion(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance every controlled state by one Euler-Maruyama interval.

    Parameters
    ----------
    states : np.ndarray
        Current states with shape (K, d).
    noises : np.ndarray
        Standard-normal innovations with the same shape as `states`.
    drift_matrix : np.ndarray
        Matrix F of the affine drift.
    drift_bias : np.ndarray
        Bias b of the affine drift.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    policy_matrix : np.ndarray
        Matrix U of the affine feedback policy.
    policy_bias : np.ndarray
        Bias c of the affine feedback policy.
    dt : float
        Positive time increment.

    Raises
    ------
    ValueError
        If an input has an incompatible shape, contains a non-finite value, if
        `sigma_diag` is not positive, or if `dt` is not finite and positive.

    Returns
    -------
    next_states : np.ndarray
        Advanced states with shape (K, d).
    """
    return next_states  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_advance_controlled_diffusion(
    states: np.ndarray,
    noises: np.ndarray,
    drift_matrix: np.ndarray,
    drift_bias: np.ndarray,
    sigma_diag: np.ndarray,
    policy_matrix: np.ndarray,
    policy_bias: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reference implementation."""
    x = np.asarray(states, dtype=float)
    xi = np.asarray(noises, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("states must be a non-empty two-dimensional array")
    if xi.shape != x.shape:
        raise ValueError("noises must have the same shape as states")
    sample_count, dimension = x.shape
    if np.asarray(drift_matrix).shape != (dimension, dimension):
        raise ValueError("drift_matrix must have shape (d, d)")
    if np.asarray(drift_bias).shape != (dimension,):
        raise ValueError("drift_bias must have shape (d,)")
    if np.asarray(sigma_diag).shape != (dimension,):
        raise ValueError("sigma_diag must have shape (d,)")
    if np.asarray(policy_matrix).shape != (dimension, dimension):
        raise ValueError("policy_matrix must have shape (d, d)")
    if np.asarray(policy_bias).shape != (dimension,):
        raise ValueError("policy_bias must have shape (d,)")
    arrays = (
        x,
        xi,
        np.asarray(drift_matrix, dtype=float),
        np.asarray(drift_bias, dtype=float),
        np.asarray(sigma_diag, dtype=float),
        np.asarray(policy_matrix, dtype=float),
        np.asarray(policy_bias, dtype=float),
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all array inputs must be finite")
    sigma = arrays[4]
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    drift = x @ arrays[2].T + arrays[3]
    policy = x @ arrays[5].T + arrays[6]
    diffusion = np.broadcast_to(sigma, (sample_count, dimension))
    return x + (drift + diffusion * policy) * dt + diffusion * xi * np.sqrt(dt)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
x = np.array([[-0.8, -0.5, 0.2], [0.4, 0.5, -0.1]])
xi = np.array([[0.2, -1.1, 0.5], [-0.2, -0.7, 0.9]])
F = np.array([[0.05, -0.08, 0.02], [0.03, -0.12, 0.04], [-0.01, 0.06, -0.09]])
b = np.array([0.01, -0.02, 0.015])
sigma = np.array([0.35, 0.22, 0.18])
U = np.array([[-0.25, 0.08, -0.04], [0.05, -0.18, 0.06], [-0.03, 0.04, -0.12]])
c = np.array([0.02, -0.01, 0.015])
dt = 0.15
""",
            "call": "np.round(advance_controlled_diffusion(x, xi, F, b, sigma, U, c, dt), 12).tolist()",
            "gold_call": "np.round(_oracle_advance_controlled_diffusion(x, xi, F, b, sigma, U, c, dt), 12).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.zeros((1, 1))
xi = np.zeros((1, 1))
F = np.zeros((1, 1))
b = np.zeros(1)
sigma = np.ones(1)
U = np.zeros((1, 1))
c = np.zeros(1)
dt = 0.01
""",
            "call": "advance_controlled_diffusion(x, xi, F, b, sigma, U, c, dt).tolist()",
            "gold_call": "_oracle_advance_controlled_diffusion(x, xi, F, b, sigma, U, c, dt).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.zeros((2, 2))
xi = np.zeros((2, 2))
F = np.zeros((2, 2))
b = np.zeros(2)
sigma = np.array([0.2, 0.0])
U = np.zeros((2, 2))
c = np.zeros(2)
def run_model():
    try:
        advance_controlled_diffusion(x, xi, F, b, sigma, U, c, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_advance_controlled_diffusion(x, xi, F, b, sigma, U, c, 0.1)
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
