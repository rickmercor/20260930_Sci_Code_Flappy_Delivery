"""
Advance the risk-neutral asset process on the uniform grid.

Under the risk-neutral measure, the Euler-Maruyama state update must retain the

multiplicative dependence of each asset on its current level and respect the

asset-by-Brownian orientation of the volatility matrix. Every grid state is

retained because later BSDE components begin and end at different dates.

Inputs

------

x0: Positive initial asset vector of shape (d,).

q: Dividend-yield vector of shape (d,).

sigma: Volatility matrix of shape (d, dim_w).

r: Risk-free rate.

h: Positive uniform time step.

increments: Brownian array of shape (n_steps, n_paths, dim_w).

Returns

-------

paths: Float array of shape (n_steps + 1, n_paths, d).

Returns
-------
np.ndarray of shape (n_steps + 1, n_paths, d), asset paths as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def simulate_gbm_paths(
    x0: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    r: float,
    h: float,
    increments: np.ndarray,
) -> np.ndarray:
    """Simulate risk-neutral geometric Brownian paths by Euler-Maruyama.

    Parameters
    ----------
    x0 : np.ndarray
        Positive initial asset vector of shape ``(d,)``.
    q : np.ndarray
        Finite dividend-yield vector of shape ``(d,)``.
    sigma : np.ndarray
        Finite volatility matrix of shape ``(d, dim_w)``.
    r : float
        Finite risk-free rate.
    h : float
        Positive finite step size.
    increments : np.ndarray
        Brownian increments of shape ``(n_steps, n_paths, dim_w)``.

    Raises
    ------
    ValueError
        If dimensions disagree, any input is nonfinite, x0 or h is not
        positive, or an Euler update produces a nonpositive asset value.

    Returns
    -------
    paths : np.ndarray
        Asset paths of shape ``(n_steps + 1, n_paths, d)``.
    """
    return paths  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_simulate_gbm_paths(
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

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """x0 = np.array([48.0, 52.0])
q = np.array([0.01, 0.015])
sigma = np.array([[0.24, 0.05], [0.08, 0.20]])
r = 0.06
h = 0.125
increments = np.array([[[0.10, -0.20], [0.05, 0.03]], [[-0.04, 0.08], [0.02, -0.01]]])
""",
            "call": "np.round(simulate_gbm_paths(x0, q, sigma, r, h, increments), 12).tolist()",
            "gold_call": "np.round(_oracle_simulate_gbm_paths(x0, q, sigma, r, h, increments), 12).tolist()",
        },
        {
            "setup": """x0 = np.array([10.0])
q = np.array([0.0])
sigma = np.array([[0.2]])
r = 0.0
h = 0.5
increments = np.zeros((1, 1, 1))
""",
            "call": "np.round(simulate_gbm_paths(x0, q, sigma, r, h, increments), 12).tolist()",
            "gold_call": "np.round(_oracle_simulate_gbm_paths(x0, q, sigma, r, h, increments), 12).tolist()",
        },
        {
            "setup": """x0 = np.array([10.0, 11.0])
q = np.zeros(2)
sigma = np.eye(3)
increments = np.zeros((1, 1, 2))
def run_model():
    try:
        simulate_gbm_paths(x0, q, sigma, 0.0, 0.1, increments)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_simulate_gbm_paths(x0, q, sigma, 0.0, 0.1, increments)
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
