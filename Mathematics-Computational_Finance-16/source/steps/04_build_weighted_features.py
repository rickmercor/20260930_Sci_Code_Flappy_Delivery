"""
The explicit BSDE regression operator is the identity plus a noise-weighted gradient, so its feature construction must represent both the pointwise value and the coordinate-specific derivative contribution that the operator mixes in. The feature construction uses one all-value channel and one channel per coordinate. In derivative channel i, every coordinate keeps phi(z) = [1, z, z^2] except coordinate i, which uses Sigma_i phi'(z), where Sigma_i = s_i xi_i sqrt(dt).

Inputs
------
points: Left-endpoint states with shape (K, d).
noises: Fixed innovations with shape (K, d).
sigma_diag: Positive diffusion diagonal with shape (d,).
dt: Positive time increment.

Returns
-------
weighted_features: Float array of shape (d + 1, K, d, 3).

Returns
-------
np.ndarray of shape (d + 1, K, d, 3), the value and weighted derivative features
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_weighted_features(
    points: np.ndarray,
    noises: np.ndarray,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Build value and noise-weighted derivative feature channels.

    Parameters
    ----------
    points : np.ndarray
        Left-endpoint states with shape (K, d).
    noises : np.ndarray
        Standard-normal innovations with the same shape as `points`.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.
    dt : float
        Positive time increment.

    Raises
    ------
    ValueError
        If an input has an incompatible shape or non-finite entry, if
        `sigma_diag` is not positive, or if `dt` is not finite and positive.

    Returns
    -------
    weighted_features : np.ndarray
        Channel tensor with shape (d + 1, K, d, 3).
    """
    return weighted_features  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_build_weighted_features(
    points: np.ndarray,
    noises: np.ndarray,
    sigma_diag: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reference implementation."""
    x = np.asarray(points, dtype=float)
    xi = np.asarray(noises, dtype=float)
    sigma = np.asarray(sigma_diag, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("points must be a non-empty two-dimensional array")
    if xi.shape != x.shape:
        raise ValueError("noises must have the same shape as points")
    if sigma.shape != (x.shape[1],):
        raise ValueError("sigma_diag must have shape (d,)")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(xi)) or not np.all(np.isfinite(sigma)):
        raise ValueError("array inputs must be finite")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    sample_count, dimension = x.shape
    basis = np.stack((np.ones_like(x), x, x * x), axis=2)
    derivative = np.stack((np.zeros_like(x), np.ones_like(x), 2.0 * x), axis=2)
    weighted = np.broadcast_to(basis, (dimension + 1, sample_count, dimension, 3)).copy()
    sigma_noise = xi * sigma[None, :] * np.sqrt(dt)
    for coordinate in range(dimension):
        weighted[coordinate + 1, :, coordinate, :] = (
            sigma_noise[:, coordinate, None] * derivative[:, coordinate, :]
        )
    return weighted

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
sigma = np.array([0.35, 0.22, 0.18])
""",
            "call": "np.round(build_weighted_features(x, xi, sigma, 0.15), 12).tolist()",
            "gold_call": "np.round(_oracle_build_weighted_features(x, xi, sigma, 0.15), 12).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.zeros((1, 1))
xi = np.zeros((1, 1))
sigma = np.ones(1)
""",
            "call": "build_weighted_features(x, xi, sigma, 0.01).tolist()",
            "gold_call": "_oracle_build_weighted_features(x, xi, sigma, 0.01).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.zeros((2, 2))
xi = np.zeros((1, 2))
sigma = np.ones(2)
def run_model():
    try:
        build_weighted_features(x, xi, sigma, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_build_weighted_features(x, xi, sigma, 0.1)
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
