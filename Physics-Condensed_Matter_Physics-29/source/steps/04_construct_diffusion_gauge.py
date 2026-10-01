"""
Construct the structured complex noise factor from packed singular vectors.

The opposite-spin Hubbard diffusion tensor admits a lower-rank numerical gauge assembled from the square roots of the retained singular values and two coupled singular-vector sectors. The conjugation and imaginary block phases are essential for reproducing the complex Itô diffusion rather than a Hermitian covariance.

Returns
-------
np.ndarray, the structured complex noise factor with shape (2*m, 2*k)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_diffusion_gauge(factors: "np.ndarray") -> "np.ndarray":
    """Construct the structured numerical diffusion gauge.

    Parameters
    ----------
    factors : np.ndarray
        Packed complex factor array with shape ``(2*m + 1, k)``: left singular
        vectors, one singular-value row, then right singular vectors.

    Returns
    -------
    gauge : np.ndarray
        Complex noise matrix with shape ``(2*m, 2*k)``.

    Raises
    ------
    ValueError
        If the packed array has an invalid shape, non-finite values, or singular
        values that are not finite nonnegative reals.
    """
    return gauge

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_diffusion_gauge(factors: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    packed = np.asarray(factors, dtype=complex)
    if (
        packed.ndim != 2
        or packed.shape[0] < 3
        or packed.shape[0] % 2 != 1
        or not np.all(np.isfinite(packed))
    ):
        raise ValueError("factors must have finite shape (2*m + 1, k)")
    m = (packed.shape[0] - 1) // 2
    k = packed.shape[1]
    if k > m:
        raise ValueError("the retained rank cannot exceed m")
    if k == 0:
        return np.empty((2 * m, 0), dtype=complex)

    left = packed[:m, :]
    singular_row = packed[m, :]
    right = packed[m + 1 :, :]
    if np.max(np.abs(singular_row.imag)) > 1.0e-12:
        raise ValueError("singular values must be real")
    singular_values = singular_row.real
    if np.any(singular_values < 0.0):
        raise ValueError("singular values must be nonnegative")

    root = np.sqrt(singular_values)
    upper = left * root[None, :]
    lower = np.conj(right) * root[None, :]
    return np.block(
        [[upper, 1j * upper], [1j * lower, lower]]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic test specifications."""
    return [
        {
            "setup": """import numpy as np
root2 = np.sqrt(2.0)
left = np.eye(2, dtype=complex)
right = np.array([[1.0 / root2, 1.0j / root2], [1.0j / root2, 1.0 / root2]], dtype=complex)
singular_values = np.array([4.0, 1.0], dtype=complex)
factors = np.vstack((left, singular_values[None, :], right))
""",
            "call": "construct_diffusion_gauge(factors.copy())",
            "gold_call": "_oracle_construct_diffusion_gauge(factors.copy())",
        },
        {
            "setup": """import numpy as np
left = np.array([[1.0], [0.0], [0.0]], dtype=complex)
right = np.array([[0.0], [1.0j], [0.0]], dtype=complex)
factors = np.vstack((left, np.array([[0.25 + 0.0j]]), right))
""",
            "call": "construct_diffusion_gauge(factors.copy())",
            "gold_call": "_oracle_construct_diffusion_gauge(factors.copy())",
        },
        {
            "setup": """import numpy as np
factors = np.empty((5, 0), dtype=complex)
""",
            "call": "construct_diffusion_gauge(factors.copy())",
            "gold_call": "_oracle_construct_diffusion_gauge(factors.copy())",
        },
        {
            "setup": """import numpy as np
n_up = np.array([[0.7, 0.08 + 0.03j], [0.05 - 0.02j, 0.3]], dtype=complex)
n_down = np.array([[0.3, -0.04 + 0.01j], [-0.06 - 0.02j, 0.7]], dtype=complex)
block = _oracle_compute_opposite_spin_diffusion(n_up, n_down, 0.85)
factors = _oracle_compute_randomized_svd_factors(block, 4, 1234, 64.0)
""",
            "call": "construct_diffusion_gauge(factors.copy())",
            "gold_call": "_oracle_construct_diffusion_gauge(factors.copy())",
        },
        {
            "setup": """import numpy as np
factors = np.array([[1.0], [-1.0], [1.0]], dtype=complex)
def run_model():
    try:
        construct_diffusion_gauge(factors.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_construct_diffusion_gauge(factors.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
