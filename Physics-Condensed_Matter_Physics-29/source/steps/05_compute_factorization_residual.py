"""
Measure how accurately the structured noise factor reproduces the diffusion tensor.

Complex phase-space Itô equations require the diffusion factorization associated with the stochastic covariance, not a Hermitian positive-semidefinite decomposition. A normalized Frobenius residual provides a scale-independent diagnostic while retaining a meaningful zero-diffusion limit.

Returns
-------
float, the nonnegative relative Frobenius residual of D - B B^T as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_factorization_residual(
    diffusion_block: "np.ndarray",
    gauge: "np.ndarray",
) -> float:
    """Compute the relative residual of the complex diffusion factorization.

    Parameters
    ----------
    diffusion_block : np.ndarray
        Finite complex square opposite-spin block with shape ``(m, m)``.
    gauge : np.ndarray
        Finite complex noise matrix with ``2*m`` rows.

    Returns
    -------
    residual : float
        Nonnegative native Python float. For a zero target diffusion tensor, the
        absolute Frobenius residual is returned.

    Raises
    ------
    ValueError
        If input shapes or values are incompatible.
    """
    return residual

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_factorization_residual(
    diffusion_block: "np.ndarray",
    gauge: "np.ndarray",
) -> float:
    """Reference implementation."""
    block = np.asarray(diffusion_block, dtype=complex)
    noise = np.asarray(gauge, dtype=complex)
    if (
        block.ndim != 2
        or block.shape[0] != block.shape[1]
        or block.shape[0] < 1
        or noise.ndim != 2
        or noise.shape[0] != 2 * block.shape[0]
        or not np.all(np.isfinite(block))
        or not np.all(np.isfinite(noise))
    ):
        raise ValueError("diffusion_block and gauge have incompatible finite shapes")

    zeros = np.zeros_like(block)
    target = np.block([[zeros, block], [block.T, zeros]])
    error = float(np.linalg.norm(target - noise @ noise.T, ord="fro"))
    scale = float(np.linalg.norm(target, ord="fro"))
    return error if scale == 0.0 else error / scale

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic test specifications."""
    return [
        {
            "setup": """import numpy as np
n_up = np.array([
    [0.82 + 0.00j, 0.12 + 0.05j, -0.04 + 0.02j],
    [0.12 - 0.05j, 0.47 + 0.00j, 0.09 - 0.03j],
    [-0.04 - 0.02j, 0.09 + 0.03j, 0.21 + 0.00j],
], dtype=complex)
n_down = np.array([
    [0.18 + 0.00j, -0.07 + 0.02j, 0.05 - 0.01j],
    [-0.07 - 0.02j, 0.53 + 0.00j, -0.11 + 0.04j],
    [0.05 + 0.01j, -0.11 - 0.04j, 0.79 + 0.00j],
], dtype=complex)
hopping = np.array([
    [0.0, 1.0, 0.35],
    [1.0, 0.0, 0.8],
    [0.35, 0.8, 0.0],
], dtype=float)
block = _oracle_compute_opposite_spin_diffusion(n_up, n_down, 1.3)
factors = _oracle_compute_randomized_svd_factors(block, 6, 314159, 64.0)
gauge = _oracle_construct_diffusion_gauge(factors)
""",
            "call": "compute_factorization_residual(block.copy(), gauge.copy())",
            "gold_call": "_oracle_compute_factorization_residual(block.copy(), gauge.copy())",
        },
        {
            "setup": """import numpy as np
block = np.array([[2.0j]], dtype=complex)
gauge = np.array([[1.0, 1.0j], [1.0j, 1.0]], dtype=complex)
""",
            "call": "compute_factorization_residual(block.copy(), gauge.copy())",
            "gold_call": "_oracle_compute_factorization_residual(block.copy(), gauge.copy())",
        },
        {
            "setup": """import numpy as np
block = np.array([[2.0j]], dtype=complex)
gauge = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
""",
            "call": "compute_factorization_residual(block.copy(), gauge.copy())",
            "gold_call": "_oracle_compute_factorization_residual(block.copy(), gauge.copy())",
        },
        {
            "setup": """import numpy as np
block = np.zeros((3, 3), dtype=complex)
gauge = np.empty((6, 0), dtype=complex)
""",
            "call": "compute_factorization_residual(block.copy(), gauge.copy())",
            "gold_call": "_oracle_compute_factorization_residual(block.copy(), gauge.copy())",
        },
        {
            "setup": """import numpy as np
block = np.eye(2, dtype=complex)
gauge = np.zeros((3, 2), dtype=complex)
def run_model():
    try:
        compute_factorization_residual(block.copy(), gauge.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_factorization_residual(block.copy(), gauge.copy())
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
