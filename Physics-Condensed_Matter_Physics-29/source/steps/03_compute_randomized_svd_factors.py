"""
Compute deterministic low-rank singular factors of the opposite-spin diffusion block.

The numerical diffusion gauge uses a randomized range finder to compress the diffusion column space before a projected singular-value decomposition. A fixed sketch distribution, tolerance rule, and phase convention are required because later stochastic trajectories depend on the orientation of the retained singular vectors.

Returns
-------
np.ndarray, the packed complex low-rank singular factors with shape (2*m + 1, k)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_randomized_svd_factors(
    diffusion_block: "np.ndarray",
    rank: int,
    seed: int,
    tolerance_factor: float = 64.0,
) -> "np.ndarray":
    """Compute packed low-rank factors for the numerical diffusion gauge.

    Parameters
    ----------
    diffusion_block : np.ndarray
        Finite complex square opposite-spin block with shape ``(m, m)``.
    rank : int
        Sketch width satisfying ``1 <= rank <= m``.
    seed : int
        Seed used by ``numpy.random.default_rng`` for a real standard-normal
        sketch of shape ``(m, rank)``.
    tolerance_factor : float, default=64.0
        Positive multiplier in the singular-value cutoff
        ``tolerance_factor * eps * m * s_max``.

    Returns
    -------
    factors : np.ndarray
        Complex array with shape ``(2*m + 1, k)``. The first ``m`` rows are
        left singular vectors, row ``m`` stores singular values, and the final
        ``m`` rows are right singular vectors. The phase pivot is the smallest
        index within ``1e-10`` relative of each left-vector magnitude maximum.

    Raises
    ------
    ValueError
        If the block, rank, seed, or tolerance is outside the documented domain.
    """
    return factors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np


def _oracle_compute_randomized_svd_factors(
    diffusion_block: "np.ndarray",
    rank: int,
    seed: int,
    tolerance_factor: float = 64.0,
) -> "np.ndarray":
    """Reference implementation."""
    block = np.asarray(diffusion_block, dtype=complex)
    if (
        block.ndim != 2
        or block.shape[0] != block.shape[1]
        or block.shape[0] < 1
        or not np.all(np.isfinite(block))
    ):
        raise ValueError("diffusion_block must be a finite nonempty square array")
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise ValueError("rank must be an integer")
    rank_value = int(rank)
    if rank_value < 1 or rank_value > block.shape[0]:
        raise ValueError("rank must satisfy 1 <= rank <= matrix dimension")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(tolerance_factor, bool) or not isinstance(tolerance_factor, Real):
        raise ValueError("tolerance_factor must be positive and finite")
    tolerance_value = float(tolerance_factor)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance_factor must be positive and finite")

    scaled = -0.5j * block
    rng = np.random.default_rng(int(seed))
    sketch = rng.standard_normal((block.shape[1], rank_value))
    image = scaled @ sketch
    basis, _ = np.linalg.qr(image, mode="reduced")
    projected = basis.conj().T @ scaled
    reduced_left, singular_values, right_adjoint = np.linalg.svd(
        projected, full_matrices=False
    )
    left = basis @ reduced_left

    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.empty((2 * block.shape[0] + 1, 0), dtype=complex)

    cutoff = (
        tolerance_value
        * np.finfo(float).eps
        * max(scaled.shape)
        * singular_values[0]
    )
    keep = singular_values > cutoff
    left = left[:, keep]
    singular_values = singular_values[keep]
    right_adjoint = right_adjoint[keep, :]

    for column in range(left.shape[1]):
        magnitudes = np.abs(left[:, column])
        maximum = float(np.max(magnitudes))
        candidates = np.flatnonzero(magnitudes >= maximum * (1.0 - 1.0e-10))
        pivot = int(candidates[0])
        phase = np.exp(-1j * np.angle(left[pivot, column]))
        left[:, column] *= phase
        right_adjoint[column, :] *= np.conj(phase)

    right = right_adjoint.conj().T
    return np.vstack(
        (left, singular_values.astype(complex)[None, :], right)
    )

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
interaction = 1.3
diffusion_block = _oracle_compute_opposite_spin_diffusion(n_up.copy(), n_down.copy(), interaction)
""",
            "call": "compute_randomized_svd_factors(diffusion_block.copy(), 6, 314159, 64.0)",
            "gold_call": "_oracle_compute_randomized_svd_factors(diffusion_block.copy(), 6, 314159, 64.0)",
        },
        {
            "setup": """import numpy as np
u = np.array([1.0, 2.0j], dtype=complex)
v = np.array([0.5 - 0.2j, -1.0j], dtype=complex)
scaled = np.outer(u, np.conj(v))
diffusion_block = 2.0j * scaled
""",
            "call": "compute_randomized_svd_factors(diffusion_block.copy(), 1, 7, 64.0)",
            "gold_call": "_oracle_compute_randomized_svd_factors(diffusion_block.copy(), 1, 7, 64.0)",
        },
        {
            "setup": """import numpy as np
diffusion_block = np.zeros((4, 4), dtype=complex)
""",
            "call": "compute_randomized_svd_factors(diffusion_block.copy(), 3, 11, 64.0)",
            "gold_call": "_oracle_compute_randomized_svd_factors(diffusion_block.copy(), 3, 11, 64.0)",
        },
        {
            "setup": """import numpy as np
scaled = np.diag([1.0, 1.0e-16, 0.0]).astype(complex)
diffusion_block = 2.0j * scaled
""",
            "call": "compute_randomized_svd_factors(diffusion_block.copy(), 3, 19, 64.0)",
            "gold_call": "_oracle_compute_randomized_svd_factors(diffusion_block.copy(), 3, 19, 64.0)",
        },
        {
            "setup": """import numpy as np
diffusion_block = np.eye(2, dtype=complex)
def run_model():
    try:
        compute_randomized_svd_factors(diffusion_block.copy(), 3, 0, 64.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_randomized_svd_factors(diffusion_block.copy(), 3, 0, 64.0)
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
