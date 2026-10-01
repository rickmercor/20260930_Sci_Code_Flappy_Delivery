"""
Complete the deterministic numerical computation and return the spectral condition number of the final full-space Gram matrix.

The final scalar is sketch-dependent: the randomized orthogonalization determines the non-orthonormal full-space search basis, and the resulting Gram matrix G_j records that geometry. The final condition number is therefore a direct numerical diagnostic of the source method's sketched search space.

Returns
-------
float, the spectral condition number kappa_2(G_9) of the final full-space Gram matrix after the complete restarted two-cycle computation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_once_restarted_ritz_entry(
    n: int = 96,
    k: int = 3,
    jmax: int = 9,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> float:
    """Run the complete deterministic calculation and return the final Gram condition number.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Number of retained spectral states.
    jmax : int
        Maximum search-space dimension.
    sketch_size : int
        Number of rows in the sketch.
    zeta : int
        Positive sketch sparsity parameter.
    seed : int
        Seed for the deterministic sketch construction.

    Returns
    -------
    float
        Spectral condition number of the final full-space Gram matrix.

    Raises
    ------
    ValueError
        If any numerical parameter is invalid, the dimensions are
        inconsistent, or the final Gram matrix is not positive definite.

    Notes
    -----
    The returned scalar is the final spectral condition number specified by the benchmark.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_once_restarted_ritz_entry(
    n: int = 96,
    k: int = 3,
    jmax: int = 9,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> float:
    """Reference implementation integrating the earlier scientific steps."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer")
    if not isinstance(jmax, (int, np.integer)) or isinstance(jmax, (bool, np.bool_)):
        raise ValueError("jmax must be an integer")
    if int(k) != 3:
        raise ValueError("this benchmark fixes k=3")
    if int(jmax) < 2 * int(k) or int(jmax) % int(k) != 0:
        raise ValueError("invalid jmax")

    k = int(k)
    jmax = int(jmax)

    A, V0, S = _oracle_build_instance(
        n, sketch_size=sketch_size, zeta=zeta, seed=seed
    )
    A, S, Vt, Q = _oracle_rcgs_initial((A, V0, S))
    W = A @ Vt

    # First cycle: two normal sketch-orthonormal expansion transitions.
    extraction = _oracle_generalized_ritz((A, S, Vt, Q), k=k)
    while extraction[2].shape[1] < jmax:
        correction = _oracle_davidson_correction(extraction)
        A, S, Vt, Q, W = _oracle_sketched_block_expansion(
            (A, S, Vt, Q, W, correction[-1])
        )
        extraction = _oracle_generalized_ritz((A, S, Vt, Q), k=k)

    # One source-defined restart, with no re-sketch-orthonormalization.
    restarted = _oracle_restart_rotation(extraction)
    A, S, Vt, Q, W = restarted

    # Second cycle: two post-restart/oblique expansion transitions.
    extraction = _oracle_generalized_ritz((A, S, Vt, Q), k=k)
    while extraction[2].shape[1] < jmax:
        correction = _oracle_davidson_correction(extraction)
        A, S, Vt, Q, W = _oracle_post_restart_expansion(
            (A, S, Vt, Q, W, correction[-1])
        )
        extraction = _oracle_generalized_ritz((A, S, Vt, Q), k=k)

    Vt_final = extraction[2]
    eigs = np.linalg.eigvalsh(Vt_final.T @ Vt_final)
    if eigs.size != Vt_final.shape[1] or np.any(eigs <= 0.0):
        raise ValueError("final Gram matrix is not positive definite")

    result = float(eigs[-1] / eigs[0])
    if not np.isfinite(result):
        raise ValueError("result is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and alternate-seed test specifications."""
    return [
        {
            "setup": """
import numpy as np
n=96
k=3
jmax=9
sketch_size=45
zeta=4
seed=2026
""",
            "call": "solve_once_restarted_ritz_entry(n, k, jmax, sketch_size, zeta, seed)",
            "gold_call": "_oracle_solve_once_restarted_ritz_entry(n, k, jmax, sketch_size, zeta, seed)",
            "tol": 1e-7,
        },
        {
            "setup": """
import numpy as np
n=8
k=3
jmax=6
sketch_size=8
zeta=1
seed=7
""",
            "call": "solve_once_restarted_ritz_entry(n, k, jmax, sketch_size, zeta, seed)",
            "gold_call": "_oracle_solve_once_restarted_ritz_entry(n, k, jmax, sketch_size, zeta, seed)",
            "tol": 1e-7,
        },
        {
            "setup": """
import numpy as np
n=12
k=3
jmax=6
sketch_size=6
zeta=2
seed=123
""",
            "call": "solve_once_restarted_ritz_entry(n, k, jmax, sketch_size, zeta, seed)",
            "gold_call": "_oracle_solve_once_restarted_ritz_entry(n, k, jmax, sketch_size, zeta, seed)",
            "tol": 1e-7,
        },
    ]
