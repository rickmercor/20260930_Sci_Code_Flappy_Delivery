"""
Construct the reduced-space operators required by the iterative solver from the system matrix and the assembled global deflation basis, and return them in the prescribed packed representation. The packed output is the concatenation $$ [\operatorname{ravel}(R,\mathrm{order}="C"),\, \operatorname{ravel}(A_c,\mathrm{order}="C"),\, \operatorname{ravel}(C,\mathrm{order}="C")]. $$ For \(A\in\mathbb{R}^{n\times n}\) and \(P\in\mathbb{R}^{n\times k_S}\), the three matrices have shapes $$ R\in\mathbb{R}^{k_S\times n}, \qquad A_c\in\mathbb{R}^{k_S\times k_S}, \qquad C\in\mathbb{R}^{n\times n}.$$

The deflation method couples the original linear system to a lower-dimensional coarse space. The restriction and reduced coarse operator define the coarse correction used by the subsequent solver stages. The returned data are packed deterministically so that downstream stages can reconstruct each operator without ambiguity.

Returns
-------
return np.empty(0, dtype=np.float64)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_coarse_operators(
    A: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Construct the reduced solver representation required by the next stage.

    Parameters
    ----------
    A : np.ndarray
        Two-dimensional square floating-point system matrix with shape
        ``(n, n)``. The matrix must be finite and have the dimensions required
        for the supplied global basis.
    P : np.ndarray
        Two-dimensional finite floating-point global basis representation with
        shape ``(n, k)``. Its row dimension must match the dimension of ``A``,
        and its column dimension determines the reduced-space dimension.

    Returns
    -------
    np.ndarray
        One-dimensional ``float64`` packed representation containing the
        reduced operators required by the following solver stage. The packing
        uses C-order flattening for each component and follows the prescribed
        component ordering of the solver interface:
        restriction-related data, reduced operator data, then full-space
        correction data.

        For an input with ``A.shape == (n, n)`` and ``P.shape == (n, k)``,
        the returned array has length ``k*n + k*k + n*n``.

    Raises
    ------
    ValueError
        If ``A`` or ``P`` is not two-dimensional.
        If ``A`` is not square.
        If either input contains a zero-sized dimension.
        If ``P.shape[0]`` does not equal ``A.shape[0]``.
        If the supplied basis dimensions are incompatible with the system
        matrix.
        If the required reduced operators cannot be formed with the supplied
        dimensions.
        If either input contains a non-finite value.
    TypeError
        If ``A`` or ``P`` cannot be interpreted as numerical NumPy arrays.
    """
    return np.empty(0, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_coarse_operators(
    A: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Reference coarse-space construction with deterministic C-order packing."""
    A = np.asarray(A, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if P.ndim != 2 or P.shape[0] != A.shape[0]:
        raise ValueError("P must have the same row count as A.")

    R = P.T
    Ac = R @ A @ P
    C = P @ np.linalg.solve(Ac, R)

    return np.concatenate([
        R.ravel(order="C"),
        Ac.ravel(order="C"),
        C.ravel(order="C"),
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

A = np.array([
    [6., 1., 0., 0., 0., 0., 0., 0.],
    [1., 6., 1., 0., 0., 0., 0., 0.],
    [0., 1., 6., 1., 0., 0., 0., 0.],
    [0., 0., 1., 6., 1., 0., 0., 0.],
    [0., 0., 0., 1., 6., 1., 0., 0.],
    [0., 0., 0., 0., 1., 6., 1., 0.],
    [0., 0., 0., 0., 0., 1., 6., 1.],
    [0., 0., 0., 0., 0., 0., 1., 6.],
], dtype=np.float64)

P = np.eye(8, 4, dtype=np.float64)
""",
            "call": "build_coarse_operators(A, P)",
            "gold_call": "_oracle_build_coarse_operators(A, P)",
        },
        {
            "setup": """
import numpy as np

A = np.diag(
    np.array([2., 3., 4., 5., 6., 7., 8., 9.], dtype=np.float64)
)

P = np.eye(8, 4, dtype=np.float64)
""",
            "call": "build_coarse_operators(A, P)",
            "gold_call": "_oracle_build_coarse_operators(A, P)",
        },
        {
            "setup": """
import numpy as np

A = np.array([
    [5., 1., 0., 0., 0., 0., 0., 0.],
    [1., 5., 1., 0., 0., 0., 0., 0.],
    [0., 1., 5., 1., 0., 0., 0., 0.],
    [0., 0., 1., 5., 1., 0., 0., 0.],
    [0., 0., 0., 1., 5., 1., 0., 0.],
    [0., 0., 0., 0., 1., 5., 1., 0.],
    [0., 0., 0., 0., 0., 1., 5., 1.],
    [0., 0., 0., 0., 0., 0., 1., 5.],
], dtype=np.float64)

P = np.array([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [1., 1., 0., 0.],
    [0., 1., 1., 0.],
    [0., 0., 1., 1.],
    [1., 0., 0., 1.],
], dtype=np.float64)

P, _ = np.linalg.qr(P, mode="reduced")
""",
            "call": "build_coarse_operators(A, P)",
            "gold_call": "_oracle_build_coarse_operators(A, P)",
        },
    ]
