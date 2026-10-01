"""
Prepare an augmented starting block for the classical block Krylov strategy. The routine validates the block and produces a column-orthonormal starting basis suitable for subsequent classical Krylov reduction.

Classical block Krylov methods use a matrix-valued inner product and maintain orthogonality between individual columns and generated blocks. A reduced QR factorization provides a standard way to obtain an orthonormal starting block. Because that factorization is unique only up to the signs of its columns, the sign of each column is fixed by requiring the diagonal of the triangular factor to be nonnegative, which is also what modified Gram-Schmidt produces and what keeps the first basis column a positive multiple of the first column of the input block.

Returns
-------
np.ndarray, a column-orthonormal starting block with the same shape as Omega_ell, with the sign of each column fixed so that the diagonal of the triangular factor is nonnegative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def classical_orthonormalize(

    Omega_ell: "np.ndarray",

) -> "np.ndarray":

    """Construct the classical orthonormal starting block.

    Args:
        Omega_ell: Two-dimensional starting block.

    Returns:
        The orthonormalized starting block with the same shape as
        ``Omega_ell``.

    Raises:
        ValueError: If ``Omega_ell`` is not two-dimensional, is empty,
            contains non-finite values, has fewer rows than columns, or
            cannot be orthonormalized.
    """

    return orthonormal_block  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_classical_orthonormalize(
    Omega_ell: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of classical initialization."""
    Omega_ell = np.asarray(Omega_ell, dtype=float)

    if Omega_ell.ndim != 2:
        raise ValueError("Omega_ell must be two-dimensional")

    n, ell = Omega_ell.shape

    if n == 0 or ell == 0:
        raise ValueError("Omega_ell must be non-empty")
    if n < ell:
        raise ValueError("Omega_ell must have at least as many rows as columns")
    if not np.all(np.isfinite(Omega_ell)):
        raise ValueError("Omega_ell must contain only finite values")

    rank = np.linalg.matrix_rank(Omega_ell)
    if rank < ell:
        raise ValueError("Omega_ell must have full column rank")

    V1, R = np.linalg.qr(Omega_ell, mode="reduced")

    # Fix the QR sign convention so the diagonal of R is nonnegative.
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0

    V1 = V1 * signs

    return V1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
Omega_ell = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
""",
            "call": """
classical_orthonormalize(Omega_ell)
""",
            "gold_call": """
_oracle_classical_orthonormalize(Omega_ell)
""",
        },
        {
            "setup": """
import numpy as np
Omega_ell = np.eye(4)
""",
            "call": """
classical_orthonormalize(Omega_ell)
""",
            "gold_call": """
_oracle_classical_orthonormalize(Omega_ell)
""",
        },
        {
            "setup": """
import numpy as np
Omega_ell = np.array([
    [1.0, 2.0, 0.5],
    [0.0, 1.0, 1.5],
    [2.0, 0.0, 1.0],
    [1.0, -1.0, 2.0],
])
""",
            "call": """
classical_orthonormalize(Omega_ell)
""",
            "gold_call": """
_oracle_classical_orthonormalize(Omega_ell)
""",
        },
        {
            "setup": """
import numpy as np
Omega_ell = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, -1.0],
])
""",
            "call": """
classical_orthonormalize(Omega_ell)
""",
            "gold_call": """
_oracle_classical_orthonormalize(Omega_ell)
""",
        },
    ]
