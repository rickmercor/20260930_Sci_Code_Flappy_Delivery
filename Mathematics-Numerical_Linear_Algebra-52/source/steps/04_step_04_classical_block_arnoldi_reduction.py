"""
Construct the classical block Krylov basis and its reduced representation for a specified matrix, starting block, and depth. The routine validates the dimensions and iteratively generates the requested block Krylov basis.

Block Arnoldi methods generate a sequence of matrix blocks spanning successive Krylov subspaces. The classical formulation uses matrix-valued block inner products and orthogonalizes newly generated blocks against previously constructed blocks. Each newly generated block is normalized by the reduced QR factorization of the orthogonalized remainder, with the sign of each column fixed so that the diagonal of the triangular factor is nonnegative, the same convention used for the starting block.

Returns
-------
tuple[np.ndarray, np.ndarray], in this order: the leading reduced matrix of shape (depth*ell, depth*ell), then the stacked block Krylov basis of shape (n, depth*ell).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def classical_block_arnoldi(

    A: "np.ndarray",

    V1: "np.ndarray",

    ell: int,

    depth: int,

) -> "tuple[np.ndarray, np.ndarray]":

    """Construct the classical block Krylov reduction.

    Args:
        A: Two-dimensional square matrix defining the reduction.
        V1: Initial block used in the reduction.
        ell: Block size.
        depth: Number of reduction steps.

    Returns:
        The reduced matrix and associated Krylov basis.

    Raises:
        ValueError: If ``A`` is not a finite square two-dimensional array.
        ValueError: If ``V1`` is not a finite two-dimensional array or has
            incompatible dimensions.
        ValueError: If ``ell`` is not a positive integer.
        ValueError: If ``depth`` is not a positive integer.
        ValueError: If the reduction cannot be completed for the requested
            inputs.
    """

    return reduced_matrix, krylov_basis  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_classical_block_arnoldi(
    A: "np.ndarray",
    V1: "np.ndarray",
    ell: int,
    depth: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation of classical block Arnoldi."""
    A = np.asarray(A, dtype=float)
    V1 = np.asarray(V1, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if V1.ndim != 2:
        raise ValueError("V1 must be two-dimensional")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(V1)):
        raise ValueError("inputs must contain only finite values")

    n = A.shape[0]

    if V1.shape[0] != n:
        raise ValueError("V1 has incompatible row dimension")

    if not isinstance(ell, (int, np.integer)) or isinstance(ell, (bool, np.bool_)):
        raise ValueError("ell must be an integer")
    if not isinstance(depth, (int, np.integer)) or isinstance(depth, (bool, np.bool_)):
        raise ValueError("depth must be an integer")
    if ell <= 0:
        raise ValueError("ell must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if V1.shape[1] != ell:
        raise ValueError("V1 must have ell columns")

    if np.linalg.matrix_rank(V1) < ell:
        raise ValueError("V1 must have full column rank")

    V = [V1]
    H = np.zeros((depth * ell, depth * ell), dtype=float)

    for k in range(depth):
        W = A @ V[k]

        for j in range(k + 1):
            Hjk = V[j].T @ W
            H[j * ell:(j + 1) * ell,
              k * ell:(k + 1) * ell] = Hjk
            W = W - V[j] @ Hjk

        if k < depth - 1:
            if not np.all(np.isfinite(W)):
                raise ValueError("Arnoldi remainder is non-finite")

            if np.linalg.matrix_rank(W) < ell:
                raise ValueError("Arnoldi breakdown: remainder is rank deficient")

            Vnext, R = np.linalg.qr(W, mode="reduced")

            # Fix the QR sign convention so the diagonal of R is nonnegative,
            # matching the convention used for the starting block. This makes
            # the generated basis independent of the LAPACK driver.
            signs = np.sign(np.diag(R))
            signs[signs == 0.0] = 1.0

            Vnext = Vnext * signs
            R = signs[:, None] * R

            if np.linalg.matrix_rank(R) < ell:
                raise ValueError("Arnoldi breakdown during QR")

            H[(k + 1) * ell:(k + 2) * ell,
              k * ell:(k + 1) * ell] = R

            V.append(Vnext)

    Vq = np.hstack(V)
    Hqq = H[:depth * ell, :depth * ell]

    return Hqq, Vq

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
A = np.diag([1.0, 2.0, 3.0, 4.0])
V1 = np.eye(4, 2)
ell = 2
depth = 1
""",
            "call": """
classical_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_classical_block_arnoldi(A, V1, ell, depth)
""",
        },
        {
            "setup": """
import numpy as np
A = np.array([
    [2.0, 1.0, 0.0],
    [0.0, 3.0, 1.0],
    [1.0, 0.0, 4.0],
])
V1 = np.eye(3, 1)
ell = 1
depth = 2
""",
            "call": """
classical_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_classical_block_arnoldi(A, V1, ell, depth)
""",
        },
        {
            "setup": """
import numpy as np
A = np.array([
    [2.0, 0.2, 0.1, 0.0],
    [0.0, 1.5, 0.3, 0.2],
    [0.1, 0.0, 2.5, 0.4],
    [0.2, 0.1, 0.0, 3.0],
])
V1 = np.eye(4, 2)
ell = 2
depth = 2
""",
            "call": """
classical_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_classical_block_arnoldi(A, V1, ell, depth)
""",
        },
        {
            "setup": """
import numpy as np
A = np.eye(3)
V1 = np.ones((3, 1))
ell = 1
depth = 2
""",
            "call": """
classical_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_classical_block_arnoldi(A, V1, ell, depth)
""",
        },
    ]
