"""
Construct the global block Krylov basis and its reduced representation for a specified matrix, starting block, and depth.

The global formulation treats each matrix block as a single Krylov object through the scalar inner product trace(M_1^T M_2). The resulting recurrence differs from classical block Arnoldi because block interactions are represented by scalar multiples of the identity: each block of the reduced matrix is trace(V_i^T A V_j) I_ell, and each subdiagonal block is the Frobenius norm of the orthogonalized remainder times I_ell. No sign convention is needed, because that norm is positive.

Returns
-------
tuple[np.ndarray, np.ndarray], in this order: the leading global reduced matrix of shape (depth*ell, depth*ell), then the stacked global Krylov basis of shape (n, depth*ell).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def global_block_arnoldi(

    A: "np.ndarray",

    V1: "np.ndarray",

    ell: int,

    depth: int,

) -> "tuple[np.ndarray, np.ndarray]":

    """Construct the global block Krylov reduction.

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


def _oracle_global_block_arnoldi(
    A: "np.ndarray",
    V1: "np.ndarray",
    ell: int,
    depth: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation of global block Arnoldi."""
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

    norm_v1 = np.linalg.norm(V1, ord="fro")
    if not np.isfinite(norm_v1) or norm_v1 <= 0.0:
        raise ValueError("V1 must have positive Frobenius norm")

    V = [V1]
    H = np.zeros((depth * ell, depth * ell), dtype=float)

    for k in range(depth):
        W = A @ V[k]

        for j in range(k + 1):
            scalar = np.trace(V[j].T @ W)
            Hjk = scalar * np.eye(ell)

            H[j * ell:(j + 1) * ell,
              k * ell:(k + 1) * ell] = Hjk

            W = W - V[j] @ Hjk

        if k < depth - 1:
            beta = np.linalg.norm(W, ord="fro")

            if not np.isfinite(beta) or beta <= 0.0:
                raise ValueError("Arnoldi breakdown: zero or non-finite remainder")

            H[(k + 1) * ell:(k + 2) * ell,
              k * ell:(k + 1) * ell] = beta * np.eye(ell)

            V.append(W / beta)

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
V1 = np.eye(4, 2) / np.sqrt(2.0)
ell = 2
depth = 1
""",
            "call": """
global_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_global_block_arnoldi(A, V1, ell, depth)
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
V1 = np.ones((3, 1))
V1 = V1 / np.linalg.norm(V1, ord="fro")
ell = 1
depth = 2
""",
            "call": """
global_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_global_block_arnoldi(A, V1, ell, depth)
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
V1 = np.eye(4, 2) / np.sqrt(2.0)
ell = 2
depth = 2
""",
            "call": """
global_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_global_block_arnoldi(A, V1, ell, depth)
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
global_block_arnoldi(A, V1, ell, depth)
""",
            "gold_call": """
_oracle_global_block_arnoldi(A, V1, ell, depth)
""",
        },
    ]
