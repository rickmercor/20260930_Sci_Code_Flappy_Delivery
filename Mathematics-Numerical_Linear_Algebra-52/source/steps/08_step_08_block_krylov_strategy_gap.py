"""
Assemble the randomized initialization, classical and global reductions, and corresponding matrix-function approximations into one deterministic comparison. The routine returns the Euclidean distance between the two resulting approximations.

Classical and global block Krylov methods use different block inner products and normalization conventions. Applying both strategies to the same matrix, target vector, randomized starting block, and Krylov depth provides a direct numerical measure of the difference between their resulting approximations. Both strategies must start from one realized randomized block: the classical branch orthonormalizes it, the global branch normalizes it in the Frobenius norm, and the global reconstruction scaling is taken from the block before that normalization.

Returns
-------
float, the Euclidean 2-norm gap ||y_classical - y_global||_2 as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def block_krylov_strategy_gap(
    A: "np.ndarray",
    b: "np.ndarray",
    ell: int,
    depth: int,
    seed: int,
) -> float:
    """Compute the Euclidean gap between classical and global approximations.

    Args:
        A: Square system matrix.
        b: One-dimensional target vector compatible with ``A``.
        ell: Block size compatible with the supplied matrix.
        depth: Number of reduction steps.
        seed: Integer seed defining the deterministic randomized instance.

    Returns:
        The Euclidean 2-norm of the difference between the classical and
        global matrix-function approximations, as a native Python float.

    Raises:
        ValueError: If the supplied matrix or vector has invalid dimensions
            or contains non-finite values.
        ValueError: If the supplied block size, depth, or seed is invalid.
        ValueError: If the computation cannot be completed for the supplied
            inputs.
    """
    return gap  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_block_krylov_strategy_gap(
    A: "np.ndarray",
    b: "np.ndarray",
    ell: int,
    depth: int,
    seed: int,
) -> float:
    """Reference solution for the complete block Krylov comparison."""

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if b.ndim != 1:
        raise ValueError("b must be one-dimensional")

    if b.size != A.shape[0]:
        raise ValueError("b has incompatible dimension")

    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")

    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values")

    if not isinstance(ell, (int, np.integer)) or isinstance(
        ell, (bool, np.bool_)
    ):
        raise ValueError("ell must be an integer")

    if not isinstance(depth, (int, np.integer)) or isinstance(
        depth, (bool, np.bool_)
    ):
        raise ValueError("depth must be an integer")

    if not isinstance(seed, (int, np.integer)) or isinstance(
        seed, (bool, np.bool_)
    ):
        raise ValueError("seed must be an integer")

    if ell < 2 or ell > A.shape[0]:
        raise ValueError("ell must satisfy 2 <= ell <= n")

    if depth < 1:
        raise ValueError("depth must be positive")

    # Step 01: randomized augmented block
    # This is the original, unnormalized paper-specific block.
    Omega_ell = _oracle_build_augmented_block(
        b,
        ell,
        seed,
    )

    # Step 02: classical starting block
    V1_C = _oracle_classical_orthonormalize(
        Omega_ell
    )

    # Step 03: global starting block
    # This is normalized for the global Arnoldi recurrence.
    V1_G = _oracle_global_normalize(
        Omega_ell
    )

    # Step 04: classical block Arnoldi
    Hqq_C, Vq_C = _oracle_classical_block_arnoldi(
        A,
        V1_C,
        ell,
        depth,
    )

    # Step 05: global block Arnoldi
    Hqq_G, Vq_G = _oracle_global_block_arnoldi(
        A,
        V1_G,
        ell,
        depth,
    )

    # Step 06: classical matrix-function approximation
    y_C = _oracle_classical_fAb_approx(
        Vq_C,
        Hqq_C,
        b,
    )

    # Step 07: global matrix-function approximation
    # The scaling comes from the unnormalized Omega_ell, not from V1_G,
    # whose Frobenius norm is 1.
    y_G = _oracle_global_fAb_approx(
        Vq_G,
        Hqq_G,
        Omega_ell,
    )

    # Final comparison
    return float(
        np.linalg.norm(y_C - y_G, ord=2)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    return [

        {
            "setup": """import numpy as np

n = 10

A = np.array([[1.0/(i + 2*j + 1) for j in range(n)] for i in range(n)])

b = np.arange(1, n+1, dtype=float)

ell = 4

depth = 3

seed = 552017

""",

            "call": "block_krylov_strategy_gap(A, b, ell, depth, seed)",

            "gold_call": "_oracle_block_krylov_strategy_gap(A, b, ell, depth, seed)",

        },

        {
            "setup": """import numpy as np

n = 6

A = np.diag(np.arange(1, n+1, dtype=float))

b = np.ones(n)

ell = 2

depth = 3

seed = 0

""",

            "call": "block_krylov_strategy_gap(A, b, ell, depth, seed)",

            "gold_call": "_oracle_block_krylov_strategy_gap(A, b, ell, depth, seed)",

        },

        {
            "setup": """import numpy as np

n = 6

A = np.array([[1.0/(i + j + 1) for j in range(n)] for i in range(n)])

b = np.arange(1, n+1, dtype=float)

ell = 3

depth = 2

seed = 7

""",

            "call": "block_krylov_strategy_gap(A, b, ell, depth, seed)",

            "gold_call": "_oracle_block_krylov_strategy_gap(A, b, ell, depth, seed)",

        },

        {
            "setup": """import numpy as np

n = 4

A = np.eye(n)

b = np.ones(n)

ell = 2

depth = 1

seed = 0

""",

            "call": "block_krylov_strategy_gap(A, b, ell, depth, seed)",

            "gold_call": "_oracle_block_krylov_strategy_gap(A, b, ell, depth, seed)",

        },

    ]
