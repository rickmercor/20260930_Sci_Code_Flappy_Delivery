"""
Construct the initial block used by the block Krylov procedures from a target vector and reproducible randomized directions.

Block Krylov methods can enrich the starting subspace by combining the vector of interest with additional randomized directions. A fixed pseudorandom seed makes this initialization deterministic and reproducible across evaluations.

Returns
-------
np.ndarray, the augmented block whose first column is b and whose remaining columns are the enrichment directions orthonormalized against b/||b||_2 and against one another.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_augmented_block(

    b: "np.ndarray",

    ell: int,

    seed: int,

) -> "np.ndarray":

    """Construct a deterministic augmented block from a target vector.

    Args:
        b: One-dimensional target vector.
        ell: Requested block size.
        seed: Integer seed defining the deterministic randomized instance.

    Returns:
        The constructed augmented block.

    Raises:
        ValueError: If ``b`` is not one-dimensional, contains non-finite
            values, or has zero norm.
        ValueError: If ``ell`` is not an integer satisfying
            ``2 <= ell <= len(b)``.
        ValueError: If ``seed`` is not an integer.
        ValueError: If the randomized construction cannot produce a valid
            augmented block.
    """

    return augmented_block  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_augmented_block(

    b: "np.ndarray",

    ell: int,

    seed: int,

) -> "np.ndarray":

    """
    Construct the paper-specific randomized starting block.

    The cited paper defines the alternative initialization for the
    matrix-function setting by using b as the first column and
    Gaussian enrichment directions that are orthonormalized relative
    to b / ||b||_2 and to each other.
    """

    b = np.asarray(b, dtype=float)

    if b.ndim != 1:
        raise ValueError("b must be one-dimensional.")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values.")
    if np.linalg.norm(b) == 0.0:
        raise ValueError("b must have nonzero norm.")
    if not isinstance(ell, (int, np.integer)):
        raise ValueError("ell must be an integer.")
    if ell < 2 or ell > b.size:
        raise ValueError("ell must satisfy 2 <= ell <= len(b).")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")

    n = b.size

    rng = np.random.default_rng(int(seed))
    G = rng.standard_normal((n, ell - 1))

    b_hat = b / np.linalg.norm(b)

    Q = np.empty((n, ell - 1), dtype=float)

    for j in range(ell - 1):

        w = G[:, j].copy()

        w -= b_hat * np.dot(b_hat, w)

        for k in range(j):

            w -= Q[:, k] * np.dot(Q[:, k], w)

        nw = np.linalg.norm(w)

        if nw <= np.finfo(float).eps:
            raise ValueError(
                "Randomized enrichment produced a dependent direction."
            )

        Q[:, j] = w / nw

    return np.column_stack((b, Q))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    return [

        {
            "setup": """
import numpy as np

b = np.arange(1.0, 6.0)

ell = 3

seed = 552017

""",

            "call": """
build_augmented_block(b, ell, seed)
""",

            "gold_call": """
_oracle_build_augmented_block(b, ell, seed)
""",

        },

        {
            "setup": """
import numpy as np

b = np.array([2.0, 3.0])

ell = 2

seed = 7

""",

            "call": """
build_augmented_block(b, ell, seed)
""",

            "gold_call": """
_oracle_build_augmented_block(b, ell, seed)
""",

        },

        {
            "setup": """
import numpy as np

b = np.array([1.0, -2.0, 3.0, 4.0])

ell = 4

seed = 99

""",

            "call": """
build_augmented_block(b, ell, seed)
""",

            "gold_call": """
_oracle_build_augmented_block(b, ell, seed)
""",

        },

        {
            "setup": """
import numpy as np

b = np.ones(4)

ell = 4

seed = 1

""",

            "call": """
build_augmented_block(b, ell, seed)
""",

            "gold_call": """
_oracle_build_augmented_block(b, ell, seed)
""",

        },

    ]
