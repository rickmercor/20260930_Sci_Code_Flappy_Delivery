"""
Construct the complete deterministic instance used by the benchmark computation: the sparse symmetric operator, the prescribed three-column initial block and the fixed sparse-sign sketch. The operator is real, symmetric and tridiagonal with entries fixed by the benchmark specification, the initial block contains three deterministic, mutually orthonormal directions, and the sketch is generated from the supplied seed as one continuous random stream. The three objects are returned together so that the next step can consume the complete numerical state directly.

Large-scale electronic-structure calculations need only a small part of the spectrum of a very large Hermitian operator, and iterative eigensolvers work with sparse operators whose action is cheap to evaluate. They start from a low-dimensional search block and, in randomized variants, use a sparse random embedding to represent the geometry of the search space in a much smaller space. A controlled finite instance isolates the numerical behaviour of the eigensolver from application data and lets every stage of the computation be reproduced.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], containing the symmetric operator A, the prescribed 3-column initial search block V0, and the fixed sparse-sign sketch S
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_instance(
    n: int = 96,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> tuple:
    """Construct the operator, the initial block and the sparse-sign sketch.

    Parameters
    ----------
    n : int
        Matrix dimension, with n >= 4 and n divisible by 4.
    sketch_size : int
        Number of rows of the sketch.
    zeta : int
        Number of nonzero entries in each sketch column,
        1 <= zeta <= sketch_size.
    seed : int
        Seed of the single random stream used for the sketch.

    Returns
    -------
    tuple
        (A, V0, S): the n-by-n operator, the n-by-3 initial block and the
        sketch_size-by-n sketch.

    Raises
    ------
    ValueError
        If n < 4, n is not divisible by 4, or any sketch parameter is
        invalid.

    Notes
    -----
    For dimension n the three columns of V0 are the constant,
    alternating-sign and period-four vectors specified by the benchmark,
    each divided by sqrt(n). The sketch is drawn column by column from
    numpy.random.default_rng(seed): for each column the rows are drawn
    with rng.choice(sketch_size, zeta, replace=False) and then the signs
    with 2*rng.integers(0, 2, zeta) - 1, without reseeding. Each selected
    entry is the drawn sign divided by sqrt(zeta) (divided by 2 for the
    benchmark zeta = 4).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_instance(
    n: int = 96,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> tuple:
    """Reference implementation."""
    for name, value in (("n", n), ("sketch_size", sketch_size), ("zeta", zeta), ("seed", seed)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")

    n = int(n)
    s = int(sketch_size)
    z = int(zeta)

    if n < 4 or n % 4 != 0:
        raise ValueError("n must satisfy n >= 4 and be divisible by 4")

    if s < 1 or z < 1 or z > s:
        raise ValueError("sketch parameters must satisfy 1 <= zeta <= sketch_size")

    A = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        idx = i + 1
        A[i, i] = np.log(99.0 + idx)

    for i in range(n - 1):
        idx = i + 1
        value = (
            0.19 * np.sin(0.73 * idx)
            + 0.004 * np.cos(0.41 * idx)
        )
        A[i, i + 1] = value
        A[i + 1, i] = value

    j = np.arange(1, n + 1, dtype=np.float64)
    v1 = np.ones(n, dtype=np.float64) / np.sqrt(n)
    v2 = ((-1.0) ** (j - 1.0)) / np.sqrt(n)
    v3 = np.tile(np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64), n // 4) / np.sqrt(n)
    V0 = np.column_stack((v1, v2, v3))

    rng = np.random.default_rng(int(seed))
    S = np.zeros((s, n), dtype=np.float64)
    scale = 1.0 / np.sqrt(float(z))

    for col in range(n):
        rows = rng.choice(s, z, replace=False)
        signs = 2 * rng.integers(0, 2, size=z) - 1
        S[rows, col] = signs.astype(np.float64) * scale

    return A, V0, S

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test specifications."""
    return [
        # Normal: benchmark instance.
        {
            "setup": "import numpy as np\n",
            "call": "build_instance(96, 45, 4, 2026)",
            "gold_call": "_oracle_build_instance(96, 45, 4, 2026)",
        },

        # Boundary: smallest dimension; every column uses every sketch row.
        {
            "setup": "import numpy as np\n",
            "call": "build_instance(4, 4, 4, 7)",
            "gold_call": "_oracle_build_instance(4, 4, 4, 7)",
        },

        # Edge: one-row sketch with one entry per column.
        {
            "setup": "import numpy as np\n",
            "call": "build_instance(8, 1, 1, 0)",
            "gold_call": "_oracle_build_instance(8, 1, 1, 0)",
        },

        # Invalid: dimension not divisible by four.
        {
            "setup": (
                "import numpy as np\n"
                "def run_model():\n"
                "    try:\n"
                "        build_instance(6, 4, 2, 1)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_build_instance(6, 4, 2, 1)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
