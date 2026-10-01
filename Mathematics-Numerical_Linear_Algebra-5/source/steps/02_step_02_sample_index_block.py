"""
Draws a block of distinct indices without replacement from a probability vector, using sequential inverse-cumulative-distribution-function sampling from a supplied random number generator, and is the shared sampling primitive used for both the column block and the row block of every iteration.

Deterministic, version-proof without-replacement sampling.

Randomized block Kaczmarz methods need a way to draw a fixed-size block of distinct indices from a probability vector every iteration, and the exact algorithm used to do that draw matters: any two algorithms that both "sample without replacement proportional to weight" can still produce a different index sequence from the same generator, which silently changes every downstream number. This step pins one specific, version-proof algorithm: for each of the requested draws in turn, take a fresh uniform draw from the supplied generator, form the cumulative sums of the *current* weight vector (the one with previously drawn indices already zeroed out), and select the smallest index whose cumulative sum is at least the uniform draw times the current vector's own total weight; that selected index's weight is then set to zero before the next draw, so it can never be drawn again within the same call. This is used instead of a generator's own built-in weighted-sampling-without-replacement routine because a numeric library may guarantee bit-for-bit reproducibility for its basic uniform-draw primitive across versions without extending that same guarantee to more complex sampling routines, and a benchmark whose pinned answer depends on the exact draw sequence cannot risk silently changing behavior on a different installed version of the library. The same routine is used for both the column-index block and the row-index block of every iteration, since both draws are the identical sampling primitive applied to a different probability vector and a different block size.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sample_index_block(probabilities: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a block of distinct indices by sequential inverse-CDF sampling.

    Parameters
    ----------
    probabilities : numpy.ndarray
        (n,) nonnegative weight (or probability) vector to sample from.
    block_size : int
        Number of distinct indices to draw (must be a positive integer
        no larger than the number of strictly positive entries of
        probabilities).
    rng : numpy.random.Generator
        A live NumPy random Generator object (e.g. from
        numpy.random.default_rng), consumed one `rng.random()` call per
        drawn index, in order. This must be the SAME generator object
        threaded across every call within a march, never a freshly
        seeded generator per call.

    Returns
    -------
    indices : numpy.ndarray
        (block_size,) array of distinct integer indices into
        probabilities, in the order they were drawn.

    Raises
    ------
    ValueError
        If probabilities is not 1-D, if any entry is negative, if
        block_size is not a positive integer, or if fewer than
        block_size entries of probabilities are strictly positive.
    """
    return indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sample_index_block(probabilities: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of sample_index_block."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be 1-D")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be nonnegative")
    if not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    n_positive = int(np.sum(probabilities > 0.0))
    if n_positive < block_size:
        raise ValueError(
            f"fewer than block_size={block_size} entries have strictly "
            f"positive weight (have {n_positive})"
        )

    w = probabilities.copy()
    chosen = np.empty(block_size, dtype=np.int64)
    for t in range(block_size):
        total = w.sum()
        u = rng.random()
        cum = np.cumsum(w)
        idx = int(np.searchsorted(cum, u * total, side="left"))
        idx = min(idx, w.shape[0] - 1)
        chosen[t] = idx
        w[idx] = 0.0
    return chosen

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: a normal draw (sum of drawn indices, sensitive to draw
    order and to which indices are chosen), a boundary case where
    block_size equals the number of strictly positive entries (forces
    every positive-weight index to be drawn exactly once), and edge cases
    (block_size larger than the number of positive entries, a negative
    weight).
    """
    return [
        {
            # normal: draw 2 of 5 indices with a fixed-seed generator; sum
            # of the drawn indices is sensitive both to which indices are
            # picked and to the draw order (since later draws see a
            # different current total than earlier ones).
            "setup": "import numpy as np",
            "call": (
                "float(np.sum(sample_index_block("
                "np.array([0.1, 0.4, 0.05, 0.3, 0.15]), 2, np.random.default_rng(11))))"
            ),
            "gold_call": (
                "float(np.sum(_oracle_sample_index_block("
                "np.array([0.1, 0.4, 0.05, 0.3, 0.15]), 2, np.random.default_rng(11))))"
            ),
        },
        {
            # boundary: block_size equals the number of strictly positive
            # entries -- every positive index must be drawn exactly once,
            # regardless of draw order, so the sum is forced.
            "setup": "import numpy as np",
            "call": (
                "float(np.sum(sample_index_block("
                "np.array([0.5, 0.0, 0.5, 0.0]), 2, np.random.default_rng(3))))"
            ),
            "gold_call": (
                "float(np.sum(_oracle_sample_index_block("
                "np.array([0.5, 0.0, 0.5, 0.0]), 2, np.random.default_rng(3))))"
            ),
        },
        {
            # edge: block_size exceeds the number of strictly positive
            # entries
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: sample_index_block(np.array([0.5, 0.0, 0.5, 0.0]), 3, np.random.default_rng(1)))",
            "gold_call": "_guard(lambda: _oracle_sample_index_block(np.array([0.5, 0.0, 0.5, 0.0]), 3, np.random.default_rng(1)))",
        },
        {
            # edge: a negative weight is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: sample_index_block(np.array([0.5, -0.1, 0.6]), 1, np.random.default_rng(1)))",
            "gold_call": "_guard(lambda: _oracle_sample_index_block(np.array([0.5, -0.1, 0.6]), 1, np.random.default_rng(1)))",
        },
    ]
