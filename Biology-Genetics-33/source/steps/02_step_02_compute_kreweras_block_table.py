"""
Turn each non-crossing pairing into the block structure of its Kreweras complement, labelling every point by the block of the complement it falls in.

The moment expansion attaches one factor to each block of the Kreweras complement of a pairing rather than one to each of its chords. Recovering those blocks from the pairing is what converts a chord diagram into a product of matrix traces.

Returns
-------
np.ndarray of shape (M, 2 * order), integer: the Kreweras block index of every point of every pairing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_kreweras_block_table(pairings: np.ndarray) -> np.ndarray:
    """Label every point by its block in the Kreweras complement of each pairing.

    Parameters
    ----------
    pairings : np.ndarray
        Integer array of shape ``(M, 2 * order)``. Row ``m`` is the involution
        of a non-crossing pair partition of the points ``0, ..., 2 * order - 1``,
        so entry ``(m, v)`` is the partner of point ``v``.

    Returns
    -------
    block_table : np.ndarray
        Integer array of shape ``(M, 2 * order)``. Entry ``(m, v)`` is the index
        of the block of the Kreweras complement of pairing ``m`` that contains
        point ``v``. Within each row the blocks are numbered ``0, 1, 2, ...`` in
        order of their smallest member.

    Raises
    ------
    ValueError
        If ``pairings`` is not a two-dimensional, non-empty array of integer
        values, if its second dimension is not a positive even number, if any
        entry lies outside the range of point labels, or if any row fails to be
        a fixed-point-free involution.
    """
    return block_table  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_kreweras_block_table(pairings: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    table = np.asarray(pairings)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("pairings must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(np.asarray(table, dtype=float))):
        raise ValueError("pairings must contain only finite entries")
    if not np.allclose(np.asarray(table, dtype=float),
                       np.round(np.asarray(table, dtype=float)), rtol=0.0, atol=1e-12):
        raise ValueError("pairings entries must be integer valued")
    table = np.asarray(np.round(np.asarray(table, dtype=float)), dtype=np.int64)

    n_points = table.shape[1]
    if n_points % 2 != 0:
        raise ValueError("pairings must have an even number of columns")
    if np.any(table < 0) or np.any(table >= n_points):
        raise ValueError("pairings entries must be valid point labels")

    block_table = np.zeros_like(table)
    for row in range(table.shape[0]):
        involution = table[row]
        if np.any(involution[involution] != np.arange(n_points)):
            raise ValueError("each row of pairings must be an involution")
        if np.any(involution == np.arange(n_points)):
            raise ValueError("each row of pairings must be fixed-point free")

        # The Kreweras complement is the permutation obtained by following the
        # cyclic successor and then the pairing; its cycles are the blocks.
        successor = (np.arange(n_points) + 1) % n_points
        complement = involution[successor]

        labels = np.full(n_points, -1, dtype=np.int64)
        next_label = 0
        for start in range(n_points):
            if labels[start] >= 0:
                continue
            point = start
            while labels[point] < 0:
                labels[point] = next_label
                point = int(complement[point])
            next_label += 1
        block_table[row] = labels

    return block_table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: both pairings of four points (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
pairings = np.array([[1, 0, 3, 2], [3, 2, 1, 0]], dtype=np.int64)
""",
            "call": "sig(compute_kreweras_block_table(pairings), 1.0)",
            "gold_call": "sig(_oracle_compute_kreweras_block_table(pairings), 1.0)",
        },
        # --- Valid: the five non-crossing pairings of six points, where blocks of
        #     size three first appear ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
pairings = np.array([[1, 0, 3, 2, 5, 4],
                     [1, 0, 5, 4, 3, 2],
                     [3, 2, 1, 0, 5, 4],
                     [5, 2, 1, 4, 3, 0],
                     [5, 4, 3, 2, 1, 0]], dtype=np.int64)
""",
            "call": "sig(compute_kreweras_block_table(pairings), 1.0)",
            "gold_call": "sig(_oracle_compute_kreweras_block_table(pairings), 1.0)",
        },
        # --- Boundary: the single pairing of two points ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
pairings = np.array([[1, 0]], dtype=np.int64)
""",
            "call": "sig(compute_kreweras_block_table(pairings), 1.0)",
            "gold_call": "sig(_oracle_compute_kreweras_block_table(pairings), 1.0)",
        },
        # --- Edge: the fully nested pairing of eight points, whose complement is a
        #     single large block together with singletons ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
pairings = np.array([[7, 6, 5, 4, 3, 2, 1, 0],
                     [1, 0, 3, 2, 5, 4, 7, 6]], dtype=np.int64)
""",
            "call": "sig(compute_kreweras_block_table(pairings), 1.0)",
            "gold_call": "sig(_oracle_compute_kreweras_block_table(pairings), 1.0)",
        },
        # --- Edge: the number of distinct blocks, which must be the order plus one
        #     for every non-crossing pairing ---
        {
            "setup": """import numpy as np
pairings = np.array([[1, 0, 3, 2, 5, 4],
                     [5, 2, 1, 4, 3, 0]], dtype=np.int64)
""",
            "call": "int(np.asarray(compute_kreweras_block_table(pairings)).max() + 1)",
            "gold_call": "int(np.asarray(_oracle_compute_kreweras_block_table(pairings)).max() + 1)",
        },
        # --- Invalid: a row that is not an involution ---
        {
            "setup": """import numpy as np
pairings = np.array([[1, 2, 3, 0]], dtype=np.int64)
def run_model():
    try:
        compute_kreweras_block_table(pairings)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_kreweras_block_table(pairings)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an odd number of points, which cannot be paired ---
        {
            "setup": """import numpy as np
pairings = np.array([[1, 0, 2]], dtype=np.int64)
def run_model():
    try:
        compute_kreweras_block_table(pairings)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_kreweras_block_table(pairings)
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
