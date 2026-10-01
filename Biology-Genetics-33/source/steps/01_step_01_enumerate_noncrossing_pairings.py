"""
Enumerate every non-crossing pair partition of an even number of ordered points and return each of them as the involution that sends a point to its partner.

The asymptotic moments of a Gaussian quadratic form are indexed by the pairings of the points that survive the large-dimension limit. Enumerating them is the combinatorial skeleton on which the rest of the moment calculation is hung.

Returns
-------
np.ndarray of shape (M, 2 * order), integer: one partner map per non-crossing pair partition.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def enumerate_noncrossing_pairings(order: int) -> np.ndarray:
    """Enumerate the non-crossing pair partitions of ``2 * order`` ordered points.

    Parameters
    ----------
    order : int
        Half the number of points to be paired; a positive integer. The points
        are labelled ``0, 1, ..., 2 * order - 1`` in cyclic order.

    Returns
    -------
    pairings : np.ndarray
        Integer array of shape ``(M, 2 * order)``. Row ``m`` is the involution
        of the ``m``-th non-crossing pair partition, so entry ``(m, v)`` is the
        point that ``v`` is paired with. Rows are ordered lexicographically by
        that involution.

    Raises
    ------
    ValueError
        If ``order`` is not an integer value, or if it is smaller than one.
    """
    return pairings  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_enumerate_noncrossing_pairings(order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    if not _is_number(order) or float(order) != float(int(order)):
        raise ValueError("order must be an integer value")
    order = int(order)
    if order < 1:
        raise ValueError("order must be at least one")

    def _pairings_on(points):
        # A non-crossing pairing of an ordered list of points is built by pairing
        # the first point with one of the points at odd offset from it; the two
        # arcs it cuts out are then paired independently and never cross it.
        if not points:
            return [[]]
        collected = []
        head = points[0]
        for offset in range(1, len(points), 2):
            partner = points[offset]
            inside = points[1:offset]
            outside = points[offset + 1:]
            for left in _pairings_on(inside):
                for right in _pairings_on(outside):
                    collected.append([(head, partner)] + left + right)
        return collected

    n_points = 2 * order
    rows = []
    for pairs in _pairings_on(list(range(n_points))):
        involution = np.zeros(n_points, dtype=np.int64)
        for first, second in pairs:
            involution[first] = second
            involution[second] = first
        rows.append(involution)

    pairings = np.array(rows, dtype=np.int64)
    # Lexicographic order by the involution makes the enumeration canonical.
    keys = tuple(pairings[:, column] for column in range(n_points - 1, -1, -1))
    return pairings[np.lexsort(keys)]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the order used by the between-family moment block (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
order = 2
""",
            "call": "sig(enumerate_noncrossing_pairings(order), 1.0)",
            "gold_call": "sig(_oracle_enumerate_noncrossing_pairings(order), 1.0)",
        },
        # --- Valid: the third-moment order, where crossing pairings first appear
        #     and must be excluded ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
order = 3
""",
            "call": "sig(enumerate_noncrossing_pairings(order), 1.0)",
            "gold_call": "sig(_oracle_enumerate_noncrossing_pairings(order), 1.0)",
        },
        # --- Boundary: the smallest admissible order, a single pair ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
order = 1
""",
            "call": "sig(enumerate_noncrossing_pairings(order), 1.0)",
            "gold_call": "sig(_oracle_enumerate_noncrossing_pairings(order), 1.0)",
        },
        # --- Edge: a larger order, where the count grows as a Catalan number and
        #     an enumeration that merely filters all pairings would be visible ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
order = 5
""",
            "call": "sig(enumerate_noncrossing_pairings(order), 1.0e3)",
            "gold_call": "sig(_oracle_enumerate_noncrossing_pairings(order), 1.0e3)",
        },
        # --- Edge: the number of pairings alone, which pins the enumeration size
        #     independently of the ordering convention ---
        {
            "setup": """import numpy as np
order = 4
""",
            "call": "int(np.asarray(enumerate_noncrossing_pairings(order)).shape[0])",
            "gold_call": "int(np.asarray(_oracle_enumerate_noncrossing_pairings(order)).shape[0])",
        },
        # --- Invalid: a non-positive order ---
        {
            "setup": """import numpy as np
order = 0
def run_model():
    try:
        enumerate_noncrossing_pairings(order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_enumerate_noncrossing_pairings(order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a fractional order, which does not index a moment ---
        {
            "setup": """import numpy as np
order = 2.5
def run_model():
    try:
        enumerate_noncrossing_pairings(order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_enumerate_noncrossing_pairings(order)
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
