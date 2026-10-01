"""
Select the minimal probability-ranked support reaching a mass target.

Deterministic configuration selection ranks the complete determinant archive by normalized probability and retains the smallest leading subset whose cumulative mass reaches a prescribed target. Lower and upper size bounds constrain the selected support. Exact probability ties are resolved by the original input order to make the selection reproducible.

Returns
-------
A one-dimensional integer array containing the selected zero-based indices in decreasing-probability order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_cumulative_support(
    probabilities: 'np.ndarray',
    fraction: float,
    min_k: int,
    max_k: int,
) -> 'np.ndarray':
    """Apply cumulative-mass selection with deterministic input-order ties.

    Returns
    -------
    np.ndarray
        Selected zero-based indices in decreasing-probability order.

    Raises
    ------
    ValueError
        If probabilities or selector bounds are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_select_cumulative_support(
    probabilities: 'np.ndarray',
    fraction: float,
    min_k: int,
    max_k: int,
) -> 'np.ndarray':

    p = np.asarray(probabilities, dtype=float)

    if (
        p.ndim != 1
        or p.size == 0
        or np.any(~np.isfinite(p))
        or np.any(p < 0.0)
    ):
        raise ValueError(
            "probabilities must be a finite nonnegative vector"
        )

    total = float(np.sum(p))
    if total <= 0.0:
        raise ValueError("probability mass must be positive")

    if (
        not np.isfinite(fraction)
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError("fraction must be in (0,1]")

    if not isinstance(
        min_k,
        (int, np.integer),
    ) or not isinstance(
        max_k,
        (int, np.integer),
    ):
        raise ValueError("bounds must be integers")

    min_k = int(min_k)
    max_k = int(max_k)

    if not 1 <= min_k <= max_k:
        raise ValueError("invalid size bounds")

    p = p / total
    index = np.arange(p.size)
    order = np.lexsort((index, -p))

    cumulative = np.cumsum(p[order])
    k = (
        int(
            np.searchsorted(
                cumulative,
                float(fraction),
                side="left",
            )
        )
        + 1
    )
    k = min(
        max(k, min_k),
        max_k,
        p.size,
    )

    return order[:k].astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\np=np.array([.08,.41,.12,.24,.15])",
            "call": "select_cumulative_support(p.copy(),.70,1,5)",
            "gold_call": "_oracle_select_cumulative_support(p,.70,1,5)",
        },
        {
            "setup": "import numpy as np\np=np.array([.5,.3,.2])",
            "call": "select_cumulative_support(p.copy(),.8,1,3)",
            "gold_call": "_oracle_select_cumulative_support(p,.8,1,3)",
        },
        {
            "setup": "import numpy as np\np=np.array([.7,.1,.1,.1])",
            "call": "select_cumulative_support(p.copy(),.5,3,4)",
            "gold_call": "_oracle_select_cumulative_support(p,.5,3,4)",
        },
        {
            "setup": "import numpy as np\np=np.array([2.,2.,2.,2.,1.])",
            "call": "select_cumulative_support(p.copy(),.95,1,3)",
            "gold_call": "_oracle_select_cumulative_support(p,.95,1,3)",
        },
        {
            "setup": "import numpy as np\np=np.array([.5,.3,.2])\ndef check(fn):\n try: fn(p.copy(),0.,1,3)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(select_cumulative_support)",
            "gold_call": "check(_oracle_select_cumulative_support)",
        },
    ]
