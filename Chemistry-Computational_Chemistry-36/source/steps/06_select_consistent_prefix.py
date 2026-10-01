"""
Select a converged prefix without admitting boundary gamma fits.

A variance threshold and an interior-efficiency check make the benchmark's
prefix decision explicit rather than hiding a visual diagnostic.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_consistent_prefix(
    prefix_table: 'np.ndarray',
    variance_limit: float,
    gamma_margin: float = 0.02,
) -> 'np.ndarray':
    """Return the largest eligible prefix row.

    Rows are ``[n_sets,gamma,mean_log_k0,variance]`` in increasing consecutive
    ``n_sets`` order.  An eligible row has variance no larger than
    ``variance_limit`` and gamma inside the closed interval
    ``[gamma_margin,1-gamma_margin]``.

    Raises ``ValueError`` for malformed diagnostics or no eligible prefix.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_consistent_prefix(
    prefix_table: 'np.ndarray',
    variance_limit: float,
    gamma_margin: float = 0.02,
) -> 'np.ndarray':
    try:
        table = np.asarray(prefix_table)
        limit = float(variance_limit)
        margin = float(gamma_margin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("prefix diagnostics must be numeric") from exc
    if (
        table.ndim != 2 or table.shape[1] != 4 or table.shape[0] == 0
        or not np.issubdtype(table.dtype, np.number) or not np.isrealobj(table)
    ):
        raise ValueError("prefix_table must have shape (n,4)")
    table = table.astype(float, copy=False)
    if (
        np.any(~np.isfinite(table)) or np.any(table[:, 3] < 0.0)
        or not np.isfinite(limit) or limit < 0.0
        or not np.isfinite(margin) or not 0.0 <= margin < 0.5
    ):
        raise ValueError("invalid prefix values or selection controls")
    counts = table[:, 0]
    if np.any(counts != np.rint(counts)) or np.any(np.diff(counts) != 1.0) or counts[0] < 3.0:
        raise ValueError("prefix counts must be consecutive integers from at least three")
    eligible = (
        (table[:, 3] <= limit)
        & (table[:, 1] >= margin)
        & (table[:, 1] <= 1.0 - margin)
    )
    if not np.any(eligible):
        raise ValueError("no prefix satisfies variance and interior-gamma checks")
    return table[np.flatnonzero(eligible)[-1]].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nt=np.array([[3,.7,-2.,.001],[4,.6,-2.1,.002],[5,.5,-2.2,.008]])", "call": "select_consistent_prefix(t,.003)", "gold_call": "_oracle_select_consistent_prefix(t,.003)", "tol": 1e-12},
        {"setup": "import numpy as np\nt=np.array([[3,.01,-2.,.001],[4,.04,-2.1,.002],[5,.98,-2.2,.002]])", "call": "select_consistent_prefix(t,.01,.03)", "gold_call": "_oracle_select_consistent_prefix(t,.01,.03)", "tol": 1e-12},
        {"setup": "import numpy as np\nt=np.array([[3,.5,-2.,0.]])", "call": "select_consistent_prefix(t,0.,0.)", "gold_call": "_oracle_select_consistent_prefix(t,0.,0.)", "tol": 1e-12},
        {"setup": "import numpy as np\ndef check(fn):\n out=[]\n for t,v,m in ((np.array([[3,.0,-2.,.1]]),.01,.02),(np.array([[3,.5,-2.,-1.]]),.1,.02),(np.array([[3,.5,-2.,.1],[5,.5,-2.,.1]]),.1,.02)):\n  try: fn(t,v,m)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(select_consistent_prefix)", "gold_call": "check(_oracle_select_consistent_prefix)", "tol": 0.0},
    ]
