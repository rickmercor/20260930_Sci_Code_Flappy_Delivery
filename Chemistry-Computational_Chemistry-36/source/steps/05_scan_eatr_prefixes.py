"""
Scan least-to-most-biased set prefixes for flooding-fit convergence.

Progressive prefixes expose when stronger bias stops improving a common-rate
fit and begins to violate the shared-efficiency regime.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scan_eatr_prefixes(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    order: 'np.ndarray',
    min_sets: int = 3,
) -> 'np.ndarray':
    """Return rows ``[n_sets,gamma,mean_log_k0,variance]`` for prefixes.

    ``order`` must be a permutation of all set indices already sorted from
    the smallest to largest full-efficiency log acceleration.  Fit every
    prefix from ``min_sets`` through the complete order.

    Raises ``ValueError`` for an invalid permutation or prefix control.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_scan_eatr_prefixes(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    order: 'np.ndarray',
    min_sets: int = 3,
) -> 'np.ndarray':
    try:
        bias = np.asarray(restored_bias)
        permutation = np.asarray(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("prefix inputs must be arrays") from exc
    if bias.ndim != 3 or permutation.shape != (bias.shape[0],):
        raise ValueError("order must cover every set")
    if not np.issubdtype(permutation.dtype, np.integer):
        raise ValueError("order must contain integer indices")
    if sorted(permutation.tolist()) != list(range(bias.shape[0])):
        raise ValueError("order must be a permutation of the set indices")
    if not isinstance(min_sets, (int, np.integer)) or isinstance(min_sets, (bool, np.bool_)):
        raise ValueError("min_sets must be an integer")
    if not 3 <= int(min_sets) <= bias.shape[0]:
        raise ValueError("min_sets must lie between three and the set count")
    rows = []
    for count in range(int(min_sets), bias.shape[0] + 1):
        fitted = _oracle_fit_eatr_flooding(
            log_k_observed, bias, lengths, temperature, permutation[:count]
        )
        rows.append([float(count), fitted[0], fitted[1], fitted[2]])
    result = np.asarray(rows, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError("prefix scan produced a nonfinite result")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nv=np.array([[[1.,1.2],[1.1,1.3]],[[2.,2.2],[2.1,2.3]],[[3.,3.2],[3.1,3.3]],[[4.,4.2],[4.1,4.3]],[[5.,5.2],[5.1,5.3]]]); l=np.full((5,2),2); y=np.array([-1.1,-.8,-.5,-.2,.1])"
    return [
        {"setup": setup, "call": "scan_eatr_prefixes(y,v,l,300.,np.arange(5),3)", "gold_call": "_oracle_scan_eatr_prefixes(y,v,l,300.,np.arange(5),3)", "tol": 1e-10},
        {"setup": setup, "call": "scan_eatr_prefixes(y,v,l,300.,np.array([4,3,2,1,0]),4)", "gold_call": "_oracle_scan_eatr_prefixes(y,v,l,300.,np.array([4,3,2,1,0]),4)", "tol": 1e-10},
        {"setup": "import numpy as np\nv=np.zeros((3,1,2)); l=np.full((3,1),2); y=np.zeros(3)", "call": "scan_eatr_prefixes(y,v,l,1e5,np.array([1,0,2]),3)", "gold_call": "_oracle_scan_eatr_prefixes(y,v,l,1e5,np.array([1,0,2]),3)", "tol": 1e-10},
        {"setup": "import numpy as np\ndef check(fn):\n v=np.zeros((4,1,2)); l=np.full((4,1),2); y=np.zeros(4); out=[]\n for order,m in ((np.array([0,1,1,3]),3),(np.arange(4),2),(np.arange(3),3)):\n  try: fn(y,v,l,300.,order,m)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(scan_eatr_prefixes)", "gold_call": "check(_oracle_scan_eatr_prefixes)", "tol": 0.0},
    ]
