"""
Measure flooding-estimate sensitivity to removing each selected set.

Deletion refits reveal whether the common rate is supported by the selected
ladder or dominated by a single bias condition.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def leave_one_set_out_log_rates(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
) -> 'np.ndarray':
    """Return refitted unbiased log-rates after each one-set deletion.

    Preserve the order of ``indices`` in the output.  Each deletion must leave
    at least three sets and is refitted independently with the bounded common
    efficiency calculation.

    Raises ``ValueError`` unless at least four distinct valid sets are given.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_leave_one_set_out_log_rates(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
) -> 'np.ndarray':
    try:
        chosen = np.asarray(indices)
        bias = np.asarray(restored_bias)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("leave-one-out inputs must be arrays") from exc
    if (
        bias.ndim != 3 or chosen.ndim != 1 or chosen.size < 4
        or not np.issubdtype(chosen.dtype, np.integer)
        or np.any(chosen < 0) or np.any(chosen >= bias.shape[0])
        or len(np.unique(chosen)) != len(chosen)
    ):
        raise ValueError("at least four distinct valid set indices are required")
    values = []
    for position in range(chosen.size):
        keep = np.delete(chosen, position)
        values.append(
            _oracle_fit_eatr_flooding(
                log_k_observed, restored_bias, lengths, temperature, keep
            )[1]
        )
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError("leave-one-out refits are not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\nv=np.array([[[1.,1.2],[1.1,1.3]],[[2.,2.2],[2.1,2.3]],[[3.,3.2],[3.1,3.3]],[[4.,4.2],[4.1,4.3]],[[5.,5.2],[5.1,5.3]]]); l=np.full((5,2),2); y=np.array([-1.1,-.8,-.5,-.2,.1])"
    return [
        {"setup": setup, "call": "leave_one_set_out_log_rates(y,v,l,300.,np.arange(5))", "gold_call": "_oracle_leave_one_set_out_log_rates(y,v,l,300.,np.arange(5))", "tol": 1e-10},
        {"setup": setup, "call": "leave_one_set_out_log_rates(y,v,l,300.,np.array([4,2,0,3]))", "gold_call": "_oracle_leave_one_set_out_log_rates(y,v,l,300.,np.array([4,2,0,3]))", "tol": 1e-10},
        {"setup": "import numpy as np\nv=np.zeros((4,1,2)); l=np.full((4,1),2); y=np.zeros(4)", "call": "leave_one_set_out_log_rates(y,v,l,1e6,np.arange(4))", "gold_call": "_oracle_leave_one_set_out_log_rates(y,v,l,1e6,np.arange(4))", "tol": 1e-10},
        {"setup": "import numpy as np\ndef check(fn):\n v=np.zeros((5,1,2)); l=np.full((5,1),2); y=np.zeros(5); out=[]\n for idx in (np.arange(3),np.array([0,1,1,2]),np.array([0,1,2,5])):\n  try: fn(y,v,l,300.,idx)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(leave_one_set_out_log_rates)", "gold_call": "check(_oracle_leave_one_set_out_log_rates)", "tol": 0.0},
    ]
