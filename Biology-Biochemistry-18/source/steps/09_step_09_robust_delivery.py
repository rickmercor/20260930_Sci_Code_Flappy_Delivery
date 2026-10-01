"""
Apply the calibration limit and choose the largest worst-condition delivery fraction.

Eligibility is a simultaneous physiological property. Robust delivery is the minimum condition score; a more productive design can be ineligible.

Returns
-------
A finite ndarray of length three: score, candidate index, condition index.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def robust_delivery(net: "np.ndarray", calibration: "np.ndarray", route: int,
                    demand: float, limit: float, tie: float) -> "np.ndarray":
    """
    net is finite (C,P,N); calibration is length C, nonnegative with
    positive infinity allowed for infeasibility. route is a zero-based
    reaction index; demand is a positive denominator. limit and tie are
    finite and nonnegative. Eligible means calibration<=limit. The
    score is min_condition(net[route]/demand). Return length three:
    selected score, zero-based candidate index, zero-based limiting
    condition index, all numerical. Candidate scores within tie of the
    maximum are tied; choose the lowest candidate index. Limiting
    condition scores within tie of that candidate minimum are tied;
    choose the lowest condition index. Invalid values/alignment or no
    eligible design raise ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_robust_delivery(net: "np.ndarray", calibration: "np.ndarray", route: int,
                    demand: float, limit: float, tie: float) -> "np.ndarray":
    import numpy as np
    n, delta = np.asarray(net, dtype=float), np.asarray(calibration, dtype=float)
    if (n.ndim != 3 or min(n.shape) == 0 or delta.shape != (n.shape[0],)
            or not np.isfinite(n).all() or np.any(np.isnan(delta)) or np.any(delta < 0)
            or int(route) != route or not 0 <= route < n.shape[2]
            or not np.isfinite(demand) or demand <= 0 or not np.isfinite(limit)
            or limit < 0 or not np.isfinite(tie) or tie < 0):
        raise ValueError('Aligned performance data and valid selection settings required')
    scores = np.min(n[:, :, int(route)]/demand, axis=1)
    eligible = np.flatnonzero(delta <= limit)
    if eligible.size == 0:
        raise ValueError('No eligible design')
    best = np.max(scores[eligible])
    c = int(eligible[np.flatnonzero(best-scores[eligible] <= tie)[0]])
    condition = int(np.flatnonzero(n[c, :, int(route)]/demand-scores[c] <= tie)[0])
    return np.array([scores[c], c, condition], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'n=np.array([[[5.],[4.]],[[6.],[5.]],[[4.2],[4.2]]]); '
               'd=np.array([2.,4.,1.])\n',
      'call': 'robust_delivery(n,d,0,8.,3.,1e-10)',
      'gold_call': '_oracle_robust_delivery(n,d,0,8.,3.,1e-10)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'n=np.array([[[5.],[4.]],[[6.],[5.]],[[4.2],[4.2]]]); '
               'd=np.array([2.,4.,1.])\n'
               'd=np.array([2.,2.,2.]); n[1,1,0]=4.; n[2,:,:]=4.',
      'call': 'robust_delivery(n,d,0,8.,2.,1e-10)',
      'gold_call': '_oracle_robust_delivery(n,d,0,8.,2.,1e-10)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'n=np.array([[[5.],[4.]],[[6.],[5.]],[[4.2],[4.2]]]); '
               'd=np.array([2.,4.,1.])\n'
               'd=np.array([np.inf,3.,np.inf])',
      'call': 'robust_delivery(n,d,0,8.,3.,1e-10)',
      'gold_call': '_oracle_robust_delivery(n,d,0,8.,3.,1e-10)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'n=np.array([[[5.],[4.]],[[6.],[5.]],[[4.2],[4.2]]]); '
               'd=np.array([2.,4.,1.])\n'
               'd[:]=4.\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(robust_delivery, (n,d,0,8.,3.,1e-10,))',
      'gold_call': '_raises_value_error(_oracle_robust_delivery, (n,d,0,8.,3.,1e-10,))',
      'tol': 0.0}]
