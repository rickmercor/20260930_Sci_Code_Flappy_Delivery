"""
For a calibrated equilibrium model, hold all kinetic rates and T1 fixed and maximize |dF/dlog(k6)| over a supplied interval of T2. Consider the admissible interior stationary point and the interval endpoints.

The sensitivity formula and conservation laws permit a one-variable optimization using z=Xp/T1 or an equivalent physical coordinate. Derive the stationary condition and the mapping back to T2, and compare every admissible stationary point with the interval endpoints. A local numerical optimum alone is not a globality certificate.

Returns
-------
np.ndarray of shape (3,), containing finite [T2_star,F_star,max_abs_G].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def optimize_sensitivity(
    coeffs: np.ndarray,
    T1: float,
    T2_lower: float,
    T2_upper: float,
) -> np.ndarray:
    """Return [T2_star,F_star,maximum_absolute_sensitivity].

    coeffs has the Step 1 layout. Hold the kinetic rates and T1
    fixed while maximizing |dF/dlog(k6)| over the supplied closed
    interval of T2. The objective is absolute sensitivity, not
    F or signed G. Consider every admissible interior stationary
    point and both interval endpoints to establish globality.
    Accept an endpoint optimum.

    Returns:
        np.ndarray: Finite [T2_star,F_star,max_abs_G], with
            T2_star inside the supplied interval.

    Raises:
        ValueError: If the coefficients, fixed total, or interval
            are invalid; if the bounds are not finite and strictly
            positive with lower<upper; or if no admissible
            equilibrium exists in the interval.
    """
    return np.empty(3, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def _oracle_optimize_sensitivity(
    coeffs, T1, T2_lower, T2_upper
):
    import numpy as np

    _context = {
        "np": np,
        "_checked_array": _checked_array,
        "_oracle_solve_positive_equilibrium":
            _oracle_solve_positive_equilibrium,
        "_oracle_compute_fraction_sensitivity":
            _oracle_compute_fraction_sensitivity
    }
    for _fn in (
        _oracle_solve_positive_equilibrium,
        _oracle_compute_fraction_sensitivity
    ):
        _fn.__globals__.update(_context)

    c = _checked_array(coeffs, (7,), "coeffs", True)
    t1,lo,hi = map(float,(T1,T2_lower,T2_upper))

    if (
        not np.all(np.isfinite([t1,lo,hi]))
        or t1 <= 0 or lo <= 0 or hi <= lo
    ):
        raise ValueError("invalid optimization interval")

    S = 1+np.sum(c[:5])
    B = np.sum(c[[0,1,2,3]])
    C = 1+c[2]+c[4]
    beta,yp = c[5:]

    s,v,r = S/B,C/B,beta/B
    a = v*t1/r
    z = 1/(1+np.sqrt(1+a))
    x = (1-z)/s

    q = t1*x/B
    y = beta*q/(t1*z)
    t2star = y+C*q+yp-t1

    candidates = [lo,hi]
    if lo < t2star < hi:
        candidates.append(t2star)

    results = []
    for t2 in candidates:
        try:
            F,G = _oracle_compute_fraction_sensitivity(c,t1,t2)
            results.append((abs(G),t2,F))
        except ValueError:
            continue

    if not results:
        raise ValueError("no admissible equilibrium in interval")

    value,t2,F = max(results,key=lambda item:(item[0],-item[1]))
    return np.array([t2,F,value])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np

def capture(fn, *args):
    try:
        fn(*args)
        return 0.0
    except ValueError:
        return 1.0
    except Exception:
        return 2.0

coeffs = np.array([
    0.9489489489489491,
    0.4633103691927221,
    0.2752191290298749,
    0.7207207207207207,
    0.3358980541217181,
    0.5259259259259259,
    0.8715294886755629
], dtype=float)
"""

    return [
        {
            "description": "Interior sensitivity maximum for an independent synthetic model.",
            "setup": common,
            "call": "optimize_sensitivity(coeffs,1.35,0.2,2.0)",
            "gold_call": "_oracle_optimize_sensitivity(coeffs,1.35,0.2,2.0)",
            "tol": 1e-9
        },
        {
            "description": "Restricted interval has its maximum at the lower endpoint.",
            "setup": common,
            "call": "optimize_sensitivity(coeffs,1.35,0.8,1.2)",
            "gold_call": "_oracle_optimize_sensitivity(coeffs,1.35,0.8,1.2)",
            "tol": 1e-9
        },
        {
            "description": "Reversed optimization interval is rejected.",
            "setup": common,
            "call": "capture(optimize_sensitivity,coeffs,1.35,1.2,0.8)",
            "gold_call": "capture(_oracle_optimize_sensitivity,coeffs,1.35,1.2,0.8)",
            "tol": 1e-9
        }
    ]
