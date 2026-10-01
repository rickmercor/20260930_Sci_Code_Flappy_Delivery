"""
Forward and backward electron-transfer rate constants from a donor-population trace by least-squares fitting to reversible first-order two-state kinetics.

Effective transfer rates are extracted from the computed dynamics by fitting it to the kinetic scheme D <-> A with forward rate k_DA and backward rate k_AD, solved for p_D(0) = 1. The fit minimises the unweighted sum of squared differences between the kinetic solution and the supplied donor populations over all sample times, with both rates constrained to be non-negative.

The coherent oscillations of the computed trace are not part of the kinetic model, so the residual is not zero; return the rates of the global minimum of the sum of squares.

Returns
-------
numpy.ndarray [k_DA, k_AD] in ps^-1: the least-squares forward and backward rate constants
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_transfer_rates(times_ps: "np.ndarray", p_donor: "np.ndarray") -> "np.ndarray":
    '''Least-squares rate constants of reversible two-state kinetics.

    Parameters
    ----------
    times_ps : numpy.ndarray
        1-D array of sample times in ps, starting at 0 and strictly increasing.
    p_donor : numpy.ndarray
        1-D array of donor populations at those times, same length (>= 3).

    Returns
    -------
    rates : numpy.ndarray
        Array [k_DA, k_AD] in ps^-1 minimising the sum of squared residuals.

    Raises
    ------
    ValueError
        If the arrays are not one dimensional of equal length at least 3, contain
        non-finite values, or the times do not start at 0 and increase strictly.
    '''
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fit_transfer_rates(times_ps: "np.ndarray", p_donor: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.optimize import minimize_scalar
    t = np.asarray(times_ps, dtype=float)
    p = np.asarray(p_donor, dtype=float)
    if t.ndim != 1 or t.shape != p.shape or t.size < 3:
        raise ValueError("times and populations must be 1-D arrays of equal length >= 3")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
        raise ValueError("inputs must be finite")
    if abs(t[0]) > 1e-12 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and increase strictly")

    def _best(logk):
        e = np.exp(-np.exp(logk) * t)
        u = 1.0 - e
        den = u @ u
        pinf = 0.0 if den == 0.0 else float(np.clip(((p - e) @ u) / den, 0.0, 1.0))
        r = e + pinf * u - p
        return r @ r, pinf

    grid = np.linspace(np.log(1e-3 / t[-1]), np.log(1e3 / t[1]), 2001)
    vals = np.array([_best(g)[0] for g in grid])
    j = int(np.argmin(vals))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
    res = minimize_scalar(lambda g: _best(g)[0], bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-13, 'maxiter': 500})
    logk = res.x if res.fun <= vals[j] else grid[j]
    k = float(np.exp(logk))
    pinf = _best(logk)[1]
    return np.array([k * (1.0 - pinf), k * pinf])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: noise-free reversible kinetics ---
        {
            "setup": """import numpy as np
t = 0.05 * np.arange(61)
p = 0.2 + 0.8 * np.exp(-2.5 * t)
""",
            "call": "fit_transfer_rates(t.copy(), p.copy())",
            "gold_call": "_oracle_fit_transfer_rates(t.copy(), p.copy())",
            "tol": 0.001,
        },
        # --- Normal: a trace with a damped coherent oscillation on top of the relaxation ---
        {
            "setup": """import numpy as np
t = 0.01 * np.arange(501)
p = 0.3 + 0.7 * np.exp(-1.9 * t) + 0.05 * np.exp(-3.0 * t) * np.cos(12.0 * t)
""",
            "call": "fit_transfer_rates(t.copy(), p.copy())",
            "gold_call": "_oracle_fit_transfer_rates(t.copy(), p.copy())",
            "tol": 0.001,
        },
        # --- Boundary: irreversible decay, the backward rate sits on its bound ---
        {
            "setup": """import numpy as np
t = 0.02 * np.arange(151)
p = np.exp(-1.3 * t)
""",
            "call": "fit_transfer_rates(t.copy(), p.copy())",
            "gold_call": "_oracle_fit_transfer_rates(t.copy(), p.copy())",
            "tol": 0.001,
        },
        # --- Boundary: a relaxation that undershoots zero, so the unconstrained best fit would need a negative backward rate ---
        {
            "setup": """import numpy as np
t = 0.02 * np.arange(151)
p = 1.05 * np.exp(-1.5 * t) - 0.05
""",
            "call": "fit_transfer_rates(t.copy(), p.copy())",
            "gold_call": "_oracle_fit_transfer_rates(t.copy(), p.copy())",
            "tol": 0.001,
        },
        # --- Edge: three samples, the minimum number ---
        {
            "setup": """import numpy as np
t = np.array([0.0, 0.4, 1.1])
p = np.array([1.0, 0.62, 0.41])
""",
            "call": "fit_transfer_rates(t.copy(), p.copy())",
            "gold_call": "_oracle_fit_transfer_rates(t.copy(), p.copy())",
            "tol": 0.001,
        },
        # --- Invalid: times that do not start at zero ---
        {
            "setup": """import numpy as np
t = np.array([0.1, 0.2, 0.3, 0.4])
p = np.array([1.0, 0.8, 0.7, 0.65])
def run_model():
    try:
        fit_transfer_rates(t.copy(), p.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_fit_transfer_rates(t.copy(), p.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
