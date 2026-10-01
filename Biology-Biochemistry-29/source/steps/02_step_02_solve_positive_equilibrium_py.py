"""
Solve the two conservation laws for the complete nine-species equilibrium. Use the physically admissible root of the reduced equation and reconstruct every species. Validate the original conservation laws and reject nonpositive or nonfinite states.

With S=1+sum(c[:5]), C=1+c_XDYp+c_XTYp, beta=c[5], and y0=T1+T2-Yp, the conservation laws imply Y=y0-CQ and SCQ²-(S y0+beta+T1 C)Q+T1 y0=0. The physical root can be evaluated stably as 2T1 y0/(h+sqrt(h²-4SC T1 y0)), where h=S y0+beta+T1 C. The other algebraic root is not automatically physical. All nine concentrations must be strictly positive.

Returns
-------
np.ndarray of shape (9,), ordered [Q,Y,X,XD,XDYp,XT,XTYp,Xp,Yp], or ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_positive_equilibrium(
    coeffs: np.ndarray,
    T1: float,
    T2: float,
) -> np.ndarray:
    """Return [Q, Y, X, XD, XDYp, XT, XTYp, Xp, Yp].

    coeffs has the seven-entry layout from Step 1.
    T1 and T2 are finite strictly positive scalars. Solve the
    conservation laws and retain only a strictly positive
    equilibrium satisfying the original equations. Do not impose
    a positivity threshold larger than zero.

    Returns:
        np.ndarray: The nine finite, strictly positive
            equilibrium concentrations.

    Raises:
        ValueError: If the coefficients or totals are invalid,
            or if no strictly positive admissible equilibrium exists.
    """
    return np.empty(9, dtype=float)

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

def _oracle_solve_positive_equilibrium(coeffs, T1, T2):
    import numpy as np

    c = _checked_array(coeffs, (7,), "coeffs", True)
    t1, t2 = float(T1), float(T2)

    if not np.all(np.isfinite([t1, t2])) or t1 <= 0 or t2 <= 0:
        raise ValueError("totals must be finite and positive")

    S = 1 + np.sum(c[:5])
    C = 1 + c[2] + c[4]
    beta, yp = c[5:]
    y0 = t1 + t2 - yp

    if y0 <= 0:
        raise ValueError("no positive equilibrium")

    h = S*y0 + beta + t1*C
    disc = h*h - 4*S*C*t1*y0

    if disc < 0:
        raise ValueError("no real equilibrium")

    q = 2*t1*y0/(h + np.sqrt(disc))
    y = y0 - C*q
    xp = beta*q/y

    state = np.array([q, y, *(c[:5]*q), xp, yp])
    return _checked_array(state, (9,), "state", True)

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
            "description": "Normal positive equilibrium.",
            "setup": common,
            "call": "solve_positive_equilibrium(coeffs, 1.35, 0.75)",
            "gold_call": "_oracle_solve_positive_equilibrium(coeffs, 1.35, 0.75)",
            "tol": 1e-9
        },
        {
            "description": "Boundary y0=0 has no strictly positive equilibrium.",
            "setup": common + """
t1 = 0.5
t2 = coeffs[6] - t1
""",
            "call": "capture(solve_positive_equilibrium, coeffs, t1, t2)",
            "gold_call": "capture(_oracle_solve_positive_equilibrium, coeffs, t1, t2)",
            "tol": 1e-9
        },
        {
            "description": "Nonpositive equilibrium coefficient is rejected.",
            "setup": common + """
bad = coeffs.copy()
bad[5] = 0.0
""",
            "call": "capture(solve_positive_equilibrium, bad, 1.35, 0.75)",
            "gold_call": "capture(_oracle_solve_positive_equilibrium, bad, 1.35, 0.75)",
            "tol": 1e-9
        }
    ]
