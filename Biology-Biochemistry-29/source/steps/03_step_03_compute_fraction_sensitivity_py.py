"""
Compute the steady-state fraction F and signed total logarithmic sensitivity G=dF/dlog(k6) at fixed conserved totals and other kinetic rates. Include the induced response of both free concentrations.

The first conservation law gives F=1-BQ/T1, where B=c_X+c_XD+c_XDYp+c_XT. Implicit differentiation gives dQ/dlog(k6)=Xp/(S+betay0/Y²) and G=-BXp/[T1*(S+beta*y0/Y²)]. The derivative is taken with T1, T2, and every rate other than k6 fixed. It is not the partial derivative at fixed equilibrium concentrations.

Returns
-------
np.ndarray of shape (2,), containing finite [F,G].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_fraction_sensitivity(
    coeffs: np.ndarray,
    T1: float,
    T2: float,
) -> np.ndarray:
    """Return [F, G], where G=dF/dlog(k6).

    Compute F=(Xp+Q+XTYp)/T1 and the signed total equilibrium
    logarithmic sensitivity. Hold T1, T2, and every kinetic rate
    other than k6 fixed. Include the induced equilibrium response
    of both Q and Y. coeffs has the Step 1 layout.

    Returns:
        np.ndarray: A finite two-entry array [F, G].

    Raises:
        ValueError: If the inputs are invalid or no strictly
            positive admissible equilibrium exists.
    """
    return np.empty(2, dtype=float)

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

def _oracle_compute_fraction_sensitivity(coeffs, T1, T2):
    import numpy as np

    # Ensure earlier oracle functions can resolve shared names
    # when Studio uses separate module namespaces.
    _context = {
        "np": np,
        "_checked_array": _checked_array,
        "_oracle_solve_positive_equilibrium":
            _oracle_solve_positive_equilibrium
    }
    _oracle_solve_positive_equilibrium.__globals__.update(_context)

    c = _checked_array(coeffs, (7,), "coeffs", True)
    t1 = float(T1)
    state = _oracle_solve_positive_equilibrium(c, t1, T2)

    q, y = state[:2]
    xp = state[7]
    S = 1 + np.sum(c[:5])
    B = np.sum(c[[0, 1, 2, 3]])
    beta, yp = c[5:]
    y0 = t1 + float(T2) - yp

    F = (xp + q + state[6])/t1
    G = -B*xp/(t1*(S + beta*y0/y**2))

    return _checked_array(np.array([F, G]), (2,), "observables")

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
            "description": "Fraction and signed sensitivity at the calibration condition.",
            "setup": common,
            "call": "compute_fraction_sensitivity(coeffs, 1.4, 1.1)",
            "gold_call": "_oracle_compute_fraction_sensitivity(coeffs, 1.4, 1.1)",
            "tol": 1e-9
        },
        {
            "description": "Small beta remains finite and nonzero.",
            "setup": common + """
tiny = coeffs.copy()
tiny[5] = 1.37e-13
""",
            "call": "compute_fraction_sensitivity(tiny, 1.4, 1.1)",
            "gold_call": "_oracle_compute_fraction_sensitivity(tiny, 1.4, 1.1)",
            "tol": 1e-9
        },
        {
            "description": "Implicit derivative agrees with a central logarithmic perturbation.",
            "setup": common + """
def derivative_error(fn):
    h = 1e-5
    cp = coeffs.copy()
    cm = coeffs.copy()
    cp[5] *= np.exp(-h)
    cm[5] *= np.exp(h)
    fd = (
        fn(cp, 1.4, 1.1)[0]
        - fn(cm, 1.4, 1.1)[0]
    )/(2*h)
    return abs(fn(coeffs, 1.4, 1.1)[1] - fd)
""",
            "call": "derivative_error(compute_fraction_sensitivity)",
            "gold_call": "derivative_error(_oracle_compute_fraction_sensitivity)",
            "tol": 1e-9
        },
        {
            "description": "Invalid total is rejected.",
            "setup": common,
            "call": "capture(compute_fraction_sensitivity, coeffs, 0.0, 1.1)",
            "gold_call": "capture(_oracle_compute_fraction_sensitivity, coeffs, 0.0, 1.1)",
            "tol": 1e-9
        }
    ]
