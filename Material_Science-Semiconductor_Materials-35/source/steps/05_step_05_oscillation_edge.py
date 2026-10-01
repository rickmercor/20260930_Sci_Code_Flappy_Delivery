"""
Find the upper edge of the oscillation region, the turning point at which the gain-balance branch ends, and the amplitude there.

As the bias is raised the stable oscillation does not fade away: the branch of the gain balance that carries it ends abruptly. Writing the balance as a curve of bias against amplitude, the bias at which amplitude a0 satisfies it rises, turns over and falls again, so the branch that is followed by sweeping the bias up ends at the turning point of that curve. Beyond it the oscillation collapses, which is the hysteresis seen at the edges of the oscillation region.

The edge is therefore the largest bias for which the gain balance still has a solution, together with the amplitude there: the point where the balance is satisfied and, at the same time, stops depending on the amplitude to first order, so that the two roots of the balance merge into one. It is a turning point rather than a crossing, so locating it by stepping the bias and watching the roots disappear is limited by the step size; the amplitude at the edge then carries an error of order the square root of that step.

Returns
-------
numpy.ndarray [V_edge, a_edge]: the highest bias in volts at which the gain balance still has a solution and the amplitude there in units of dV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def oscillation_edge(Gl: float, dV: float, iv_params: np.ndarray) -> np.ndarray:
    '''Upper edge of the oscillation region for a given load.

    Parameters
    ----------
    Gl : float
        Load conductance in mS/um^2.
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    edge : numpy.ndarray
        Array [V_edge, a_edge]: the highest bias in volts at which the gain
        balance still has a solution, and the amplitude there in units of dV.

    Raises
    ------
    ValueError
        If Gl <= 0 or dV <= 0, if no bias in 0 < V <= 2 V supports an
        oscillation at this load, or if iv_params is invalid as in step 01.
    '''
    return edge

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar

def _oracle_oscillation_edge(Gl: float, dV: float, iv_params: np.ndarray) -> np.ndarray:
    """Reference implementation. Chains step 03."""
    if Gl <= 0.0 or dV <= 0.0:
        raise ValueError("need Gl > 0 and dV > 0")

    def _bias_for(a):
        """Largest bias at which amplitude a satisfies the gain balance, or nan."""
        res = lambda V: Gl + _oracle_harmonic_coefficients(V, a, dV, iv_params, 1)[1]/(a*dV)
        Vs = np.linspace(0.05, 2.0, 160)
        vals = np.array([res(v) for v in Vs])
        hits = [k for k in range(Vs.size - 1) if vals[k]*vals[k + 1] < 0.0]
        if not hits:
            return np.nan
        k = hits[-1]
        return brentq(res, Vs[k], Vs[k + 1], xtol=1e-15, rtol=1e-15)

    xs = np.linspace(0.02, 2.5, 60)
    vs = np.array([_bias_for(x) for x in xs])
    if not np.any(np.isfinite(vs)):
        raise ValueError("no bias in 0 < V <= 2 V supports an oscillation at this load")
    j = int(np.nanargmax(vs))
    lo = xs[max(j - 1, 0)]
    hi = xs[min(j + 1, xs.size - 1)]
    res = minimize_scalar(lambda a: -_bias_for(a), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    a_edge = float(res.x)
    # one Newton step on the tangency condition dV_bias/da = 0 for full precision
    h = 1e-5
    for _ in range(40):
        f1 = (_bias_for(a_edge + h) - _bias_for(a_edge - h))/(2.0*h)
        f2 = (_bias_for(a_edge + h) - 2.0*_bias_for(a_edge) + _bias_for(a_edge - h))/h**2
        step = f1/f2
        a_edge -= step
        if abs(step) < 1e-13:
            break
    return np.array([_bias_for(a_edge), a_edge])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "oscillation_edge(19.0, 0.32, p)",
            "gold_call": "_oracle_oscillation_edge(19.0, 0.32, p)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "oscillation_edge(30.0, 0.32, p)",
            "gold_call": "_oracle_oscillation_edge(30.0, 0.32, p)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([6.2, 0.9, 0.3, 0.0, 1.60, 0.10, 0.8])",
            "call": "oscillation_edge(3.0, 0.48, p)",
            "gold_call": "_oracle_oscillation_edge(3.0, 0.48, p)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.09, 0.0, 1.10, 0.10, 15.0])",
            "call": "oscillation_edge(55.0, 0.188961, p)",
            "gold_call": "_oracle_oscillation_edge(55.0, 0.188961, p)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([20.0, 0.30, 0.07, 12.0, 0.95, 0.08, 9.0])",
            "call": "oscillation_edge(29.0, 0.162939, p)",
            "gold_call": "_oracle_oscillation_edge(29.0, 0.162939, p)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\ndef run_model():\n    try:\n        oscillation_edge(400.0, 0.32, p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_oscillation_edge(400.0, 0.32, p); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
