"""
Solve the zero-order gain balance for the stable oscillation amplitude at a given bias and load.

At zero order the oscillation amplitude is fixed by gain balance. Over one cycle the diode has to deliver exactly the fundamental current that the load conductance G_l dissipates, which with the coefficient c_1 of step 03 reads

    G_l a0 dV + c_1(a0) = 0.

The amplitude appears both explicitly and inside c_1, so the condition is solved numerically for 0 < a0 <= 3. Near the edges of the oscillation region it can have two roots: the smaller is the unstable threshold between the resting and the oscillating states, and the stable oscillation is the largest root. Where it has no root the bias supports no oscillation at that load. G_l is a conductance density in mS/um^2.

Returns
-------
float, the largest zero-order amplitude in 0 < a0 <= 3 (units of dV) satisfying the gain balance, or 0.0 if none
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def oscillation_amplitude(Vdc: float, Gl: float, dV: float, iv_params: np.ndarray) -> float:
    '''Stable zero-order oscillation amplitude at a given bias and load.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    Gl : float
        Load conductance in mS/um^2.
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    a0 : float
        Largest amplitude in 0 < a0 <= 3, in units of dV, that satisfies the
        gain balance, or 0.0 if there is none.

    Raises
    ------
    ValueError
        If Gl < 0 or dV <= 0, or if iv_params is invalid as in step 01.
    '''
    return a0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_oscillation_amplitude(Vdc: float, Gl: float, dV: float, iv_params: np.ndarray) -> float:
    """Reference implementation. Chains step 03."""
    if Gl < 0.0 or dV <= 0.0:
        raise ValueError("need Gl >= 0 and dV > 0")

    def _balance(a):
        return Gl + _oracle_harmonic_coefficients(Vdc, a, dV, iv_params, 1)[1]/(a*dV)

    xs = np.linspace(1e-3, 3.0, 1500)
    vs = np.array([_balance(x) for x in xs])
    roots = [brentq(_balance, xs[k], xs[k + 1], xtol=1e-14, rtol=1e-15)
             for k in range(xs.size - 1) if vs[k]*vs[k + 1] < 0.0]
    return float(max(roots)) if roots else 0.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "oscillation_amplitude(0.44, 19.0, 0.32, p)",
            "gold_call": "_oracle_oscillation_amplitude(0.44, 19.0, 0.32, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "oscillation_amplitude(0.64, 19.0, 0.32, p)",
            "gold_call": "_oracle_oscillation_amplitude(0.64, 19.0, 0.32, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "oscillation_amplitude(0.74, 19.0, 0.32, p)",
            "gold_call": "_oracle_oscillation_amplitude(0.74, 19.0, 0.32, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.09, 0.0, 1.10, 0.10, 15.0])",
            "call": "oscillation_amplitude(0.510174, 66.0, 0.188961, p)",
            "gold_call": "_oracle_oscillation_amplitude(0.510174, 66.0, 0.188961, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([18.0, 0.35, 0.18, 26.0, 1.05, 0.12, 6.0])",
            "call": "oscillation_amplitude(1.318271, 14.7, 0.346976, p)",
            "gold_call": "_oracle_oscillation_amplitude(1.318271, 14.7, 0.346976, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([6.2, 0.9, 0.3, 0.0, 1.60, 0.10, 0.8])",
            "call": "oscillation_amplitude(1.372103, 2.66, 0.476407, p)",
            "gold_call": "_oracle_oscillation_amplitude(1.372103, 2.66, 0.476407, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([20.0, 0.30, 0.07, 12.0, 0.95, 0.08, 9.0])",
            "call": "oscillation_amplitude(1.046625, 48.0, 0.162939, p)",
            "gold_call": "_oracle_oscillation_amplitude(1.046625, 48.0, 0.162939, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\ndef run_model():\n    try:\n        oscillation_amplitude(0.44, -1.0, 0.32, p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_oscillation_amplitude(0.44, -1.0, 0.32, p); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
