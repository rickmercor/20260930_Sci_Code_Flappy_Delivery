"""
Locate the peak and valley of the characteristic and return the peak-to-valley separation and the maximum negative differential conductance.

Everything in the oscillator model is measured against two scales taken from the dc characteristic of step 01. A negative-differential-conductance region runs from a current peak, where the slope dI/dV turns negative, to the valley, where it turns positive again. A characteristic with a second resonance has a second such region at higher bias, which can be the deeper one; the oscillator works on the first one, so only the first peak and valley above zero bias are used, searched over 0 < V <= 2 V. Their separation dV = Vv - Vp is the voltage scale: every ac voltage below is measured in units of dV. The steepest negative slope inside that first region fixes the conductance scale G0 = max(-dI/dV) there, a positive number in mS/um^2, which normalises the perturbation parameter and every first-order coefficient. A steeper slope elsewhere on the characteristic does not enter. Both scales are taken from the analytic slope of the characteristic, so that neither is limited by finite-difference noise.

Returns
-------
numpy.ndarray [Vp, Vv, dV, G0]: peak and valley biases and their separation in volts, and the maximum negative differential conductance in mS/um^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def characteristic_scales(iv_params: np.ndarray) -> np.ndarray:
    '''Voltage and conductance scales of the dc characteristic.

    Parameters
    ----------
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    scales : numpy.ndarray
        Array [Vp, Vv, dV, G0]: the peak and valley biases in volts, their
        separation in volts, and the maximum negative differential conductance
        inside that first region in mS/um^2 (a positive number).

    Raises
    ------
    ValueError
        If no current peak followed by a valley exists for 0 < V <= 2 V, or if
        iv_params is invalid as in step 01.
    '''
    return scales

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar

def _oracle_characteristic_scales(iv_params: np.ndarray) -> np.ndarray:
    """Reference implementation. Chains step 01 through the same parameterisation."""
    p = np.asarray(iv_params, dtype=float).ravel()
    if p.size != 7 or not p[2] > 0.0 or not p[5] > 0.0:
        raise ValueError("iv_params must be [A1, V1, w1, A2, V2, w2, B] with w1 > 0 and w2 > 0")
    A1, V1, w1, A2, V2, w2, B = p

    slope = lambda V: (-2.0*(np.asarray(V, dtype=float) - V1)/w1**2
                       * A1*np.exp(-((np.asarray(V, dtype=float) - V1)/w1)**2)
                       - 2.0*(np.asarray(V, dtype=float) - V2)/w2**2
                       * A2*np.exp(-((np.asarray(V, dtype=float) - V2)/w2)**2)
                       + 3.0*B*np.asarray(V, dtype=float)**2)
    Vg = np.linspace(1e-6, 2.0, 400001)
    s = np.sign(slope(Vg))
    down = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    if down.size == 0:
        raise ValueError("the characteristic has no current peak for 0 < V <= 2 V")
    i = down[0]
    up = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    up = up[up >= i]
    if up.size == 0:
        raise ValueError("the characteristic has no valley after its peak for V <= 2 V")
    j = up[0]
    f = lambda v: float(slope(v))
    Vp = brentq(f, Vg[i], Vg[i + 1], xtol=1e-15, rtol=1e-15)
    Vv = brentq(f, Vg[j], Vg[j + 1], xtol=1e-15, rtol=1e-15)
    Vin = np.linspace(Vp, Vv, 20001)
    k = int(np.argmin(slope(Vin)))
    lo, hi = Vin[max(k - 1, 0)], Vin[min(k + 1, Vin.size - 1)]
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
    return np.array([Vp, Vv, Vv - Vp, -float(res.fun)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "characteristic_scales(p)",
            "gold_call": "_oracle_characteristic_scales(p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([20.0, 0.30, 0.15, 14.0, 0.95, 0.10, 8.0])",
            "call": "characteristic_scales(p)",
            "gold_call": "_oracle_characteristic_scales(p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([6.2, 0.9, 0.3, 0.0, 1.60, 0.10, 0.8])",
            "call": "characteristic_scales(p)",
            "gold_call": "_oracle_characteristic_scales(p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([18.0, 0.35, 0.18, 26.0, 1.05, 0.12, 6.0])",
            "call": "characteristic_scales(p)",
            "gold_call": "_oracle_characteristic_scales(p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([20.0, 0.30, 0.07, 12.0, 0.95, 0.08, 9.0])",
            "call": "characteristic_scales(p)",
            "gold_call": "_oracle_characteristic_scales(p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([24.0, 0.35, 0.09, 0.0, 1.10, 0.10, 15.0])",
            "call": "characteristic_scales(p)",
            "gold_call": "_oracle_characteristic_scales(p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\np = np.array([5.0, 0.35, 0.18, 0.0, 1.10, 0.10, 40.0])\ndef run_model():\n    try:\n        characteristic_scales(p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_characteristic_scales(p); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
