"""
Recover the zero-order oscillation amplitude from the dc current drawn by the oscillating diode.

An oscillating diode does not draw the current given by its dc characteristic. Because the characteristic is curved, the cosine swing rectifies part of the ac current, and at zero order the dc current density drawn at a fixed bias becomes I(Vdc) + c_0(a0), where c_0 is the cycle average of the ac diode current from step 03. A dc current measured on the oscillating diode at a known bias therefore fixes the zero-order amplitude without any reference to the resonator. The amplitude is searched over 0 < a0 <= 3; when several amplitudes reproduce the measured current the smallest is taken, the one reached continuously from the resting state as the swing grows.

Returns
-------
float, the smallest zero-order amplitude in 0 < a0 <= 3 (units of dV) that reproduces the measured dc current
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rectified_amplitude(Vdc: float, J_dc: float, dV: float, iv_params: np.ndarray) -> float:
    '''Zero-order amplitude implied by the dc current of the oscillating diode.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    J_dc : float
        Measured dc current density of the oscillating diode in mA/um^2.
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    a0 : float
        Smallest amplitude in 0 < a0 <= 3, in units of dV, for which
        I(Vdc) + c_0(a0) equals J_dc.

    Raises
    ------
    ValueError
        If no amplitude in 0 < a0 <= 3 reproduces J_dc, if dV <= 0, or if
        iv_params is invalid as in step 01.
    '''
    return a0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_rectified_amplitude(Vdc: float, J_dc: float, dV: float, iv_params: np.ndarray) -> float:
    """Reference implementation. Chains steps 01 and 03."""
    if dV <= 0.0:
        raise ValueError("dV must be positive")
    I0 = float(_oracle_dc_current(np.array(Vdc), iv_params))

    def _mismatch(a):
        return I0 + _oracle_harmonic_coefficients(Vdc, a, dV, iv_params, 1)[0] - J_dc

    xs = np.linspace(1e-3, 3.0, 1500)
    vs = np.array([_mismatch(x) for x in xs])
    hits = [k for k in range(xs.size - 1) if vs[k]*vs[k + 1] <= 0.0]
    if not hits:
        raise ValueError("no amplitude in 0 < a0 <= 3 reproduces the measured dc current")
    k = hits[0]
    return float(brentq(_mismatch, xs[k], xs[k + 1], xtol=1e-14, rtol=1e-15))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "rectified_amplitude(0.44, 12.8, 0.32, p)",
            "gold_call": "_oracle_rectified_amplitude(0.44, 12.8, 0.32, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "rectified_amplitude(0.42, 10.0, 0.32, p)",
            "gold_call": "_oracle_rectified_amplitude(0.42, 10.0, 0.32, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "rectified_amplitude(0.60, 13.0, 0.32, p)",
            "gold_call": "_oracle_rectified_amplitude(0.60, 13.0, 0.32, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "rectified_amplitude(0.417, 10.55895344, 0.32, p)",
            "gold_call": "_oracle_rectified_amplitude(0.417, 10.55895344, 0.32, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.09, 0.0, 1.10, 0.10, 15.0])",
            "call": "rectified_amplitude(0.5021, 7.2618963, 0.188961, p)",
            "gold_call": "_oracle_rectified_amplitude(0.5021, 7.2618963, 0.188961, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([18.0, 0.35, 0.18, 26.0, 1.05, 0.12, 6.0])",
            "call": "rectified_amplitude(0.5255, 9.59577258, 0.346976, p)",
            "gold_call": "_oracle_rectified_amplitude(0.5255, 9.59577258, 0.346976, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([20.0, 0.30, 0.07, 12.0, 0.95, 0.08, 9.0])",
            "call": "rectified_amplitude(0.4306, 4.18150134, 0.162939, p)",
            "gold_call": "_oracle_rectified_amplitude(0.4306, 4.18150134, 0.162939, p)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\ndef run_model():\n    try:\n        rectified_amplitude(0.44, 25.0, 0.32, p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_rectified_amplitude(0.44, 25.0, 0.32, p); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
