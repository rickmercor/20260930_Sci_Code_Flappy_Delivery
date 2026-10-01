"""
Evaluate the quasi-static dc current density of the resonant-tunnelling diode for a parameterised characteristic.

The oscillator model keeps a single nonlinearity, the quasi-static dc characteristic of the resonant-tunnelling diode. It is written as two Gaussian resonant-tunnelling peaks on a cubic excess-current background,

    I(V) = A1 exp(-((V - V1) / w1)^2) + A2 exp(-((V - V2) / w2)^2) + B V^3,

with iv_params = [A1, V1, w1, A2, V2, w2, B] in mA/um^2, V, V, mA/um^2, V, V and mA/(um^2 V^3). The second resonance sits at a higher bias than the first and can be absent (A2 = 0) or stronger than the first. Current densities are per square micrometre of diode area throughout the task, and every later step evaluates the characteristic through this function.

Returns
-------
numpy.ndarray, the dc current density I(V) in mA/um^2 with the same shape as V
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dc_current(V: np.ndarray, iv_params: np.ndarray) -> np.ndarray:
    '''Quasi-static dc current density of the diode.

    Parameters
    ----------
    V : numpy.ndarray
        Bias values in volts, any shape.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    I : numpy.ndarray
        Current density in mA/um^2, same shape as V.

    Raises
    ------
    ValueError
        If iv_params does not hold exactly seven values or either width w1, w2
        is not positive.
    '''
    return I

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_dc_current(V: np.ndarray, iv_params: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    p = np.asarray(iv_params, dtype=float).ravel()
    if p.size != 7 or not p[2] > 0.0 or not p[5] > 0.0:
        raise ValueError("iv_params must be [A1, V1, w1, A2, V2, w2, B] with w1 > 0 and w2 > 0")
    A1, V1, w1, A2, V2, w2, B = p
    V = np.asarray(V, dtype=float)
    return A1*np.exp(-((V - V1)/w1)**2) + A2*np.exp(-((V - V2)/w2)**2) + B*V**3

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nV = np.linspace(0.0, 0.9, 10)\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "dc_current(V, p)",
            "gold_call": "_oracle_dc_current(V, p)",
        },
        {
            "setup": "import numpy as np\nV = np.array([-0.25, 0.0, 0.35, 1.2])\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "dc_current(V, p)",
            "gold_call": "_oracle_dc_current(V, p)",
        },
        {
            "setup": "import numpy as np\nV = np.linspace(0.2, 1.4, 7)\np = np.array([18.0, 0.35, 0.18, 26.0, 1.05, 0.12, 6.0])",
            "call": "dc_current(V, p)",
            "gold_call": "_oracle_dc_current(V, p)",
        },
        {
            "setup": "import numpy as np\nV = np.array([0.4])\np = np.array([24.0, 0.35, 0.0, 15.0])\ndef run_model():\n    try:\n        dc_current(V, p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_dc_current(V, p); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
