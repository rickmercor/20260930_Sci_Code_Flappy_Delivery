"""
Compute the cycle average and cosine Fourier coefficients of the ac diode current under a cosine voltage swing.

When the oscillator runs, the voltage across the diode swings about its bias point. At zero order in the expansion the swing is a pure cosine, v = a0 dV cos(tau), where a0 is the amplitude in units of the peak-to-valley separation dV and tau is the phase of the oscillation. The ac diode current i_D(tau) = I(Vdc + v) - I(Vdc) is then an even, 2*pi-periodic function of tau and is expanded as

    i_D(tau) = c_0 + sum_{n >= 1} c_n cos(n tau).

So c_0 is the cycle average of the ac diode current, the part of it that the curvature of the characteristic rectifies into a dc shift, and c_n for n >= 1 are its cosine Fourier coefficients. These coefficients carry all of the diode physics that reaches the oscillator: c_0 changes the dc current, c_1 fixes the oscillation amplitude, and c_n for n >= 2 produce the higher harmonics and the frequency shift. The integrands are smooth and periodic, so the cycle integrals can be evaluated to near machine precision.

Returns
-------
numpy.ndarray [c_0, c_1, ..., c_nmax] of the ac diode current in mA/um^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def harmonic_coefficients(Vdc: float, a0: float, dV: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    '''Fourier coefficients of the ac diode current under a cosine swing.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    a0 : float
        Swing amplitude in units of dV (a0 = 0 means no swing).
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).
    nmax : int
        Highest harmonic index returned.

    Returns
    -------
    c : numpy.ndarray
        Array [c_0, c_1, ..., c_nmax] in mA/um^2, where c_0 is the cycle
        average of i_D and c_n (n >= 1) its cosine Fourier coefficients.

    Raises
    ------
    ValueError
        If a0 < 0, dV <= 0 or nmax < 1, or if iv_params is invalid as in
        step 01.
    '''
    return c

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_harmonic_coefficients(Vdc: float, a0: float, dV: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    """Reference implementation. Chains step 01."""
    if a0 < 0.0 or dV <= 0.0 or int(nmax) < 1:
        raise ValueError("need a0 >= 0, dV > 0 and nmax >= 1")
    nmax = int(nmax)
    M = 2048
    t = (np.arange(M) + 0.5)*np.pi/M
    f = _oracle_dc_current(Vdc + a0*dV*np.cos(t), iv_params) - _oracle_dc_current(np.array(Vdc), iv_params)
    c = np.empty(nmax + 1)
    c[0] = np.mean(f)
    for n in range(1, nmax + 1):
        c[n] = 2.0*np.mean(f*np.cos(n*t))
    return c

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "harmonic_coefficients(0.44, 0.85, 0.32, p, 12)",
            "gold_call": "_oracle_harmonic_coefficients(0.44, 0.85, 0.32, p, 12)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])",
            "call": "harmonic_coefficients(0.64, 1.07, 0.32, p, 6)",
            "gold_call": "_oracle_harmonic_coefficients(0.64, 1.07, 0.32, p, 6)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([6.2, 0.9, 0.3, 0.0, 1.60, 0.10, 0.8])",
            "call": "harmonic_coefficients(1.1, 0.0, 0.48, p, 3)",
            "gold_call": "_oracle_harmonic_coefficients(1.1, 0.0, 0.48, p, 3)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.09, 0.0, 1.10, 0.10, 15.0])",
            "call": "harmonic_coefficients(0.40, 1.60, 0.188961, p, 12)",
            "gold_call": "_oracle_harmonic_coefficients(0.40, 1.60, 0.188961, p, 12)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([20.0, 0.30, 0.07, 12.0, 0.95, 0.08, 9.0])",
            "call": "harmonic_coefficients(0.36, 2.20, 0.162939, p, 16)",
            "gold_call": "_oracle_harmonic_coefficients(0.36, 2.20, 0.162939, p, 16)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\np = np.array([24.0, 0.35, 0.18, 0.0, 1.10, 0.10, 15.0])\ndef run_model():\n    try:\n        harmonic_coefficients(0.44, 0.85, 0.32, p, 0); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try:\n        _oracle_harmonic_coefficients(0.44, 0.85, 0.32, p, 0); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
