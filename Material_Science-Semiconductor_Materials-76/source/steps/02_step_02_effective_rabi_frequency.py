"""
Step 02 - Effective Rabi frequency of the gain medium.

Effective Rabi frequency of the gain medium of a semiconductor laser.

In a two-level laser the polarisation and the population inversion, driven by a
strong field, oscillate at the Rabi frequency. The medium of this laser is the
one of step 01, and the analogous characteristic frequency is defined for a
field that is held fixed: F constant, real and monochromatic at the reference
frequency (the maximum of the unsaturated gain, so that no detuning term
appears), with intensity X = |F|^2.

With the field clamped in that way the medium has a steady state, and a small
disturbance of that state oscillates while its amplitude changes. The
effective Rabi frequency (ERF) is the angular frequency of that oscillation,
and the damping coefficient is the magnitude of the rate at which its
amplitude changes; both are reported as positive numbers, whether the
oscillation dies away or grows. If the disturbance does not oscillate at all,
the medium is overdamped and neither number is defined.

Both are returned in the scaled units of step 01, i.e. as angular rates in
units of 1/tau_d; dividing by 2*pi*tau_d converts them to ordinary
frequencies.

Returns
-------
numpy.ndarray of shape (2,): [ERF, damping] of the clamped-field medium, angular rates in units of 1/tau_d
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_rabi_frequency(X: float, Gamma: float, alpha: float, b: float) -> "np.ndarray":
    '''Effective Rabi frequency and damping of the medium for a clamped field.

    Parameters
    ----------
    X : float
        Field intensity |F|^2 (scaled), >= 0.
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    b : float
        Scaled carrier recovery rate, > 0.

    Returns
    -------
    rates : np.ndarray
        Shape (2,): [ERF, damping], the angular frequency of the clamped-field
        medium's oscillation and the magnitude of the rate at which its amplitude
        changes, both positive, in units of 1/tau_d.

    Raises
    ------
    ValueError
        If any argument is not a finite real number, if X < 0, if Gamma <= 0
        or b <= 0, or if the clamped-field medium is overdamped, so that it has no
        oscillation.
    '''
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _erf_check(name, value, positive, nonneg=False):
    """Return value as a finite float with the requested sign constraint."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0) or (nonneg and value < 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _oracle_effective_rabi_frequency(X: float, Gamma: float, alpha: float, b: float) -> "np.ndarray":
    X = _erf_check("X", X, False, nonneg=True)
    Gamma = _erf_check("Gamma", Gamma, True)
    alpha = _erf_check("alpha", alpha, False)
    b = _erf_check("b", b, True)
    f = np.sqrt(X)
    cp = 1.0 + 1j * alpha
    cm = 1.0 - 1j * alpha
    jac = np.array([[-Gamma * cp, 0.0, -Gamma * cp ** 2 * f],
                    [0.0, -Gamma * cm, -Gamma * cm ** 2 * f],
                    [0.5 * b * f, 0.5 * b * f, -b]], dtype=complex)
    ev = np.linalg.eigvals(jac)
    scale = max(1.0, float(np.max(np.abs(ev))))
    cplx = ev[np.abs(ev.imag) > 1e-10 * scale]
    if cplx.size < 2:
        raise ValueError("no complex-conjugate eigenvalue pair: overdamped medium")
    pick = cplx[np.argmax(np.abs(cplx.imag))]
    return np.array([abs(pick.imag), abs(pick.real)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: quantum cascade laser medium well above threshold ---
        {
            "setup": "import numpy as np\n",
            "call": "effective_rabi_frequency(8.158, 0.06, 0.95, 0.014)",
            "gold_call": "_oracle_effective_rabi_frequency(8.158, 0.06, 0.95, 0.014)",
            "tol": 1e-10,
        },
        # --- Boundary: zero intensity (laser threshold) ---
        {
            "setup": "import numpy as np\n",
            "call": "effective_rabi_frequency(0.0, 0.06, 1.05, 0.02)",
            "gold_call": "_oracle_effective_rabi_frequency(0.0, 0.06, 1.05, 0.02)",
            "tol": 1e-10,
        },
        # --- Edge: two-level medium (alpha = 0) above its oscillation onset ---
        {
            "setup": "import numpy as np\n",
            "call": "effective_rabi_frequency(3.0, 0.06, 0.0, 0.014)",
            "gold_call": "_oracle_effective_rabi_frequency(3.0, 0.06, 0.0, 0.014)",
            "tol": 1e-10,
        },
        # --- Edge: slow quantum-well-like carriers and large alpha ---
        {
            "setup": "import numpy as np\n",
            "call": "effective_rabi_frequency(2.5, 0.06, 3.0, 1.0e-4)",
            "gold_call": "_oracle_effective_rabi_frequency(2.5, 0.06, 3.0, 1.0e-4)",
            "tol": 1e-10,
        },
        # --- Invalid: two-level medium below the oscillation onset (overdamped) ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        effective_rabi_frequency(0.2, 0.06, 0.0, 0.014)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_effective_rabi_frequency(0.2, 0.06, 0.0, 0.014)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {   # a strongly amplitude-phase coupled medium, alpha above unity
            "setup": 'import numpy as np',
            "call": "effective_rabi_frequency(5.0, 0.06, 2.0, 0.014)",
            "gold_call": "_oracle_effective_rabi_frequency(5.0, 0.06, 2.0, 0.014)",
            "tol": 1e-10,
        },
        {   # the two-level limit well above the splitting threshold
            "setup": 'import numpy as np',
            "call": "effective_rabi_frequency(1.5, 0.06, 0.0, 0.014)",
            "gold_call": "_oracle_effective_rabi_frequency(1.5, 0.06, 0.0, 0.014)",
            "tol": 1e-10,
        },
        {   # a weaker field on a slower medium at alpha = 1.4
            "setup": 'import numpy as np',
            "call": "effective_rabi_frequency(0.8, 0.02, 1.4, 6.0e-3)",
            "gold_call": "_oracle_effective_rabi_frequency(0.8, 0.02, 1.4, 6.0e-3)",
            "tol": 1e-10,
        },
        # --- Invalid: an overdamped medium, and arguments outside their ranges ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        effective_rabi_frequency(0.35, 0.06, 0.0, 0.014)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    try:\n"
                     "        effective_rabi_frequency(-1.0, 0.06, 0.95, 0.014)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 16\n"
                     "    try:\n"
                     "        effective_rabi_frequency(2.0, 0.06, 0.95, 0.0)\n"
                     "    except ValueError:\n"
                     "        mask += 4\n"
                     "    except Exception:\n"
                     "        mask += 32\n"
                     "    return mask\n"
                     "def run_gold():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        _oracle_effective_rabi_frequency(0.35, 0.06, 0.0, 0.014)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    try:\n"
                     "        _oracle_effective_rabi_frequency(-1.0, 0.06, 0.95, 0.014)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 16\n"
                     "    try:\n"
                     "        _oracle_effective_rabi_frequency(2.0, 0.06, 0.95, 0.0)\n"
                     "    except ValueError:\n"
                     "        mask += 4\n"
                     "    except Exception:\n"
                     "        mask += 32\n"
                     "    return mask\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
]
