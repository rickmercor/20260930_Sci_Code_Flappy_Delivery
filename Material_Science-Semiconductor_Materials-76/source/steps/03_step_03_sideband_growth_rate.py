"""
Step 03 - Growth rate of a sideband perturbation of the continuous wave.

Linear stability of continuous-wave emission against a sideband perturbation.

Multimode emission and frequency combs in a ring laser start when the single-frequency solution becomes unstable. The ring, its scaled units and its continuous-wave solutions are those of step 01: the wave of wavenumber k has F = F0 exp(-i k eta + i omega t), P = P0 exp(-i k eta + i omega t) and D = D0, with F0 real and positive.

An infinitesimal sideband perturbation of that wave carries a wavenumber offset kn, measured from the continuous wave itself rather than from the reference frequency. The number returned here is the fastest exponential rate at which such a perturbation of offset kn grows on the wave: positive means the continuous wave is unstable against that sideband, negative that the disturbance dies away.

The rate is returned in the scaled units of step 01, i.e. in units of 1/tau_d.

Returns
-------
float, the largest real part of the sideband growth exponents, in units of 1/tau_d
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sideband_growth_rate(kn: float, k: float, Gamma: float, alpha: float, sigma: float, b: float, mu: float) -> float:
    '''Growth rate of a sideband of offset kn on the continuous wave of wavenumber k.

    Parameters
    ----------
    kn : float
        Scaled wavenumber offset of the sideband from the continuous wave.
    k : float
        Scaled wavenumber of the continuous wave (step 01).
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    sigma : float
        Scaled field loss rate, > 0.
    b : float
        Scaled carrier recovery rate, > 0.
    mu : float
        Pump parameter, above the lasing threshold of the continuous wave.

    Returns
    -------
    rate : float
        Fastest exponential growth rate of a perturbation of offset kn, in
        units of 1/tau_d.

    Raises
    ------
    ValueError
        If kn or b is not a finite real number or b <= 0, or for any
        condition under which step 01 raises for (k, Gamma, alpha, sigma, mu).
    '''
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sb_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _sb_matrix(kn, k, omega, d0, f0, p0, Gamma, alpha, sigma, b):
    """Linear evolution matrix of (dF_n, dF*_-n, dP_n, dP*_-n, dD_n)."""
    cp = 1.0 + 1j * alpha
    cm = 1.0 - 1j * alpha
    m = np.zeros((5, 5), dtype=complex)
    m[0, 0] = -sigma + 1j * (k - omega + kn)
    m[0, 2] = -sigma
    m[1, 1] = -sigma - 1j * (k - omega - kn)
    m[1, 3] = -sigma
    m[2, 0] = -Gamma * cp ** 2 * d0
    m[2, 2] = -1j * omega - Gamma * cp
    m[2, 4] = -Gamma * cp ** 2 * f0
    m[3, 1] = -Gamma * cm ** 2 * d0
    m[3, 3] = 1j * omega - Gamma * cm
    m[3, 4] = -Gamma * cm ** 2 * np.conj(f0)
    m[4, 0] = 0.5 * b * np.conj(p0)
    m[4, 1] = 0.5 * b * p0
    m[4, 2] = 0.5 * b * np.conj(f0)
    m[4, 3] = 0.5 * b * f0
    m[4, 4] = -b
    return m


def _oracle_sideband_growth_rate(kn: float, k: float, Gamma: float, alpha: float, sigma: float, b: float, mu: float) -> float:
    kn = _sb_check("kn", kn, False)
    b = _sb_check("b", b, True)
    omega, d0, x, p_re, p_im = _oracle_cw_emission_state(k, Gamma, alpha, sigma, mu)
    f0 = np.sqrt(x)
    p0 = p_re + 1j * p_im
    m = _sb_matrix(kn, float(k), omega, d0, f0, p0, float(Gamma), float(alpha), float(sigma), b)
    return float(np.max(np.linalg.eigvals(m).real))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: sideband inside the unstable band of a terahertz ring laser ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_growth_rate(0.0891, 0.0, 0.06, 0.95, 1.6e-3, 0.014, 9.2)",
            "gold_call": "_oracle_sideband_growth_rate(0.0891, 0.0, 0.06, 0.95, 1.6e-3, 0.014, 9.2)",
            "tol": 1e-9,
        },
        # --- Normal: damped sideband outside the band, same laser ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_growth_rate(0.0557, 0.0, 0.06, 0.95, 1.6e-3, 0.014, 9.2)",
            "gold_call": "_oracle_sideband_growth_rate(0.0557, 0.0, 0.06, 0.95, 1.6e-3, 0.014, 9.2)",
            "tol": 1e-9,
        },
        # --- Boundary: long-wavelength sideband with alpha > 1, near the phase mode ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_growth_rate(0.012, 0.0, 0.06, 1.05, 1.6e-3, 0.014, 4.0)",
            "gold_call": "_oracle_sideband_growth_rate(0.012, 0.0, 0.06, 1.05, 1.6e-3, 0.014, 4.0)",
            "tol": 1e-9,
        },
        # --- Edge: continuous wave detuned from the gain maximum, negative offset ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_growth_rate(-0.074, 0.011, 0.06, 0.8, 4.0e-3, 0.03, 6.5)",
            "gold_call": "_oracle_sideband_growth_rate(-0.074, 0.011, 0.06, 0.8, 4.0e-3, 0.03, 6.5)",
            "tol": 1e-9,
        },
        # --- Edge: two-level medium (alpha = 0), Rabi-type sideband ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_growth_rate(0.16, 0.0, 0.06, 0.0, 1.6e-3, 0.014, 12.0)",
            "gold_call": "_oracle_sideband_growth_rate(0.16, 0.0, 0.06, 0.0, 1.6e-3, 0.014, 12.0)",
            "tol": 1e-9,
        },
        # --- Invalid: pump below the lasing threshold of the continuous wave ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        sideband_growth_rate(0.0891, 0.0, 0.06, 0.95, 1.6e-3, 0.014, 0.9)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_sideband_growth_rate(0.0891, 0.0, 0.06, 0.95, 1.6e-3, 0.014, 0.9)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {   # alpha above unity, where the unstable band reaches down to zero offset
            "setup": 'import numpy as np',
            "call": "sideband_growth_rate(0.002, 0.0, 0.06, 2.0, 1.6e-3, 0.014, 6.0)",
            "gold_call": "_oracle_sideband_growth_rate(0.002, 0.0, 0.06, 2.0, 1.6e-3, 0.014, 6.0)",
            "tol": 1e-9,
        },
        {   # a sideband of a strongly detuned wave, where the pulled frequency enters
            "setup": 'import numpy as np',
            "call": "sideband_growth_rate(0.06, 0.03, 0.06, 0.95, 1.6e-3, 0.014, 9.2)",
            "gold_call": "_oracle_sideband_growth_rate(0.06, 0.03, 0.06, 0.95, 1.6e-3, 0.014, 9.2)",
            "tol": 1e-9,
        },
        {   # detuning of the other sign, damped sideband
            "setup": 'import numpy as np',
            "call": "sideband_growth_rate(0.09, -0.025, 0.06, 0.9, 2.5e-3, 0.02, 7.0)",
            "gold_call": "_oracle_sideband_growth_rate(0.09, -0.025, 0.06, 0.9, 2.5e-3, 0.02, 7.0)",
            "tol": 1e-9,
        },
        {   # a fast medium at alpha = 1.5 with a large offset
            "setup": 'import numpy as np',
            "call": "sideband_growth_rate(0.2, 0.0, 0.09, 1.5, 3.0e-3, 8.0e-3, 5.0)",
            "gold_call": "_oracle_sideband_growth_rate(0.2, 0.0, 0.09, 1.5, 3.0e-3, 8.0e-3, 5.0)",
            "tol": 1e-9,
        },
        {   # a detuned wave at alpha = 1.1 with the larger loss
            "setup": 'import numpy as np',
            "call": "sideband_growth_rate(0.035, -0.012, 0.06, 1.1, 2.0e-3, 0.018, 8.0)",
            "gold_call": "_oracle_sideband_growth_rate(0.035, -0.012, 0.06, 1.1, 2.0e-3, 0.018, 8.0)",
            "tol": 1e-9,
        },
        # --- Invalid: arguments outside their documented ranges ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        sideband_growth_rate(0.0891, 0.0, 0.06, 0.95, 1.6e-3, 0.0, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 4\n"
                     "    try:\n"
                     "        sideband_growth_rate(float('nan'), 0.0, 0.06, 0.95, 1.6e-3, 0.014, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    return mask\n"
                     "def run_gold():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        _oracle_sideband_growth_rate(0.0891, 0.0, 0.06, 0.95, 1.6e-3, 0.0, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 4\n"
                     "    try:\n"
                     "        _oracle_sideband_growth_rate(float('nan'), 0.0, 0.06, 0.95, 1.6e-3, 0.014, 9.2)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    return mask\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
]
