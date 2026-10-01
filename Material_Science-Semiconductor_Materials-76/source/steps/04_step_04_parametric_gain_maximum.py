"""
Step 04 - Parametric-gain maximum of the fundamental continuous wave.

Maximum of the parametric gain of the fundamental continuous wave.

The fundamental continuous wave of the ring laser is the k = 0 solution of step 01, the single-frequency state closest to the gain maximum and the one with the lowest threshold. Its parametric gain is the growth rate of step 03, taken as a function of the sideband offset treated as a continuous variable. In the scaled units of the ring equations (time in units of tau_d, length in units of v_g tau_d) the group velocity is 1; sideband offsets are reported as ordinary (not angular) frequencies.

This step reports the global maximum of that gain over offsets f_max/4000 <= f <= f_max (positive offsets; the gain is symmetric under kn -> -kn) as a position in GHz and a height as a physical growth rate in 1/ns (the scaled growth rate divided by tau_d). The gain can be positive on more than one band of offsets inside the interval, and the maximum wanted here is the highest of them. The gain must be positive somewhere on the interval: if it is nowhere positive the laser has no parametric instability there and no peak is defined. The position has to be located to a relative precision of 1e-9 or better.

Returns
-------
numpy.ndarray of shape (2,): [frequency offset of the parametric-gain maximum in GHz, growth rate there in 1/ns]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parametric_gain_maximum(Gamma: float, alpha: float, sigma: float, b: float, mu: float, tau_d_ps: float, f_max_ghz: float) -> "np.ndarray":
    '''Position and height of the parametric-gain maximum of the k = 0 wave.

    Parameters
    ----------
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    sigma : float
        Scaled field loss rate, > 0.
    b : float
        Scaled carrier recovery rate, > 0.
    mu : float
        Pump parameter, above the lasing threshold of the k = 0 wave.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    f_max_ghz : float
        Upper end of the searched sideband offsets in GHz, > 0.

    Returns
    -------
    peak : np.ndarray
        Shape (2,): [frequency offset of the maximum in GHz, growth rate at
        the maximum in 1/ns].

    Raises
    ------
    ValueError
        If tau_d_ps or f_max_ghz is not a finite positive number, for any
        condition under which step 03 raises for these parameters, or if the
        growth rate is nowhere positive on f_max_ghz/4000 <= f <= f_max_ghz.
    '''
    return peak

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg
from scipy.optimize import brentq, minimize_scalar


def _pg_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _pg_slope(kn, state, Gamma, alpha, sigma, b):
    """d Re(lambda_max)/d kn for the k = 0 wave, from the left and right eigenvectors of the sideband matrix."""
    omega, d0, x, p_re, p_im = state
    m = _sb_matrix(kn, 0.0, omega, d0, np.sqrt(x), p_re + 1j * p_im, float(Gamma), float(alpha), float(sigma), float(b))
    lam, vl, vr = scipy.linalg.eig(m, left=True, right=True)
    j = int(np.argmax(lam.real))
    w, v = vl[:, j], vr[:, j]
    return float((1j * (np.conj(w[0]) * v[0] + np.conj(w[1]) * v[1]) / (np.conj(w) @ v)).real)


def _oracle_parametric_gain_maximum(Gamma: float, alpha: float, sigma: float, b: float, mu: float, tau_d_ps: float, f_max_ghz: float) -> "np.ndarray":
    tau = _pg_check("tau_d_ps", tau_d_ps, True) * 1e-12
    fmax = _pg_check("f_max_ghz", f_max_ghz, True)
    to_kn = lambda f_ghz: 2.0 * np.pi * f_ghz * 1e9 * tau

    def _rate(f_ghz):
        return _oracle_sideband_growth_rate(to_kn(f_ghz), 0.0, Gamma, alpha, sigma, b, mu)

    grid = np.linspace(fmax / 4000.0, fmax, 4000)
    vals = np.array([_rate(f) for f in grid])
    i = int(np.argmax(vals))
    if not vals[i] > 0.0:
        raise ValueError("the parametric gain is nowhere positive on the searched interval")
    f_best, g_best = grid[i], vals[i]
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, grid.size - 1)]
    state = _oracle_cw_emission_state(0.0, Gamma, alpha, sigma, mu)
    slope = lambda f_ghz: _pg_slope(to_kn(f_ghz), state, Gamma, alpha, sigma, b)
    if 0 < i < grid.size - 1 and slope(lo) > 0.0 > slope(hi):
        f_root = brentq(slope, lo, hi, xtol=1e-13 * fmax, rtol=4.0 * np.finfo(float).eps, maxiter=200)
        g_root = _rate(f_root)
        if g_root >= g_best:
            f_best, g_best = f_root, g_root
    else:
        res = minimize_scalar(lambda f: -_rate(f), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-10 * fmax, "maxiter": 500})
        if -res.fun >= g_best:
            f_best, g_best = res.x, -res.fun
    return np.array([f_best, g_best / tau * 1e-9], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: terahertz ring laser above its multimode threshold ---
        {
            "setup": "import numpy as np\n",
            "call": "parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 9.2, 0.1, 400.0)",
            "gold_call": "_oracle_parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 9.2, 0.1, 400.0)",
            "tol": 1e-9,
        },
        # --- Boundary: alpha > 1, gain band reaching down to low offsets ---
        {
            "setup": "import numpy as np\n",
            "call": "parametric_gain_maximum(0.06, 1.05, 1.6e-3, 0.02, 7.9, 0.1, 400.0)",
            "gold_call": "_oracle_parametric_gain_maximum(0.06, 1.05, 1.6e-3, 0.02, 7.9, 0.1, 400.0)",
            "tol": 1e-9,
        },
        # --- Edge: faster carriers, longer dephasing time, narrow search window ---
        {
            "setup": "import numpy as np\n",
            "call": "parametric_gain_maximum(0.05, 0.9, 2.5e-3, 0.03, 11.0, 0.15, 180.0)",
            "gold_call": "_oracle_parametric_gain_maximum(0.05, 0.9, 2.5e-3, 0.03, 11.0, 0.15, 180.0)",
            "tol": 1e-9,
        },
        # --- Invalid: below the multimode threshold the gain is nowhere positive ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 4.0, 0.1, 400.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 4.0, 0.1, 400.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {   # alpha above unity, where the gain is positive all the way to zero offset
            "setup": 'import numpy as np',
            "call": "parametric_gain_maximum(0.06, 2.0, 1.6e-3, 0.014, 6.0, 0.1, 400.0)",
            "gold_call": "_oracle_parametric_gain_maximum(0.06, 2.0, 1.6e-3, 0.014, 6.0, 0.1, 400.0)",
            "tol": 1e-9,
        },
        {   # a fast medium at alpha = 1.5 with a slower carrier rate
            "setup": 'import numpy as np',
            "call": "parametric_gain_maximum(0.09, 1.5, 3.0e-3, 8.0e-3, 5.0, 0.2, 250.0)",
            "gold_call": "_oracle_parametric_gain_maximum(0.09, 1.5, 3.0e-3, 8.0e-3, 5.0, 0.2, 250.0)",
            "tol": 1e-9,
        },
        # --- Invalid: a pump at which the gain is nowhere positive, and a bad interval ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 3.0, 0.1, 400.0)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 4\n"
                     "    try:\n"
                     "        parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 9.2, 0.1, -400.0)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    return mask\n"
                     "def run_gold():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        _oracle_parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 3.0, 0.1, 400.0)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 4\n"
                     "    try:\n"
                     "        _oracle_parametric_gain_maximum(0.06, 0.95, 1.6e-3, 0.014, 9.2, 0.1, -400.0)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    return mask\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: alpha above one, two bands of positive gain, the lower one reaching to 0.13 GHz ---
        {
            "setup": "import numpy as np\n",
            "call": "parametric_gain_maximum(0.06, 1.05, 1.6e-3, 0.014, 8.0, 0.1, 400.0)",
            "gold_call": "_oracle_parametric_gain_maximum(0.06, 1.05, 1.6e-3, 0.014, 8.0, 0.1, 400.0)",
             "tol": 1e-9,
        },
]
