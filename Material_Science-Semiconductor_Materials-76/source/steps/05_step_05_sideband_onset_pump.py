"""
Step 05 - Onset pump of a cavity sideband.

Pump at which a cavity sideband of the fundamental continuous wave turns unstable.

In a ring of length L with the reflection coefficient equal to one the field is
periodic in the propagation coordinate, so the admissible sideband offsets are
the cavity modes: sideband n (n = 1, 2, ...) sits n free spectral ranges from
the continuous wave, at the ordinary frequency offset n v_g / L with
v_g = c / n_group, with c = 299792458 m/s. The ring equations are written in
scaled units (time in units of tau_d, length in units of v_g tau_d), so the
sideband's scaled wavenumber offset follows from those units.

For the fundamental (k = 0) continuous wave the growth rate of sideband n
(step 03) depends on the pump mu through the continuous-wave state. The onset
pump of the sideband is the lowest mu in the window [mu_min, mu_max] at which
that growth rate changes sign from negative to positive. A growth rate can
become positive, return to negative and become positive again as mu grows; the
onset is the first crossing. The unstable and stable windows met in practice
are wider than 0.02 in mu, and the onset has to be located to 1e-9 in mu.

The window must start above the lasing threshold of the k = 0 wave and the
sideband must still be damped at mu_min; otherwise the onset lies outside the
window and is not defined here.

Returns
-------
float, the onset pump of cavity sideband n (lowest mu in the window where its growth rate turns positive)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sideband_onset_pump(n: int, L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, mu_min: float, mu_max: float) -> float:
    '''Lowest pump in [mu_min, mu_max] at which cavity sideband n of the k = 0 wave grows.

    Parameters
    ----------
    n : int
        Sideband index, n >= 1 (offset of n free spectral ranges).
    L_mm : float
        Ring length in millimetres, > 0.
    n_group : float
        Group index, > 0.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    Gamma, alpha, sigma, b : float
        Scaled gain bandwidth (> 0), linewidth enhancement factor, field loss
        rate (> 0) and carrier recovery rate (> 0).
    mu_min, mu_max : float
        Pump window, mu_min < mu_max, with mu_min above the lasing threshold of
        the k = 0 wave.

    Returns
    -------
    mu_on : float
        Onset pump of sideband n, to 1e-9.

    Raises
    ------
    ValueError
        If n is not an integer >= 1, if L_mm, n_group or tau_d_ps is not a
        finite positive number, if mu_min >= mu_max, for any condition under
        which step 03 raises at mu_min (mu_min at or below the lasing
        threshold included), if the sideband growth rate is already positive
        at mu_min, or if it does not become positive by mu_max.
    '''
    return mu_on

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _on_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _on_kn(n, L_mm, n_group, tau_d_ps):
    """Scaled wavenumber offset of cavity sideband n."""
    vg = 299792458.0 / n_group
    return 2.0 * np.pi * n * vg * (tau_d_ps * 1e-12) / (L_mm * 1e-3)


def _oracle_sideband_onset_pump(n: int, L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, mu_min: float, mu_max: float) -> float:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be an integer >= 1")
    L_mm = _on_check("L_mm", L_mm, True)
    n_group = _on_check("n_group", n_group, True)
    tau_d_ps = _on_check("tau_d_ps", tau_d_ps, True)
    mu_min = _on_check("mu_min", mu_min, True)
    mu_max = _on_check("mu_max", mu_max, True)
    if not mu_min < mu_max:
        raise ValueError("mu_min must be smaller than mu_max")
    kn = _on_kn(int(n), L_mm, n_group, tau_d_ps)
    rate = lambda mu: _oracle_sideband_growth_rate(kn, 0.0, Gamma, alpha, sigma, b, mu)
    if rate(mu_min) > 0.0:
        raise ValueError("the sideband is already unstable at mu_min")
    grid = np.arange(mu_min, mu_max + 1e-12, 0.005)
    if grid[-1] < mu_max:
        grid = np.append(grid, mu_max)
    prev = rate(grid[0])
    for lo, hi in zip(grid[:-1], grid[1:]):
        cur = rate(hi)
        if prev <= 0.0 < cur:
            return float(brentq(rate, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=500))
        prev = cur
    raise ValueError("the sideband does not become unstable by mu_max")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: 8th cavity sideband of a 4.7 mm terahertz ring ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 12.0)",
            "gold_call": "_oracle_sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Boundary: 3rd sideband of a short 1.5 mm ring with faster carriers ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_onset_pump(3, 1.5, 3.6, 0.1, 0.06, 0.92, 1.6e-3, 0.02, 2.0, 12.0)",
            "gold_call": "_oracle_sideband_onset_pump(3, 1.5, 3.6, 0.1, 0.06, 0.92, 1.6e-3, 0.02, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Edge: sideband that grows, is damped again and grows a second time in the window ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 16.0)",
            "gold_call": "_oracle_sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 16.0)",
            "tol": 1e-8,
        },
        # --- Edge: a sideband that is damped at first, then grows only at high pump ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_onset_pump(6, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 16.0)",
            "gold_call": "_oracle_sideband_onset_pump(6, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 16.0)",
            "tol": 1e-8,
        },
        # --- Invalid: sideband 3 of the 4.7 mm ring never grows below mu = 12 ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        sideband_onset_pump(3, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 12.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_sideband_onset_pump(3, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 2.0, 12.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {   # a sideband that grows, is damped again and grows once more: the onset is the first crossing
            "setup": 'import numpy as np',
            "call": "sideband_onset_pump(8, 4.7, 3.3, 0.1, 0.06, 0.9, 1.6e-3, 0.02, 2.0, 12.0)",
            "gold_call": "_oracle_sideband_onset_pump(8, 4.7, 3.3, 0.1, 0.06, 0.9, 1.6e-3, 0.02, 2.0, 12.0)",
            "tol": 1e-8,
        },
        {   # a lower sideband of the same re-entrant ring
            "setup": 'import numpy as np',
            "call": "sideband_onset_pump(6, 4.7, 3.3, 0.1, 0.06, 0.9, 1.6e-3, 0.02, 2.0, 12.0)",
            "gold_call": "_oracle_sideband_onset_pump(6, 4.7, 3.3, 0.1, 0.06, 0.9, 1.6e-3, 0.02, 2.0, 12.0)",
            "tol": 1e-8,
        },
        {   # a low sideband of a 3.4 mm ring at alpha = 0.96
            "setup": 'import numpy as np',
            "call": "sideband_onset_pump(4, 3.4, 3.4, 0.12, 0.06, 0.96, 2.0e-3, 0.018, 2.5, 13.0)",
            "gold_call": "_oracle_sideband_onset_pump(4, 3.4, 3.4, 0.12, 0.06, 0.96, 2.0e-3, 0.018, 2.5, 13.0)",
            "tol": 1e-8,
        },
        # --- Invalid: a sideband already unstable at mu_min, and an empty window ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 9.0, 12.0)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 4\n"
                     "    try:\n"
                     "        sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 9.0, 9.0)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    return mask\n"
                     "def run_gold():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        _oracle_sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 9.0, 12.0)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 4\n"
                     "    try:\n"
                     "        _oracle_sideband_onset_pump(8, 4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 9.0, 9.0)\n"
                     "    except ValueError:\n"
                     "        mask += 2\n"
                     "    except Exception:\n"
                     "        mask += 8\n"
                     "    return mask\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Normal: sideband whose growth window closes again inside the pump window ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_onset_pump(6, 3.8, 3.5, 0.1, 0.06, 0.96, 2.5e-3, 0.012, 2.0, 12.0)",
            "gold_call": "_oracle_sideband_onset_pump(6, 3.8, 3.5, 0.1, 0.06, 0.96, 2.5e-3, 0.012, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Normal: the sideband of the same ring that grows later and keeps growing ---
        {
            "setup": "import numpy as np\n",
            "call": "sideband_onset_pump(7, 3.8, 3.5, 0.1, 0.06, 0.96, 2.5e-3, 0.012, 2.0, 12.0)",
            "gold_call": "_oracle_sideband_onset_pump(7, 3.8, 3.5, 0.1, 0.06, 0.96, 2.5e-3, 0.012, 2.0, 12.0)",
            "tol": 1e-8,
        },
]
