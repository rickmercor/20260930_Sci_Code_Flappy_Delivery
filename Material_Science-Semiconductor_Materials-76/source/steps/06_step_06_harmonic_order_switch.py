"""
Step 06 - Multimode threshold and first change of the predicted comb order.

Harmonic order selected by linear stability and the pump at which it first changes.

Linear stability analysis predicts which harmonic frequency comb a ring laser
forms when its fundamental (k = 0) continuous wave becomes unstable: the comb
order is the index of the cavity sideband that grows fastest. The cavity
sidebands are those of step 05 (n = 1, ..., n_max, offset n v_g / L,
v_g = c / n_group) and their growth rates are those of step 03 on the k = 0
wave.

The multimode instability threshold mu_c is the lowest pump in the window at
which any of the sidebands 1 ... n_max is growing, and n_c is the sideband
that grows there. Raising the pump further, the predicted order changes at the
first pump mu_star above mu_c at which some other sideband grows faster than
n_c, and n_new is that sideband. The lead can be brief: a sideband may take
over and lose the lead again as the pump rises, and the first change is the one
wanted here whether or not it lasts. Sidebands that never grow inside the
window do not set mu_c, but every sideband 1 ... n_max takes part in the
comparison above mu_c.

In the tested cases the two lowest onset pumps of the whole set differ by at
least 0.01 in mu; any interval in which the selected sideband differs from its
neighbours is at least 0.01 wide in mu; and mu_c and mu_star have to be
located to 1e-9.

Returns
-------
numpy.ndarray of shape (4,): [mu_c, n_c, mu_star, n_new] (sideband indices stored as floats)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def harmonic_order_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float) -> "np.ndarray":
    '''Multimode threshold, onset order, first order-change pump and new order.

    Parameters
    ----------
    L_mm : float
        Ring length in millimetres, > 0.
    n_group : float
        Group index, > 0.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    Gamma, alpha, sigma, b : float
        Scaled gain bandwidth (> 0), linewidth enhancement factor, field loss
        rate (> 0) and carrier recovery rate (> 0).
    n_max : int
        Highest sideband index considered, n_max >= 2.
    mu_min, mu_max : float
        Pump window, mu_min < mu_max, mu_min above the lasing threshold of the
        k = 0 wave, with every sideband 1 ... n_max damped at mu_min.

    Returns
    -------
    result : np.ndarray
        Shape (4,): [mu_c, n_c, mu_star, n_new] with n_c and n_new stored as
        floats holding integer values.

    Raises
    ------
    ValueError
        If n_max is not an integer >= 2, for any condition under which step 05
        raises for its other arguments, if some sideband 1 ... n_max is
        already growing at mu_min, if no sideband starts to grow inside the
        window, or if the selected sideband does not change before mu_max.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _hs_check(name, value, positive):
    """Return value as a finite float, optionally requiring positivity."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("%s must be a real number" % name)
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError("%s is out of range" % name)
    return value


def _hs_kn(n, L_mm, n_group, tau_d_ps):
    """Scaled wavenumber offset of cavity sideband n."""
    vg = 299792458.0 / n_group
    return 2.0 * np.pi * n * vg * (tau_d_ps * 1e-12) / (L_mm * 1e-3)


def _oracle_harmonic_order_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float) -> "np.ndarray":
    if isinstance(n_max, bool) or not isinstance(n_max, (int, np.integer)) or int(n_max) < 2:
        raise ValueError("n_max must be an integer >= 2")
    n_max = int(n_max)
    mu_max = _hs_check("mu_max", mu_max, True)
    onsets = {}
    for n in range(1, n_max + 1):
        try:
            onsets[n] = _oracle_sideband_onset_pump(n, L_mm, n_group, tau_d_ps, Gamma, alpha, sigma, b, mu_min, mu_max)
        except ValueError as exc:
            if "does not become unstable" in str(exc):
                continue
            raise
    if not onsets:
        raise ValueError("no sideband becomes unstable inside the pump window")
    n_c = min(onsets, key=lambda n: (onsets[n], n))
    mu_c = onsets[n_c]
    kns = [_hs_kn(n, L_mm, n_group, tau_d_ps) for n in range(1, n_max + 1)]

    def _rates(mu):
        return np.array([_oracle_sideband_growth_rate(kn, 0.0, Gamma, alpha, sigma, b, mu) for kn in kns])

    def _rivals(mu):
        r = _rates(mu)
        best_other = max((n for n in range(1, n_max + 1) if n != n_c), key=lambda n: r[n - 1])
        return r[best_other - 1] - r[n_c - 1], best_other

    step = 0.005
    lo = mu_c
    while lo < mu_max:
        hi = min(lo + step, mu_max)
        d_hi, who = _rivals(hi)
        if d_hi > 0.0:
            kn_new = kns[who - 1]
            kn_old = kns[n_c - 1]
            diff = lambda mu: (_oracle_sideband_growth_rate(kn_new, 0.0, Gamma, alpha, sigma, b, mu)
                               - _oracle_sideband_growth_rate(kn_old, 0.0, Gamma, alpha, sigma, b, mu))
            mu_star = brentq(diff, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=500)
            return np.array([mu_c, float(n_c), mu_star, float(who)], dtype=float)
        lo = hi
    raise ValueError("the selected sideband does not change before mu_max")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: 4.7 mm terahertz ring, alpha below one ---
        {
            "setup": "import numpy as np\n",
            "call": "harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Boundary: long ring, the order changes shortly after the threshold ---
        {
            "setup": "import numpy as np\n",
            "call": "harmonic_order_switch(6.0, 3.5, 0.1, 0.06, 0.93, 1.6e-3, 0.012, 20, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(6.0, 3.5, 0.1, 0.06, 0.93, 1.6e-3, 0.012, 20, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Edge: stronger alpha close to one and a different group index ---
        {
            "setup": "import numpy as np\n",
            "call": "harmonic_order_switch(3.2, 3.3, 0.1, 0.06, 0.98, 2.0e-3, 0.016, 14, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(3.2, 3.3, 0.1, 0.06, 0.98, 2.0e-3, 0.016, 14, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Invalid: window ends before any sideband grows ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 7.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 16, 2.0, 7.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: alpha above one, long-wavelength sidebands already grow at mu_min ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 1.01, 1.6e-3, 0.014, 16, 1.5, 12.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 1.01, 1.6e-3, 0.014, 16, 1.5, 12.0)\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
            {   # a ring whose selected sideband never changes inside the window, which the contract rejects
            "setup": 'import numpy as np\ndef probe():\n    try:\n        harmonic_order_switch(4.7, 3.3, 0.1, 0.06, 0.9, 1.6e-3, 0.02, 16, 2.0, 12.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef probe_gold():\n    try:\n        _oracle_harmonic_order_switch(4.7, 3.3, 0.1, 0.06, 0.9, 1.6e-3, 0.02, 16, 2.0, 12.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        {   # a 5.6 mm ring at alpha = 0.94 with a faster carrier rate
            "setup": 'import numpy as np',
            "call": "harmonic_order_switch(5.6, 3.6, 0.1, 0.06, 0.94, 1.6e-3, 0.024, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(5.6, 3.6, 0.1, 0.06, 0.94, 1.6e-3, 0.024, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        {   # a 3.6 mm ring at alpha = 0.90 and the larger loss
            "setup": 'import numpy as np',
            "call": "harmonic_order_switch(3.6, 3.6, 0.1, 0.06, 0.90, 2.5e-3, 0.024, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(3.6, 3.6, 0.1, 0.06, 0.90, 2.5e-3, 0.024, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Invalid: a sideband count below the documented minimum ---
        {
            "setup": "import numpy as np\n"
                     "def run_model():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 1, 2.0, 12.0)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 2\n"
                     "    return mask\n"
                     "def run_gold():\n"
                     "    mask = 0\n"
                     "    try:\n"
                     "        _oracle_harmonic_order_switch(4.7, 3.6, 0.1, 0.06, 0.95, 1.6e-3, 0.014, 1, 2.0, 12.0)\n"
                     "    except ValueError:\n"
                     "        mask += 1\n"
                     "    except Exception:\n"
                     "        mask += 2\n"
                     "    return mask\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {   # a 4.6 mm ring at alpha = 0.92 with the faster carrier rate
            "setup": 'import numpy as np',
            "call": "harmonic_order_switch(4.6, 3.5, 0.1, 0.06, 0.92, 1.6e-3, 0.018, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(4.6, 3.5, 0.1, 0.06, 0.92, 1.6e-3, 0.018, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        {   # a 6.2 mm ring at alpha = 0.95 and the larger loss
            "setup": 'import numpy as np',
            "call": "harmonic_order_switch(6.2, 3.6, 0.1, 0.06, 0.95, 2.5e-3, 0.014, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(6.2, 3.6, 0.1, 0.06, 0.95, 2.5e-3, 0.014, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        {   # a 5.2 mm ring at alpha = 0.93 with an intermediate loss
            "setup": 'import numpy as np',
            "call": "harmonic_order_switch(5.2, 3.5, 0.1, 0.06, 0.93, 2.0e-3, 0.018, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(5.2, 3.5, 0.1, 0.06, 0.93, 2.0e-3, 0.018, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Normal: the sideband that grows first stops growing well before mu_max ---
        {
            "setup": "import numpy as np\n",
            "call": "harmonic_order_switch(3.8, 3.5, 0.1, 0.06, 0.96, 2.5e-3, 0.012, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(3.8, 3.5, 0.1, 0.06, 0.96, 2.5e-3, 0.012, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Normal: same behaviour on a longer ring, threshold sideband 8 ---
        {
            "setup": "import numpy as np\n",
            "call": "harmonic_order_switch(5.0, 3.5, 0.1, 0.06, 0.94, 2.0e-3, 0.012, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(5.0, 3.5, 0.1, 0.06, 0.94, 2.0e-3, 0.012, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
        # --- Edge: the predicted order falls to a lower sideband instead of rising ---
        {
            "setup": "import numpy as np\n",
            "call": "harmonic_order_switch(3.0, 3.6, 0.1, 0.06, 0.94, 1.6e-3, 0.030, 16, 2.0, 12.0)",
            "gold_call": "_oracle_harmonic_order_switch(3.0, 3.6, 0.1, 0.06, 0.94, 1.6e-3, 0.030, 16, 2.0, 12.0)",
            "tol": 1e-8,
        },
    ]
