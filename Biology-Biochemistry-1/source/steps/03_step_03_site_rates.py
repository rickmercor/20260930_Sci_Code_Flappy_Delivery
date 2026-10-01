"""
Step 3 - the two competing rates of every reactive site.

Each site competes between a force-accelerated homolytic scission and an experiment-based hydrolysis, and the two laws use the force differently: the scission barrier is the rise of V_eff from its shifted well minimum to its barrier top, both located in step 2, while the force-clamp relation keeps the force in nanonewtons. 

The barrier equals the sites full E_dis at vanishing load and is exactly zero once the two stationary points have merged. The scission prefactor is the released implementation's calibrated default, nu = 0.288 ps^-1 = 2.88e11 s^-1, evaluated at T = 300 K with R = 8.31446261815324e-3 kJ mol^-1 K^-1, and E_dis is whatever the site carries now, including any radical weakening. This step evaluates both laws over a whole site array and returns the two rate vectors the selection steps consume.

Returns
-------
r_hom, r_hyd : tuple of 2 np.ndarray, each shape (n,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def site_rates(site_array: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Homolysis and hydrolysis rate of every site, in s^-1.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F]: dissociation energy in kJ/mol (positive, current value
        including any weakening), force constant in kJ/mol/nm^2 (positive), pulling force
        in nN (positive).

    Returns
    -------
    tuple of np.ndarray
        (r_hom, r_hyd), each shape (n,) and dtype float, in site-array order.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, or if any of
        E_dis, k or F is not positive.
    """
    return r_hom, r_hyd  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_site_rates(site_array: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation for site_rates."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    xs = _oracle_stationary_points(arr)          # validates the array as well

    NU, RGAS, T, TH = 2.88e11, 8.31446261815324e-3, 300.0, 294.15
    r_hom, r_hyd = [], []
    for j, (E, k, F) in enumerate(arr):
        beta = np.sqrt(k / (2.0 * E))
        g = 602.2 * F
        v = lambda x: E * (1.0 - np.exp(-beta * x)) ** 2 - g * x
        dV = float(max(v(xs[j, 1]) - v(xs[j, 0]), 0.0))
        r_hom.append(float(NU * np.exp(-dV / (RGAS * T))))

        low = 26.26 * F - 19.77
        high = -20.342988 + 0.070648 * TH + 1.605233 * F
        if F <= 0.65:
            lnk = low
        elif F > 0.75:
            lnk = high
        else:
            frac = (F - 0.65) / 0.10
            lnk = (1.0 - frac) * low + frac * high
        r_hyd.append(float(np.exp(lnk)))
    return np.asarray(r_hom), np.asarray(r_hyd)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    graded = ("sites = np.array([[251.0, 15954.0, 0.50], [280.0, 16082.0, 0.62],\n"
              "                  [230.0, 27566.0, 0.74], [170.0, 73368.0, 0.86],\n"
              "                  [350.0, 22016.0, 0.98], [260.0, 45653.0, 1.10]])\n")
    invalid = setup + (
        "def run_model(s):\n"
        "    try:\n"
        "        site_rates(s)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(s):\n"
        "    try:\n"
        "        _oracle_site_rates(s)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: the graded six-site network, one site inside the transition band.
        {"setup": setup + graded,
         "call": "site_rates(sites)",
         "gold_call": "_oracle_site_rates(sites)"},
        # Normal: a lightly loaded pair, where the barrier is close to the full E_dis.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.06], [235.0, 7669.0, 0.45]])\n",
         "call": "site_rates(sites)",
         "gold_call": "_oracle_site_rates(sites)"},
        # Boundary: forces either side of the hydrolysis transition band and inside it.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.64], [216.0, 7155.0, 0.70], [216.0, 7155.0, 0.7290]])\n",
         "call": "site_rates(sites)",
         "gold_call": "_oracle_site_rates(sites)"},
        # Edge: sites loaded past the point where the two stationary points merge.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.7305], [311.0, 6524.0, 0.95]])\n",
         "call": "site_rates(sites)",
         "gold_call": "_oracle_site_rates(sites)"},
        # Edge: a single weakened site (E_dis carries 0.94 twice).
        {"setup": setup + "sites = np.array([[251.0 * 0.94 * 0.94, 15954.0, 0.5103]])\n",
         "call": "site_rates(sites)",
         "gold_call": "_oracle_site_rates(sites)"},
        # Invalid: a non-positive dissociation energy.
        {"setup": invalid,
         "call": "run_model(np.array([[0.0, 7155.0, 0.45]]))",
         "gold_call": "run_gold(np.array([[0.0, 7155.0, 0.45]]))"},
        # Invalid: an unloaded bond.
        {"setup": invalid,
         "call": "run_model(np.array([[216.0, 7155.0, 0.0]]))",
         "gold_call": "run_gold(np.array([[216.0, 7155.0, 0.0]]))"},
        # Invalid: an empty site array.
        {"setup": invalid,
         "call": "run_model(np.zeros((0, 3)))",
         "gold_call": "run_gold(np.zeros((0, 3)))"},
    ]
