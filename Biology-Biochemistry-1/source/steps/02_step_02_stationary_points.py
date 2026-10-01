"""
Step 2 - the two stationary points of the force-shifted bond potential.

The force-shifted Morse potential V_eff(x) = E_dis (1 - exp(-beta x))^2 - F_c x, with beta = sqrt(k / (2 E_dis)) and the force converted from nN to energy per unit length (1 nN = 602.2 kJ mol^-1 nm^-1), has two stationary points under load: a shifted well minimum, where the bond restoring force balances the applied force, and a barrier top beyond it. 

As the force increases the two approach each other; at the closing force they merge and the barrier vanishes. This step returns both extensions for every site; the rate step measures the barrier between them, and the relaxation step stretches each survivor from its minimum.

Returns
-------
points : np.ndarray, shape (n, 2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stationary_points(site_array: "np.ndarray") -> "np.ndarray":
    """The shifted well minimum and the barrier top of every loaded site.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F]: dissociation energy in kJ/mol (positive, current value
        including any weakening), force constant in kJ/mol/nm^2 (positive), pulling force
        in nN (positive).

    Returns
    -------
    np.ndarray
        Shape (n, 2), float, in site-array order: column 0 is the extension in nm of the
        shifted well minimum of V_eff, column 1 the extension in nm of its barrier top.
        Where the force is at or above the largest the bond's own potential can sustain the
        two stationary points have merged, and both columns hold that one merged extension.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, or if any of
        E_dis, k or F is not positive.
    """
    return points  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_stationary_points(site_array: "np.ndarray") -> "np.ndarray":
    """Reference implementation for stationary_points."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("site array must have shape (n, 3)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("site entries must be finite")
    if np.any(arr[:, 0] <= 0.0) or np.any(arr[:, 1] <= 0.0):
        raise ValueError("E_dis and k must be positive")
    if np.any(arr[:, 2] <= 0.0):
        raise ValueError("F must be positive")

    out = np.empty((arr.shape[0], 2), dtype=float)
    for j, (E, k, F) in enumerate(arr):
        beta = np.sqrt(k / (2.0 * E))
        g = 602.2 * F
        # dV_eff/dx = 0 is a quadratic in u = exp(-beta x): u^2 - u + g/(2 E beta) = 0.
        rad = 1.0 - 2.0 * g / (E * beta)
        if rad <= 0.0:
            x = np.log(2.0) / beta          # the merged double root, u = 1/2
            out[j, 0] = x
            out[j, 1] = x
            continue
        s = np.sqrt(rad)
        u_min = (1.0 + s) / 2.0             # larger root: the shifted well minimum
        u_max = (g / (E * beta)) / (1.0 + s)  # smaller root, written without cancellation
        out[j, 0] = -np.log(u_min) / beta
        out[j, 1] = -np.log(u_max) / beta
    return out

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
        "        stationary_points(s)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(s):\n"
        "    try:\n"
        "        _oracle_stationary_points(s)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: the graded six-site network.
        {"setup": setup + graded,
         "call": "stationary_points(sites)",
         "gold_call": "_oracle_stationary_points(sites)"},
        # Normal: a lightly loaded pair, where the barrier top sits far out.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.06], [235.0, 7669.0, 0.45]])\n",
         "call": "stationary_points(sites)",
         "gold_call": "_oracle_stationary_points(sites)"},
        # Boundary: a site just below the force at which the two points merge.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.7290]])\n",
         "call": "stationary_points(sites)",
         "gold_call": "_oracle_stationary_points(sites)"},
        # Edge: sites at and beyond the merge, where both columns hold one extension.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.7305], [311.0, 6524.0, 0.95]])\n",
         "call": "stationary_points(sites)",
         "gold_call": "_oracle_stationary_points(sites)"},
        # Edge: a weakened site, whose E_dis carries 0.94 twice.
        {"setup": setup + "sites = np.array([[251.0 * 0.94 * 0.94, 15954.0, 0.5103]])\n",
         "call": "stationary_points(sites)",
         "gold_call": "_oracle_stationary_points(sites)"},
        # Invalid: a non-positive dissociation energy.
        {"setup": invalid,
         "call": "run_model(np.array([[0.0, 7155.0, 0.45]]))",
         "gold_call": "run_gold(np.array([[0.0, 7155.0, 0.45]]))"},
        # Invalid: an unloaded bond has no barrier top at finite extension.
        {"setup": invalid,
         "call": "run_model(np.array([[216.0, 7155.0, 0.0]]))",
         "gold_call": "run_gold(np.array([[216.0, 7155.0, 0.0]]))"},
        # Invalid: an empty site array.
        {"setup": invalid,
         "call": "run_model(np.zeros((0, 3)))",
         "gold_call": "run_gold(np.zeros((0, 3)))"},
    ]
