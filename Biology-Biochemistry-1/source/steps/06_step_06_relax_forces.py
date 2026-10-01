"""
Step 6 - the network relaxation that follows a break.

A break removes one bond from the pool, whichever pathway broke it, and the network then relaxes. Before the break every survivor sits at the shifted well minimum of its own V_eff at the force it currently carries. The survivors are then stretched together by one common additional extension, each resisting along its own Morse bond potential, so each picks up load according to how its restoring force rises from where it sits. The common extension is the one that raises the survivors' total force by the transfer fraction t times the broken bond's current force, and it cannot carry any survivor past the largest force its potential sustains. The new forces are rounded to four decimal places, and a later relaxation starts from them rather than from the original forces; E_dis and k do not change.



This step returns the survivors' relaxed forces, one entry per surviving bond in table order.

Returns
-------
forces : np.ndarray, shape (n-1,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relax_forces(site_array: "np.ndarray",
                 broken_index: int,
                 transfer_fraction: float) -> "np.ndarray":
    """The surviving bonds' forces after the network relaxes, in nN.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F] at the current stage, in original table order. F holds the forces
        current at this break, after any previous relaxation.
    broken_index : int
        Row index (0-based) of the bond that breaks at this event, in the PRE-break array.
    transfer_fraction : float
        The network's load-transfer fraction t, finite and non-negative: relaxation leaves
        the survivors carrying t times the broken bond's force in addition to their own.

    Returns
    -------
    np.ndarray
        Shape (n-1,), float: the survivors' forces in nN, in original table order, each
        rounded to four decimal places. An empty array when the broken bond was the only
        bond in the pool.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, if any of E_dis,
        k or F is not positive, if broken_index is outside [0, n), if transfer_fraction is
        not finite or is negative, or if the survivors cannot take up the transferred load
        with every one of them still below the largest force its own potential can sustain.
    """
    return forces  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_relax_forces(site_array: "np.ndarray",
                         broken_index: int,
                         transfer_fraction: float) -> "np.ndarray":
    """Reference implementation for relax_forces."""
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
    t = float(transfer_fraction)
    if not np.isfinite(t):
        raise ValueError("transfer fraction must be finite")
    if t < 0.0:
        raise ValueError("transfer fraction must be non-negative")
    i = int(broken_index)
    n = arr.shape[0]
    if i < 0 or i >= n:
        raise ValueError("broken index out of range")

    keep = [j for j in range(n) if j != i]
    if not keep:
        return np.empty(0, dtype=float)

    E = arr[keep, 0]
    k = arr[keep, 1]
    beta = np.sqrt(k / (2.0 * E))
    g = 602.2 * arr[keep, 2]                     # kJ/mol/nm

    # Each survivor sits at its own shifted well minimum, u = exp(-beta x) = (1 + s) / 2.
    rad = 1.0 - 2.0 * g / (E * beta)
    if np.any(rad <= 0.0):
        raise ValueError("a survivor is already at or beyond the force it can sustain")
    u = (1.0 + np.sqrt(rad)) / 2.0

    def _total(d):
        """Total force the survivors carry when all are stretched by a further d."""
        v = u * np.exp(-beta * d)
        return float(np.sum(2.0 * E * beta * v * (1.0 - v)))

    target = float(np.sum(g)) + t * 602.2 * float(arr[i, 2])
    # Beyond this extension the first survivor passes its own force maximum, so the total
    # stops rising; the balance is monotone on [0, cap] and is bracketed there.
    cap = float(np.min(np.log(2.0 * u) / beta))
    if _total(cap) < target:
        raise ValueError("the survivors cannot take up the transferred load")
    lo, hi = 0.0, cap
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _total(mid) < target:
            lo = mid
        else:
            hi = mid
    v = u * np.exp(-beta * (0.5 * (lo + hi)))
    return np.round(2.0 * E * beta * v * (1.0 - v) / 602.2, 4)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    graded = ("sites = np.array([[251.0, 15954.0, 0.50], [280.0, 16082.0, 0.62],\n"
              "                  [230.0, 27566.0, 0.74], [170.0, 73368.0, 0.86],\n"
              "                  [350.0, 22016.0, 0.98], [260.0, 45653.0, 1.10]])\n")
    three = ("sites = np.array([[216.0, 7155.0, 0.45], [235.0, 7669.0, 0.55], "
             "[182.0, 22175.0, 0.62]])\n")
    invalid = setup + (
        "def run_model(s, i, t):\n"
        "    try:\n"
        "        relax_forces(s, i, t)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(s, i, t):\n"
        "    try:\n"
        "        _oracle_relax_forces(s, i, t)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: the stiffest bond of the graded network goes, and the rest take its load.
        {"setup": setup + graded,
         "call": "relax_forces(sites, 3, 0.045)",
         "gold_call": "_oracle_relax_forces(sites, 3, 0.045)"},
        # Normal: the same network, the least loaded bond going instead.
        {"setup": setup + graded,
         "call": "relax_forces(sites, 0, 0.045)",
         "gold_call": "_oracle_relax_forces(sites, 0, 0.045)"},
        # Normal: a three-bond pool spanning a wide stiffness range.
        {"setup": setup + three,
         "call": "relax_forces(sites, 1, 0.045)",
         "gold_call": "_oracle_relax_forces(sites, 1, 0.045)"},
        # Boundary: the last bond of the pool breaks.
        {"setup": setup + three,
         "call": "relax_forces(sites, 2, 0.045)",
         "gold_call": "_oracle_relax_forces(sites, 2, 0.045)"},
        # Boundary: t = 0 transfers nothing and leaves every force where it was.
        {"setup": setup + three,
         "call": "relax_forces(sites, 0, 0.0)",
         "gold_call": "_oracle_relax_forces(sites, 0, 0.0)"},
        # Boundary: a much larger transfer, well inside what the survivors can carry.
        {"setup": setup + three,
         "call": "relax_forces(sites, 0, 0.5)",
         "gold_call": "_oracle_relax_forces(sites, 0, 0.5)"},
        # Edge: a two-bond pool leaves a single survivor to take the whole transfer.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.44], [235.0, 7669.0, 0.58]])\n",
         "call": "relax_forces(sites, 1, 0.045)",
         "gold_call": "_oracle_relax_forces(sites, 1, 0.045)"},
        # Edge: breaking the only bond in the pool leaves no survivors.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.45]])\n",
         "call": "relax_forces(sites, 0, 0.045)",
         "gold_call": "_oracle_relax_forces(sites, 0, 0.045)"},
        # Invalid: a negative transfer fraction.
        {"setup": invalid + three,
         "call": "run_model(sites, 0, -0.1)",
         "gold_call": "run_gold(sites, 0, -0.1)"},
        # Invalid: a broken index outside the array.
        {"setup": invalid + three,
         "call": "run_model(sites, 3, 0.045)",
         "gold_call": "run_gold(sites, 3, 0.045)"},
        # Invalid: a survivor already loaded past the force its own potential can sustain.
        {"setup": invalid + "sites = np.array([[216.0, 7155.0, 0.80], [235.0, 7669.0, 0.55]])\n",
         "call": "run_model(sites, 1, 0.045)",
         "gold_call": "run_gold(sites, 1, 0.045)"},
        # Invalid: a transfer larger than the survivors can take up.
        {"setup": invalid + "sites = np.array([[216.0, 7155.0, 0.44], [235.0, 7669.0, 0.58]])\n",
         "call": "run_model(sites, 1, 40.0)",
         "gold_call": "run_gold(sites, 1, 40.0)"},
    ]
