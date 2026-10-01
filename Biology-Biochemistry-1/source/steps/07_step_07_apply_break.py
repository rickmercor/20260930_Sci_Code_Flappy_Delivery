"""
Step 7 - applying one break: load redistribution and radical migration.

A break consumes its bond regardless of pathway, and the network then relaxes so the survivors carry more force. The rule composes across successive breaks: after a later break it acts on the forces the previous relaxation produced, not on the original ones. 

The two pathways differ in what they leave behind: a homolytic scission leaves a radical that attacks the surviving bond with the lowest rupture barrier on the relaxed network (earliest in table order on a tie) and weakens it, while a hydrolytic cleavage is closed-shell and leaves none. Weakening multiplies that bond's dissociation energy by (1 - r) and leaves force constants and forces untouched. 

This step performs the whole break: consume, relax, and weaken if the pathway was homolytic; and returns the pool the next draw sees.

Returns
-------
07  out : np.ndarray, shape (n-1, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_break(site_array: "np.ndarray",
                broken_index: int,
                pathway: str,
                transfer_fraction: float,
                weakening_fraction: float) -> "np.ndarray":
    """The surviving pool after one break, relaxed and (if homolytic) weakened.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F] at the current stage, in original table order. F holds the forces
        current at this break, after any previous relaxation.
    broken_index : int
        Row index (0-based) of the bond that breaks at this event, in the PRE-break array.
    pathway : str
        Either "hom" for a homolytic scission or "hyd" for a hydrolytic cleavage. Only a
        homolytic break leaves a radical.
    transfer_fraction : float
        The network's load-transfer fraction t, finite and non-negative, passed to the
        relaxation of step 6.
    weakening_fraction : float
        The network's radical weakening fraction r, in [0, 1). It is validated whichever
        pathway is given, and it changes nothing when the pathway is "hyd".

    Returns
    -------
    np.ndarray
        Shape (n-1, 3), the survivors in original table order, their forces relaxed and
        rounded to four decimals, and for a homolytic break the attacked row's E_dis
        multiplied by (1 - weakening_fraction) and rounded to four decimals. An empty
        (0, 3) array when the broken bond was the only bond in the pool.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, if any of E_dis,
        k or F is not positive, if pathway is neither "hom" nor "hyd", if broken_index is
        outside [0, n), if transfer_fraction is not finite or is negative, if
        weakening_fraction is not finite or leaves [0, 1), or if the relaxation of step 6
        has no solution for this pool.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_apply_break(site_array: "np.ndarray",
                        broken_index: int,
                        pathway: str,
                        transfer_fraction: float,
                        weakening_fraction: float) -> "np.ndarray":
    """Reference implementation for apply_break."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    if pathway not in ("hom", "hyd"):
        raise ValueError("pathway must be 'hom' or 'hyd'")
    r = float(weakening_fraction)
    if not np.isfinite(r):
        raise ValueError("weakening fraction must be finite")
    if r < 0.0 or r >= 1.0:
        raise ValueError("weakening fraction must lie in [0, 1)")

    # The relaxation validates the array, the broken index and the transfer fraction.
    forces = _oracle_relax_forces(arr, broken_index, transfer_fraction)

    keep = [j for j in range(arr.shape[0]) if j != int(broken_index)]
    out = arr[keep].copy()
    if out.shape[0] == 0:
        return out
    out[:, 2] = forces
    if pathway == "hom":
        r_hom, _ = _oracle_site_rates(out)
        # argmax keeps the earliest row when two survivors are equally fast.
        tgt = int(np.argmax(r_hom))
        out[tgt, 0] = float(np.round(out[tgt, 0] * (1.0 - r), 4))
    return out

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
        "def run_model(s, i, p, t, r):\n"
        "    try:\n"
        "        apply_break(s, i, p, t, r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(s, i, p, t, r):\n"
        "    try:\n"
        "        _oracle_apply_break(s, i, p, t, r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    twice = setup + three + (
        "def twice_model(s, i, p, t, r):\n"
        "    s2 = np.array(s, dtype=float)\n"
        "    apply_break(s2, i, p, t, r)\n"
        "    return np.asarray(apply_break(s2, i, p, t, r), dtype=float)\n"
        "def twice_gold(s, i, p, t, r):\n"
        "    s2 = np.array(s, dtype=float)\n"
        "    _oracle_apply_break(s2, i, p, t, r)\n"
        "    return np.asarray(_oracle_apply_break(s2, i, p, t, r), dtype=float)\n"
    )
    chain = setup + three + (
        "def chain_model(s):\n"
        "    p = apply_break(s, 1, 'hom', 0.045, 0.06)\n"
        "    return apply_break(p, 1, 'hom', 0.045, 0.06)\n"
        "def chain_gold(s):\n"
        "    p = _oracle_apply_break(s, 1, 'hom', 0.045, 0.06)\n"
        "    return _oracle_apply_break(p, 1, 'hom', 0.045, 0.06)\n"
    )
    return [
        # Normal: a homolytic break on the graded network - relax, then weaken the fastest.
        {"setup": setup + graded,
         "call": "apply_break(sites, 2, 'hom', 0.045, 0.06)",
         "gold_call": "_oracle_apply_break(sites, 2, 'hom', 0.045, 0.06)"},
        # Normal: the same break by the other pathway - relax only, no bond is weakened.
        {"setup": setup + graded,
         "call": "apply_break(sites, 2, 'hyd', 0.045, 0.06)",
         "gold_call": "_oracle_apply_break(sites, 2, 'hyd', 0.045, 0.06)"},
        # Normal: a homolytic break on a three-bond pool.
        {"setup": setup + three,
         "call": "apply_break(sites, 1, 'hom', 0.045, 0.06)",
         "gold_call": "_oracle_apply_break(sites, 1, 'hom', 0.045, 0.06)"},
        # Boundary: t = 0 leaves the forces alone but still consumes and still weakens.
        {"setup": setup + three,
         "call": "apply_break(sites, 0, 'hom', 0.0, 0.06)",
         "gold_call": "_oracle_apply_break(sites, 0, 'hom', 0.0, 0.06)"},
        # Boundary: r = 0 relaxes but leaves every dissociation energy untouched.
        {"setup": setup + three,
         "call": "apply_break(sites, 0, 'hom', 0.045, 0.0)",
         "gold_call": "_oracle_apply_break(sites, 0, 'hom', 0.045, 0.0)"},
        # Edge: a two-bond pool leaves exactly one survivor, which takes the radical.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.44], [235.0, 7669.0, 0.58]])\n",
         "call": "apply_break(sites, 1, 'hom', 0.045, 0.06)",
         "gold_call": "_oracle_apply_break(sites, 1, 'hom', 0.045, 0.06)"},
        # Edge: breaking the only bond leaves an empty pool.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.45]])\n",
         "call": "apply_break(sites, 0, 'hom', 0.045, 0.06)",
         "gold_call": "_oracle_apply_break(sites, 0, 'hom', 0.045, 0.06)"},
        # Edge: a repeat call on the same array must not depend on the first.
        {"setup": twice,
         "call": "twice_model(sites, 1, 'hom', 0.045, 0.06)",
         "gold_call": "twice_gold(sites, 1, 'hom', 0.045, 0.06)"},
        # Normal: two successive homolytic breaks feed the radical to the same survivor,
        # whose E_dis is weakened twice (216 -> 203.04 -> 190.8576).
        {"setup": chain,
         "call": "chain_model(sites)",
         "gold_call": "chain_gold(sites)"},
        # Invalid: an unrecognised pathway label.
        {"setup": invalid + three,
         "call": "run_model(sites, 0, 'both', 0.045, 0.06)",
         "gold_call": "run_gold(sites, 0, 'both', 0.045, 0.06)"},
        # Invalid: a weakening fraction of one would zero a dissociation energy.
        {"setup": invalid + three,
         "call": "run_model(sites, 0, 'hom', 0.045, 1.0)",
         "gold_call": "run_gold(sites, 0, 'hom', 0.045, 1.0)"},
        # Invalid: a broken index outside the array.
        {"setup": invalid + three,
         "call": "run_model(sites, 3, 'hom', 0.045, 0.06)",
         "gold_call": "run_gold(sites, 3, 'hom', 0.045, 0.06)"},
        # Invalid: a non-physical site row.
        {"setup": invalid + "sites = np.array([[0.0, 7155.0, 0.45], [235.0, 7669.0, 0.55]])\n",
         "call": "run_model(sites, 1, 'hom', 0.045, 0.06)",
         "gold_call": "run_gold(sites, 1, 'hom', 0.045, 0.06)"},
    ]
