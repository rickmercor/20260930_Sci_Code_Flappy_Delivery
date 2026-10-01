"""
Step 8 - the probability that the next events follow a prescribed pathway sequence.

The emulator's loop is the same at every draw: rate every bond still in the pool, select one event with probability proportional to its rate, apply the break, and repeat on what is left. 

A question of the form "what is the probability that the next events are, in order, a hydrolysis then two homolyses?" is therefore a nested weighted sum, one layer per event, and the layers are not interchangeable, because each conditions on a pool the previous break has already relaxed and, if that break was homolytic, weakened. 

The sequence is passed as a parameter, not fixed in advance: a layer's branch weights are the shares of whichever pathway that layer targets, the break applied between layers is of that same pathway, and the last element needs no break because nothing is drawn after it. 

This step evaluates such a question for an arbitrary sequence, terminating at a one-element sequence where the answer is the requested pathway's share of the whole pool.

Returns
-------
p : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sequence_probability(site_array: "np.ndarray",
                         pathways: "list[str]",
                         transfer_fraction: float,
                         weakening_fraction: float) -> float:
    """Probability that the next events follow the given pathway sequence, in order.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F] of the pool the first event of the sequence is drawn from, in
        original table order.
    pathways : sequence of str
        The ordered pathways, each "hom" or "hyd"; length 1 to n.
    transfer_fraction : float
        The network's load-transfer fraction t, passed to every break.
    weakening_fraction : float
        The network's radical weakening fraction r, passed to every break.

    Returns
    -------
    float
        The probability in [0, 1] that the next len(pathways) events are exactly this
        sequence of pathways.

    Raises
    ------
    ValueError
        If the site array is invalid (see step 07), if pathways is empty, if it is longer
        than the number of bonds in the pool, if any element is neither "hom" nor "hyd", or
        if either rule fraction is invalid.
    """
    return p  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sequence_probability(site_array: "np.ndarray",
                                 pathways: "list[str]",
                                 transfer_fraction: float,
                                 weakening_fraction: float) -> float:
    """Reference implementation for sequence_probability."""
    import numpy as np

    arr = np.asarray(site_array, dtype=float)
    seq = list(pathways)
    if len(seq) == 0:
        raise ValueError("pathway sequence must be non-empty")
    if any(s not in ("hom", "hyd") for s in seq):
        raise ValueError("each pathway must be 'hom' or 'hyd'")
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("site array must have shape (n, 3)")
    if len(seq) > arr.shape[0]:
        raise ValueError("sequence longer than the number of bonds in the pool")

    t = float(transfer_fraction)
    if not np.isfinite(t):
        raise ValueError("transfer fraction must be finite")
    if t < 0.0:
        raise ValueError("transfer fraction must be non-negative")
    r = float(weakening_fraction)
    if not np.isfinite(r):
        raise ValueError("weakening fraction must be finite")
    if r < 0.0 or r >= 1.0:
        raise ValueError("weakening fraction must lie in [0, 1)")

    r_hom, r_hyd = _oracle_site_rates(arr)
    target, other = (r_hom, r_hyd) if seq[0] == "hom" else (r_hyd, r_hom)
    if len(seq) == 1:
        return float(_oracle_selection_probability(target, other))
    w = _oracle_branch_probabilities(target, other)
    cond = []
    for i in range(arr.shape[0]):
        pool = _oracle_apply_break(arr, i, seq[0], t, r)
        cond.append(_oracle_sequence_probability(
            pool, seq[1:], t, r))
    return float(np.sum(w * np.asarray(cond)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    graded = ("sites = np.array([[251.0, 15954.0, 0.50], [280.0, 16082.0, 0.62],\n"
              "                  [230.0, 27566.0, 0.74], [170.0, 73368.0, 0.86],\n"
              "                  [350.0, 22016.0, 0.98], [260.0, 45653.0, 1.10]])\n")
    invalid = setup + (
        "def run_model(s, q, t, r):\n"
        "    try:\n"
        "        sequence_probability(s, q, t, r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(s, q, t, r):\n"
        "    try:\n"
        "        _oracle_sequence_probability(s, q, t, r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: the graded sequence on the graded network.
        {"setup": setup + graded,
         "call": "sequence_probability(sites, ('hyd', 'hom', 'hom', 'hom'), 0.045, 0.06)",
         "gold_call": "_oracle_sequence_probability(sites, ('hyd', 'hom', 'hom', 'hom'), 0.045, 0.06)"},
        # Normal: a different mixed sequence over the same network.
        {"setup": setup + graded,
         "call": "sequence_probability(sites, ('hom', 'hyd', 'hom'), 0.045, 0.06)",
         "gold_call": "_oracle_sequence_probability(sites, ('hom', 'hyd', 'hom'), 0.045, 0.06)"},
        # Normal: an all-homolysis sequence, where every break leaves a radical.
        {"setup": setup + graded,
         "call": "sequence_probability(sites, ('hom', 'hom', 'hom'), 0.045, 0.06)",
         "gold_call": "_oracle_sequence_probability(sites, ('hom', 'hom', 'hom'), 0.045, 0.06)"},
        # Boundary: a one-element sequence is just the pathway's share of the pool.
        {"setup": setup + graded,
         "call": "sequence_probability(sites, ('hyd',), 0.045, 0.06)",
         "gold_call": "_oracle_sequence_probability(sites, ('hyd',), 0.045, 0.06)"},
        # Boundary: a sequence as long as the pool, ending on its last bond.
        {"setup": setup + "sites = np.array([[216.0, 7155.0, 0.45], [235.0, 7669.0, 0.55], [182.0, 22175.0, 0.62]])\n",
         "call": "sequence_probability(sites, ('hyd', 'hom', 'hom'), 0.045, 0.06)",
         "gold_call": "_oracle_sequence_probability(sites, ('hyd', 'hom', 'hom'), 0.045, 0.06)"},
        # Edge: r = 0 and t = 0 switch both rules off, leaving pure re-rating.
        {"setup": setup + graded,
         "call": "sequence_probability(sites, ('hyd', 'hom', 'hom', 'hom'), 0.0, 0.0)",
         "gold_call": "_oracle_sequence_probability(sites, ('hyd', 'hom', 'hom', 'hom'), 0.0, 0.0)"},
        # Invalid: an empty sequence.
        {"setup": invalid + graded,
         "call": "run_model(sites, (), 0.045, 0.06)",
         "gold_call": "run_gold(sites, (), 0.045, 0.06)"},
        # Invalid: a sequence longer than the pool.
        {"setup": invalid + "sites = np.array([[216.0, 7155.0, 0.45], [235.0, 7669.0, 0.55]])\n",
         "call": "run_model(sites, ('hom', 'hom', 'hom'), 0.045, 0.06)",
         "gold_call": "run_gold(sites, ('hom', 'hom', 'hom'), 0.045, 0.06)"},
        # Invalid: an unrecognised pathway label.
        {"setup": invalid + graded,
         "call": "run_model(sites, ('hyd', 'photolysis'), 0.045, 0.06)",
         "gold_call": "run_gold(sites, ('hyd', 'photolysis'), 0.045, 0.06)"},
    ]
