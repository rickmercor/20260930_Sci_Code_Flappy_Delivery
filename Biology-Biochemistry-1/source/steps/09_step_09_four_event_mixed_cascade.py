"""
Step 9 - orchestrator: the graded four-event mixed cascade.

This final step runs the complete established pipeline. It first reduces the supplied ensemble extension records to the numeric site table, then computes the probability that the first event is hydrolysis and the following three events are homolyses while applying the stated relaxation and radical-weakening rules after each relevant break.

Returns
-------
p : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def four_event_mixed_cascade(records: "list[tuple]",
                             transfer_fraction: float,
                             weakening_fraction: float) -> float:
    """Run the complete record-to-cascade pipeline and return P(hyd, hom, hom, hom).

    Parameters
    ----------
    records : list of tuple
        Raw ensemble records in the Step 1 format. Each entry is
        (label, E_dis, k, instrument, replicas).
    transfer_fraction : float
        The network load-transfer fraction t, applied after every break.
    weakening_fraction : float
        The radical weakening fraction r, applied after every homolytic break.

    Returns
    -------
    float
        The four-event cascade probability in [0, 1].

    Raises
    ------
    ValueError
        If the raw records are invalid under Step 1; if fewer than four valid sites are
        supplied; if transfer_fraction is not finite or is negative; if
        weakening_fraction is not finite or lies outside [0, 1); or if a reached pool
        cannot satisfy the relaxation contract of Steps 6-8.
    """
    return p  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_four_event_mixed_cascade(records: "list[tuple]",
                                     transfer_fraction: float,
                                     weakening_fraction: float) -> float:
    """Reference implementation for four_event_mixed_cascade."""
    import numpy as np

    triples = _oracle_site_table(records)
    if triples.shape[0] < 4:
        raise ValueError("the four-event cascade requires at least four sites")

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

    r_hom, r_hyd = _oracle_site_rates(triples)
    w1 = _oracle_branch_probabilities(r_hyd, r_hom)
    cond = []
    for i in range(triples.shape[0]):
        pool = _oracle_apply_break(triples, i, "hyd", t, r)
        cond.append(_oracle_sequence_probability(
            pool, ("hom", "hom", "hom"), t, r))
    return float(np.sum(w1 * np.asarray(cond)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    graded = '''records = [
    ("disulfide S-S", 251.0, 15954.0, "A",
     [[0.005954, 0, 0.025799, 0.02825],
      [0.005954, 0, 0.02952, 0.030823, 0.031486, 0.031486],
      [0.005954, 0, 0.012594, 0, 0.050917, 0.055129]]),
    ("ester C-O", 280.0, 16082.0, "B",
     [[0.05889, 0, 0.3388],
      [0.05889, 0, 0.3388, 0.36639],
      [0.05889, 0, 0.1241, 0, 0.34556, 0.37352, 0.38075]]),
    ("glycosidic C-O", 230.0, 27566.0, "A",
     [[0.003409, 0, -0.003159, 0.017358, 0.018031],
      [0.003409, 0, 0.010065, 0.011192]]),
    ("peroxide O-O", 170.0, 73368.0, "B",
     [[0.01266, 0, 0.07328],
      [0.01266, 0, 0.07328, 0.07559, 0.07792],
      [0.01266, 0, 0.02608, 0, 0.07559, 0.08029]]),
    ("peptide C-N", 350.0, 22016.0, "A",
     [[0.004252, 0, 0.028469],
      [0.004252, 0, 0.029388, 0.030323, 0.031275]]),
    ("thioester C-S", 260.0, 45653.0, "B",
     [[0.02036, 0, 0.1517, 0.16009],
      [0.02036, 0, 0.15797, 0.16653],
      [0.02036, 0, 0.04198, 0, 0.16437, 0.1687, 0.1731]]),
]
'''
    invalid = setup + (
        "def run_model(x, t, r):\n"
        "    try:\n"
        "        four_event_mixed_cascade(x, t, r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(x, t, r):\n"
        "    try:\n"
        "        _oracle_four_event_mixed_cascade(x, t, r)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        {"setup": setup + graded,
         "call": "four_event_mixed_cascade(records, 0.045, 0.06)",
         "gold_call": "_oracle_four_event_mixed_cascade(records, 0.045, 0.06)"},
        {"setup": setup + graded,
         "call": "four_event_mixed_cascade(records[:4], 0.045, 0.06)",
         "gold_call": "_oracle_four_event_mixed_cascade(records[:4], 0.045, 0.06)"},
        {"setup": setup + graded,
         "call": "four_event_mixed_cascade(records[:4], 0.0, 0.0)",
         "gold_call": "_oracle_four_event_mixed_cascade(records[:4], 0.0, 0.0)"},
        {"setup": setup + graded,
         "call": "four_event_mixed_cascade(records, 0.045, 0.0)",
         "gold_call": "_oracle_four_event_mixed_cascade(records, 0.045, 0.0)"},
        {"setup": invalid + graded,
         "call": "run_model(records[:3], 0.045, 0.06)",
         "gold_call": "run_gold(records[:3], 0.045, 0.06)"},
        {"setup": invalid + graded,
         "call": "run_model(records, 0.045, 1.0)",
         "gold_call": "run_gold(records, 0.045, 1.0)"},
        {"setup": invalid + graded,
         "call": "run_model(records, -0.1, 0.06)",
         "gold_call": "run_gold(records, -0.1, 0.06)"},
    ]
