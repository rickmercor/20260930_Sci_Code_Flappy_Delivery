"""
Compose every earlier step to rank all single substitutions of a long reference RNA and evaluate the selected double mutant through stable total-weight and pairing-event log endpoints. Orchestrator: Yes, it builds the log inside and outside tables, evaluates role-resolved single-replacement endpoints, ranks substitutions, evaluates ordered two-site total and event endpoints, and converts them to the requested share.

Pairing between two substituted sites is one route by which substitution effects become non-additive. The pairing role of a substituted base in the double mutant helps interpret the interaction.

Returns
-------
float: share of the double mutant's weight in which the strongest substitution's site is the 5' base of a pair.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_epistatic_pairing_share(
    length: int = 1500,
    seed: int = 20260912,
    minimum_gap: int = 200,
    rule_weights: tuple = (0.7, 0.44, 0.36, 0.24, 0.8),
    unpaired: tuple = ((0.28, 0.19, 0.27, 0.26), (0.24, 0.26, 0.21, 0.29)),
    pair_factors: tuple = ((0.0, 0.0, 0.0, 1.1), (0.0, 0.0, 1.9, 0.0), (0.0, 1.9, 0.0, 0.5), (1.1, 0.0, 0.7, 0.0)),
) -> float:
    """Return the 5' pairing share of the strongest substitution's site in the double mutant.

    The reference is the ``length``-base string whose base i is 'ACGU'[v_i]
    for v = numpy.random.default_rng(seed).integers(0, 4, length). Rank its
    single substitutions as in ``rank_single_substitutions`` with
    ``minimum_gap``, giving (p, c) and (q, d), and form the double mutant y
    carrying both. Evaluate the ordered finite endpoints in log space and
    return the summed weight of the parse trees of y in which position p is
    emitted as the 5' base of a pair production, divided by the ensemble
    weight of y. Grammar arguments follow ``compute_log_inside_table``
    (array-likes are accepted); the defaults reproduce the problem statement.

    Returns
    -------
    float
        Share in [0, 1].

    Raises
    ------
    ValueError
        If ``length`` is not an integer >= 2, ``seed`` is not a non-negative
        integer, or ``minimum_gap`` or the grammar is invalid as in the steps above.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_epistatic_pairing_share(
    length: int = 1500,
    seed: int = 20260912,
    minimum_gap: int = 200,
    rule_weights: tuple = (0.7, 0.44, 0.36, 0.24, 0.8),
    unpaired: tuple = ((0.28, 0.19, 0.27, 0.26), (0.24, 0.26, 0.21, 0.29)),
    pair_factors: tuple = ((0.0, 0.0, 0.0, 1.1), (0.0, 0.0, 1.9, 0.0), (0.0, 1.9, 0.0, 0.5), (1.1, 0.0, 0.7, 0.0)),
) -> float:
    import numpy as np
    for value, floor in ((length, 2), (seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < floor:
            raise ValueError("length must be an integer >= 2 and seed a non-negative integer")
    grammar = (rule_weights, unpaired, pair_factors)
    sequence = "".join("ACGU"[v] for v in np.random.default_rng(int(seed)).integers(0, 4, int(length)))
    log_inside = _oracle_compute_log_inside_table(sequence, *grammar)
    log_outside = _oracle_compute_log_outside_table(sequence, *grammar, log_inside)
    channels = _oracle_evaluate_single_replacement_channels(
        sequence, *grammar, log_inside, log_outside
    )
    ranked = _oracle_rank_single_substitutions(sequence, channels, minimum_gap)
    p, c, q, d = int(ranked[1]), "ACGU"[int(ranked[2])], int(ranked[4]), "ACGU"[int(ranked[5])]
    log_endpoints = _oracle_evaluate_ordered_double_endpoints(
        sequence, *grammar, log_inside, channels, p, c, q, d
    )
    metrics = _oracle_compute_epistatic_pairing_metrics(log_endpoints)
    return float(metrics[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    alt = (
        "import numpy as np\nw = (0.5, 0.3, 0.2, 0.2, 1.3); u = ((0.28, 0.19, 0.27, 0.26), (0.24, 0.26, 0.21, 0.29)); P = ((0.0, 0.0, 0.0, 1.1), (0.0, 0.0, 1.9, 0.0), (0.0, 1.9, 0.0, 0.9), (1.1, 0.0, 0.7, 0.0))\n"
        "def _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    return [
        {"setup": "import numpy as np\n", "call": "compute_epistatic_pairing_share(60, 3, 15)",
         "gold_call": "_oracle_compute_epistatic_pairing_share(60, 3, 15)"},
        {"setup": alt, "call": "compute_epistatic_pairing_share(45, 8, 10, w, u, P)",
         "gold_call": "_oracle_compute_epistatic_pairing_share(45, 8, 10, w, u, P)"},
        {"setup": "import numpy as np\n", "call": "compute_epistatic_pairing_share(12, 21, 3)",
         "gold_call": "_oracle_compute_epistatic_pairing_share(12, 21, 3)"},
        {"setup": alt, "call": "_status(lambda: compute_epistatic_pairing_share(1, 3, 1))",
         "gold_call": "_status(lambda: _oracle_compute_epistatic_pairing_share(1, 3, 1))"},
    ]
