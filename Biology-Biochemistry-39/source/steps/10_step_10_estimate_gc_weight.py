"""
Compose every earlier step to find the GC weight at which the distinct separated designs of a target, at its smallest admissible modulus, reach a target expected G+C fraction.

Certified inverse-folding designs can be steered towards a desired composition by weighting them exponentially in their G+C content, provided each distinct design is counted exactly once.

Returns
-------
float: the GC Boltzmann weight at which the distinct separated designs reach the target expected G+C fraction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_gc_weight(
    dot_bracket: str = "((((((((....)))))...))(((..(......)...))))..(((((...(((...)))))(((.....))(((......)))(((...)))))))",
    target_fraction: float = 0.5,
    max_modulus: int = 8,
    tolerance: float = 1e-12,
) -> float:
    """Return the GC weight that gives the target expected G+C fraction.

    Build the loop tree of ``dot_bracket`` with ``build_loop_tree`` and
    screen it with ``screen_design_obstructions``. When the shortest helix
    has at least three pairs, limit the modulus search to
    ``min(max_modulus, 2)``; otherwise search up to ``max_modulus``. Take the
    modulus from ``find_minimal_modulus``, the distinct design counts from
    ``count_distinct_separated_designs`` at that modulus, and return the
    weight from ``calibrate_gc_weight`` with the bracket ``[-50, 50]``, the
    sequence length ``len(dot_bracket)``, ``target_fraction`` and
    ``tolerance``. As a consistency check, ``evaluate_gc_ensemble`` with the
    ``tabulate_assignment_counts`` table at that modulus must report between
    1 and ``2**modulus`` proposals per accepted design at the returned
    weight, within a relative ``1e-12``. The defaults reproduce the problem
    statement.

    Parameters
    ----------
    dot_bracket : str
        Target secondary structure.
    target_fraction : float
        Target expected G+C fraction, strictly between 0 and 1.
    max_modulus : int
        Largest modulus to try, from 1 to 8.
    tolerance : float
        Final bracket width of the weight search, from ``1e-14`` to
        ``1e-6`` inclusive.

    Returns
    -------
    float
        The calibrated GC weight.

    Raises
    ------
    ValueError
        If ``max_modulus`` is not an integer from 1 to 8 (booleans are
        rejected), if the target has a five-pair loop or a three-pair loop
        with an unpaired position, if the proposals check fails, or if any
        earlier step rejects its input.
    """
    return weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_gc_weight(
    dot_bracket: str = "((((((((....)))))...))(((..(......)...))))..(((((...(((...)))))(((.....))(((......)))(((...)))))))",
    target_fraction: float = 0.5,
    max_modulus: int = 8,
    tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    if isinstance(max_modulus, bool) or not isinstance(max_modulus, (int, np.integer)):
        raise ValueError("max_modulus must be an integer from 1 to 8")
    if not 1 <= int(max_modulus) <= 8:
        raise ValueError("max_modulus must be an integer from 1 to 8")
    tree = _oracle_build_loop_tree(dot_bracket)
    five_pair, three_pair_unpaired, shortest_helix = (
        int(value) for value in _oracle_screen_design_obstructions(tree)
    )
    if five_pair or three_pair_unpaired:
        raise ValueError("the target contains a loop motif that no sequence can design")
    # Helices of three or more pairs guarantee a modulus-2 design.
    limit = min(int(max_modulus), 2) if shortest_helix >= 3 else int(max_modulus)
    modulus = _oracle_find_minimal_modulus(tree, limit)
    distinct = _oracle_count_distinct_separated_designs(tree, modulus)
    length = len(dot_bracket)
    weight = _oracle_calibrate_gc_weight(distinct, length, target_fraction, -50.0, 50.0, tolerance)
    table = _oracle_tabulate_assignment_counts(tree, modulus)
    proposals = float(_oracle_evaluate_gc_ensemble(distinct, table, length, weight)[1])
    if not 1.0 - 1e-12 <= proposals <= 2 ** modulus * (1.0 + 1e-12):
        raise ValueError("the assignment table and the distinct counts are inconsistent")
    return float(weight)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests over distinct targets and compositions."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        return float(fn())\n"
        "    except ValueError:\n"
        "        return -100.0\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "estimate_gc_weight()",
            "gold_call": "_oracle_estimate_gc_weight()",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_gc_weight('(((((....)))(((...))))).((((....))))', 0.42, 8, 1e-12)",
            "gold_call": "_oracle_estimate_gc_weight('(((((....)))(((...))))).((((....))))', 0.42, 8, 1e-12)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_gc_weight('(((((((....)))..)))((((...)))))..((((....))))', 0.47, 8, 1e-12)",
            "gold_call": "_oracle_estimate_gc_weight('(((((((....)))..)))((((...)))))..((((....))))', 0.47, 8, 1e-12)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_gc_weight('((((((..((...))..))((....))))))', 0.45, 8, 1e-12)",
            "gold_call": "_oracle_estimate_gc_weight('((((((..((...))..))((....))))))', 0.45, 8, 1e-12)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_gc_weight('((((((...)))(((....)))))).((((...))))', 0.5, 8, 1e-12)",
            "gold_call": "_oracle_estimate_gc_weight('((((((...)))(((....)))))).((((...))))', 0.5, 8, 1e-12)",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_gc_weight('(((..((...))((...)))))', 0.4, 8, 1e-12))",
            "gold_call": "_status(lambda: _oracle_estimate_gc_weight('(((..((...))((...)))))', 0.4, 8, 1e-12))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_gc_weight('((((((..((...))..))((....))))))', 0.95, 8, 1e-12))",
            "gold_call": "_status(lambda: _oracle_estimate_gc_weight('((((((..((...))..))((....))))))', 0.95, 8, 1e-12))",
        },
    ]
