"""
Compose every earlier step to obtain the expected G+C fraction of a sequence drawn uniformly from the separated sequences of a target structure at its smallest workable modulus.

The G+C content of the designs a certificate admits governs the stability and folding kinetics of the RNA that is finally synthesized, so it is a natural summary of a certified design family.

Returns
-------
float: expected G+C fraction of a uniformly drawn separated sequence at the smallest workable modulus.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_design_gc_content(
    structure: str = "((.....))((..((...((((.((......))))((((.....)))(((...))))))...)).))(((.(((((......))))))))",
    max_modulus: int = 9,
) -> float:
    """Return the expected G+C fraction of a uniform separated sequence at the smallest workable modulus.

    Build the loop table of ``structure`` (no minimum hairpin size), take
    its helix and motif profile, and find the smallest modulus ``m* >= 2``
    (up to ``max_modulus``) at which a separated sequence exists. Among the
    distinct sequences separated modulo ``m*`` (as in
    ``count_distinct_designs``), each equally likely, return the expected
    number of G and C nucleotides divided by the length ``n`` of
    ``structure``. The defaults reproduce the problem statement.

    Parameters
    ----------
    structure : str
        Dot-bracket target structure.
    max_modulus : int
        Largest modulus tried; at least 2.

    Returns
    -------
    float
        Expected G+C fraction of a uniformly drawn separated sequence.

    Raises
    ------
    ValueError
        If ``structure`` is not a valid dot-bracket string, if a loop
        carries a locally undesignable motif, if ``max_modulus`` is not an
        integer of at least 2, or if no modulus up to ``max_modulus`` admits
        a separated sequence.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_design_gc_content(
    structure: str = "((.....))((..((...((((.((......))))((((.....)))(((...))))))...)).))(((.(((((......))))))))",
    max_modulus: int = 9,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    table = _oracle_build_loop_tree(structure, 0)
    profile = _oracle_profile_helix_motifs(table)
    m_star = _oracle_find_minimal_modulus(table, profile, max_modulus)
    count, gc_total = (int(x) for x in _oracle_count_distinct_designs(table, m_star))
    if count <= 0:
        raise ValueError("no separated sequence at the smallest workable modulus")
    length = int(table[0, 2]) - 1  # row 0 stores n + 1 as its 3' position
    return float(np.float64(gc_total) / (np.float64(count) * length))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "estimate_design_gc_content('(...((.(......))).)(....)((.....))', 9)",
            "gold_call": "_oracle_estimate_design_gc_content('(...((.(......))).)(....)((.....))', 9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_design_gc_content('..(((((....)))(((...)))))..(....)..', 6)",
            "gold_call": "_oracle_estimate_design_gc_content('..(((((....)))(((...)))))..(....)..', 6)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_design_gc_content('((...))((.(....)))(...(((......))))', 9)",
            "gold_call": "_oracle_estimate_design_gc_content('((...))((.(....)))(...(((......))))', 9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_design_gc_content('..(((((......))((.....)))))(((.(((..(((((((......))((.....))))..)))...))))))..', 9)",
            "gold_call": "_oracle_estimate_design_gc_content('..(((((......))((.....)))))(((.(((..(((((((......))((.....))))..)))...))))))..', 9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_design_gc_content('((((((...)))((((....))))))).((((...))))', 3)",
            "gold_call": "_oracle_estimate_design_gc_content('((((((...)))((((....))))))).((((...))))', 3)",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_design_gc_content('(((...))((...))(...).)', 9))",
            "gold_call": "_status(lambda: _oracle_estimate_design_gc_content('(((...))((...))(...).)', 9))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_design_gc_content('(...((.(......))).)(....)((.....))', 4))",
            "gold_call": "_status(lambda: _oracle_estimate_design_gc_content('(...((.(......))).)(....)((.....))', 4))",
        },
    ]
