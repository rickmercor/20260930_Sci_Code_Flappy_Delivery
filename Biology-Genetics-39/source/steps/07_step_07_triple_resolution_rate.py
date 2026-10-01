"""
Chain the sub-problem functions 01-06 end to end on the required and forbidden triple sets and return the fraction of three-leaf subsets that the resulting phylogenetic network resolves.

The construction runs in two stages. First the required triples are turned into least-common-ancestor constraints and closed with sub-problem 01, and sub-problem 02 decides whether that input relation is realizable at all; the criterion belongs to the input, so it is applied once, before any repair, and not to the saturated relations the loop produces. Then, as long as sub-problem 03 still reports some forbidden triple as forced, that triple joins the discharged list and sub-problem 01 is asked for the closure again, which collapses its three leaf pairs onto a single ancestor and removes it from every realizing DAG while leaving the required triples intact. Because the closure of a closure adds nothing, re-closing from the full discharged list gives the same relation as repairing the previous closure in place, and the loop must stop, since a triple treated once can never be forced again and at most one pass per forbidden triple is needed. The canonical DAG of the final constraint relation is then built with sub-problem 04 and completed into a phylogenetic network with sub-problem 05, and the construction succeeds exactly when sub-problem 06 shows that this network displays every required triple and no forbidden one; if it does not, no phylogenetic network can. The reported quantity is the number of displayed triples divided by the number n*(n-1)*(n-2)/6 of three-element leaf subsets, which measures the ancestral structure the data commit to rather than the structure they were told about.

Returns
-------
float: the fraction of the n*(n-1)*(n-2)/6 three-element leaf subsets whose triple the constructed phylogenetic network displays, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def triple_resolution_rate(n: int, req_triples: np.ndarray,
                           forb_triples: np.ndarray) -> float:
    """Resolution rate of the network built from required and forbidden triples.

    Parameters
    ----------
    n : int
        Number of leaves, labelled 0 to n-1.
    req_triples : np.ndarray
        Integer array of shape (t, 3); row (x, y, z) denotes the required
        rooted triple xy|z.
    forb_triples : np.ndarray
        Integer array of shape (f, 3) listing the forbidden rooted triples.

    Returns
    -------
    rate : float
        Fraction of the three-element leaf subsets whose triple is displayed
        by the constructed phylogenetic network, as a native Python float.

    Raises
    ------
    ValueError
        If no phylogenetic network agrees with the two triple sets.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-06 (``constraint_relations``, ``realizability_flags``,
    ``forced_triple_flags``, ``canonical_dag``, ``phylogenetic_network``,
    ``displayed_triple_flags``) and feed each returned value into the next,
    rather than reimplementing them. Raise ``ValueError`` when no phylogenetic
    network agrees with the two triple sets. Include every import your
    implementation needs (for example ``import numpy as np``) inside the
    function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_triple_resolution_rate(n: int, req_triples: np.ndarray,
                                   forb_triples: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 3:
        raise ValueError("at least three leaves are needed")
    req = np.asarray(req_triples, dtype=np.int64).reshape(-1, 3)
    forb = np.asarray(forb_triples, dtype=np.int64).reshape(-1, 3)

    # -- Sub-problems 01 and 02: constraints, closure and realizability.
    discharged = np.zeros((0, 3), dtype=np.int64)
    rels = np.asarray(_oracle_constraint_relations(n, req, discharged),
                      dtype=np.int64)
    if np.any(np.asarray(_oracle_realizability_flags(n, rels)) == 0):
        raise ValueError("the required constraints are not realizable")

    # -- Sub-problem 03 in a loop: discharge every forbidden triple forced.
    for _ in range(forb.shape[0] + 1):
        flags = np.asarray(_oracle_forced_triple_flags(n, rels[1], forb))
        forced = np.nonzero(flags.reshape(-1) != 0)[0]
        if forced.size == 0:
            break
        discharged = np.vstack(
            [discharged, forb[int(forced[0])].reshape(1, 3)])
        rels = np.asarray(_oracle_constraint_relations(n, req, discharged),
                          dtype=np.int64)
    else:
        raise ValueError("the repair loop did not settle")

    # -- Sub-problems 04 and 05: canonical DAG and phylogenetic network.
    dag = _oracle_canonical_dag(n, rels[1])
    net = np.asarray(_oracle_phylogenetic_network(dag, req, forb),
                     dtype=np.int64)

    # -- Sub-problem 06: the ancestor-based display test.
    if req.shape[0] and not np.all(
            np.asarray(_oracle_displayed_triple_flags(net, n, req)) != 0):
        raise ValueError("no phylogenetic network agrees with the triple sets")
    if forb.shape[0] and np.any(
            np.asarray(_oracle_displayed_triple_flags(net, n, forb)) != 0):
        raise ValueError("no phylogenetic network agrees with the triple sets")

    # -- Sweep the whole triple space and normalise by the 3-element subsets.
    space = []
    for x in range(n):
        for y in range(x + 1, n):
            for z in range(n):
                if z != x and z != y:
                    space.append((x, y, z))
    space = np.array(space, dtype=np.int64).reshape(-1, 3)
    shown = np.asarray(_oracle_displayed_triple_flags(net, n, space))
    subsets = n * (n - 1) * (n - 2) // 6
    return float(int(np.count_nonzero(shown)) / subsets)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
REQ = np.array([[5, 7, 1], [4, 8, 1], [2, 7, 1], [0, 4, 2],
                [0, 5, 8], [7, 8, 6], [1, 3, 2], [7, 8, 1],
                [4, 5, 2], [0, 6, 2], [5, 6, 1], [0, 7, 1]], dtype=np.int64)
FORB = np.array([[2, 5, 1], [5, 8, 1], [4, 5, 3], [0, 2, 1],
                 [2, 6, 1]], dtype=np.int64)
REQ5 = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 0]], dtype=np.int64)
FORB5 = np.array([[0, 2, 4], [0, 1, 3]], dtype=np.int64)
REQ4 = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
FORB4 = np.array([[2, 3, 0]], dtype=np.int64)
NONE = np.zeros((0, 3), dtype=np.int64)
REQ7 = np.array([[0, 1, 6], [2, 3, 6], [0, 2, 4], [4, 5, 0],
                 [1, 3, 5]], dtype=np.int64)
FORB7 = np.array([[0, 1, 4], [2, 3, 4]], dtype=np.int64)
REQ8 = np.array([[0, 1, 7], [2, 3, 7], [4, 5, 7], [0, 2, 6],
                 [1, 3, 6], [4, 6, 0]], dtype=np.int64)
FORB8 = np.array([[0, 3, 7], [1, 2, 6], [5, 6, 2]], dtype=np.int64)
"""
    return [
        # --- Integration: the nine-leaf configuration of the problem statement ---
        {
            "setup": setup,
            "call": "float(triple_resolution_rate(9, REQ, FORB))",
            "gold_call": "float(_oracle_triple_resolution_rate(9, REQ, FORB))",
        },
        # --- Integration: a five-leaf instance whose repair loop runs once ---
        {
            "setup": setup,
            "call": "float(triple_resolution_rate(5, REQ5, FORB5))",
            "gold_call": "float(_oracle_triple_resolution_rate(5, REQ5, FORB5))",
        },
        # --- Integration (boundary): four leaves and a single forbidden triple ---
        {
            "setup": setup,
            "call": "float(triple_resolution_rate(4, REQ4, FORB4))",
            "gold_call": "float(_oracle_triple_resolution_rate(4, REQ4, FORB4))",
        },
        # --- Integration (boundary): no constraints at all resolves nothing ---
        {
            "setup": setup,
            "call": "float(triple_resolution_rate(4, NONE, NONE))",
            "gold_call": "float(_oracle_triple_resolution_rate(4, NONE, NONE))",
        },
        # --- Integration: seven leaves, forbidden triples that are never forced ---
        {
            "setup": setup,
            "call": "float(triple_resolution_rate(7, REQ7, FORB7))",
            "gold_call": "float(_oracle_triple_resolution_rate(7, REQ7, FORB7))",
        },
        # --- Integration: eight leaves with three forbidden triples ---
        {
            "setup": setup,
            "call": "float(triple_resolution_rate(8, REQ8, FORB8))",
            "gold_call": "float(_oracle_triple_resolution_rate(8, REQ8, FORB8))",
        },
    ]
