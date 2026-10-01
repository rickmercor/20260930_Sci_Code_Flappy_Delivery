"""
Return the weighted L1 distance after jointly refining the two ranked timed encodings to compatible event coordinates.

The final comparison composes lineage counting, F-matrix construction, event-sign recovery, joint matrix-and-time refinement, time weighting, and the weighted L1 norm. The two clocks remain network-specific. The common refined dimension is determined by event correspondence and may exceed both original dimensions.

Returns
-------
float - the weighted L1 distance between the two networks
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def network_distance(edges_a, size_a, times_a, edges_b, size_b, times_b):
    """Compute the weighted L1 distance between two ranked networks.
 
    The calculation constructs each F-matrix, aligns event sequences in
    root-to-tip order, inserts any required artificial events, builds the two
    time-weight matrices, and sums the lower-triangular weighted differences.
 
    Parameters
    ----------
    edges_a, edges_b : list of tuple of str
        Edge lists for the two networks. The root is written ``"*"``;
        internal vertices are ``"v<number>"`` and leaves are ``"L<number>"``.
    size_a, size_b : int
        Numbers of ranked intervals in the two networks.
    times_a, times_b : list of float
        Boundary-time vectors ``[root, one time per real event, tip]``,
        ordered from root to tip.
 
    Returns
    -------
    float
        The weighted L1 distance between the reconciled F-matrices.
 
    Raises
    ------
    ValueError
        If either network encoding, its ranked-interval count, or its event
        times are invalid; or if the two reconciled matrices cannot be given
        a common aligned dimension.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_network_distance(edges_a, size_a, times_a, edges_b, size_b, times_b):
    counts_a = _oracle_lineage_counts(edges_a, size_a)
    counts_b = _oracle_lineage_counts(edges_b, size_b)
    signs_a = _oracle_event_signs(counts_a)
    signs_b = _oracle_event_signs(counts_b)
    matrix_a = _oracle_count_matrix(edges_a, size_a)
    matrix_b = _oracle_count_matrix(edges_b, size_b)
    if ([matrix_a[i][i] for i in range(size_a)] != counts_a
            or [matrix_b[i][i] for i in range(size_b)] != counts_b):
        raise ValueError("F-matrix diagonals and lineage counts disagree")
    refined_a, clock_a, refined_b, clock_b = _oracle_reconcile_timed_matrices(
        signs_a, matrix_a, times_a, signs_b, matrix_b, times_b)
    weights_a = _oracle_weight_matrix(clock_a)
    weights_b = _oracle_weight_matrix(clock_b)
    return _oracle_weighted_l1(refined_a, weights_a, refined_b, weights_b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    A = ("ea = [('*','v1'),('v1','v2'),('v1','v3'),('v2','v3'),('v2','v6'),"
         "('v3','v4'),('v4','L1'),('v4','v5'),('v5','L2'),('v5','L3'),"
         "('v6','L4'),('v6','L5')]\nsa = 7\n"
         "ta = [10.0, 8.5, 6.5, 6.0, 5.5, 3.0, 1.5, 1.0]\n"
         "eb = [('*','v1'),('v1','v2'),('v1','v3'),('v2','v3'),('v2','v5'),"
         "('v3','v4'),('v4','v5'),('v4','v7'),('v5','v6'),('v6','L1'),"
         "('v6','v8'),('v7','L2'),('v7','L3'),('v8','L4'),('v8','L5')]\nsb = 9\n"
         "tb = [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, 5.0, 4.5, 3.0, 1.0]")
 
    B = ("ea = [('*','v1'),('v1','v2'),('v1','L1'),('v2','L2'),('v2','L3')]\nsa = 3\n"
         "ta = [6.0, 4.0, 2.0, 1.0]\n"
         "eb = [('*','v1'),('v1','v2'),('v1','v3'),('v2','v3'),('v2','v4'),"
         "('v3','L1'),('v4','L2'),('v4','L3')]\nsb = 5\n"
         "tb = [8.0, 7.0, 5.0, 4.0, 2.0, 1.0]")
 
    C = ("ea = [('*','v1'),('v1','v2'),('v1','L1'),('v2','L2'),('v2','L3')]\nsa = 3\n"
         "ta = [6.0, 4.0, 2.0, 1.0]\n"
         "eb = [('*','v1'),('v1','v2'),('v1','L1'),('v2','L2'),('v2','L3')]\nsb = 3\n"
         "tb = [9.0, 5.0, 3.0, 1.0]")
 
    D = ("ea = [('*','v1'),('v1','v2'),('v1','v3'),('v2','v4'),('v3','v4'),"
         "('v2','L1'),('v3','L2'),('v4','L3')]\nsa = 5\n"
         "ta = [8.0, 7.0, 6.0, 5.0, 4.0, 1.0]\n"
         "eb = [('*','v1'),('v1','v2'),('v1','v3'),('v2','v4'),('v3','v4'),"
         "('v2','v5'),('v3','v5'),('v4','v6'),('v5','L1'),('v6','L2'),('v6','L3')]\nsb = 7\n"
         "tb = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]")
 
    return [
        {
            "setup": A,
            "call": "network_distance(ea, sa, ta, eb, sb, tb)",
            "gold_call": "_oracle_network_distance(ea, sa, ta, eb, sb, tb)",
        },
        {
            "setup": B,
            "call": "network_distance(ea, sa, ta, eb, sb, tb)",
            "gold_call": "_oracle_network_distance(ea, sa, ta, eb, sb, tb)",
        },
        {
            "setup": C,
            "call": "network_distance(ea, sa, ta, eb, sb, tb)",
            "gold_call": "_oracle_network_distance(ea, sa, ta, eb, sb, tb)",
        },
        {
            "setup": D,
            "call": "network_distance(ea, sa, ta, eb, sb, tb)",
            "gold_call": "39.5",
        },
    ]
