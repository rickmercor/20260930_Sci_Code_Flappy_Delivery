"""
Run the whole pipeline: classify, encode, validate, weight, and report the distance.

The two networks are admitted to the comparison only if their encodings have the same size

n + 2m, since otherwise the entrywise difference is undefined and the events would first

have to be aligned; each network is then weighted by its own event times before the two

weighted encodings are compared.

Returns
-------
float, the time-weighted distance between the two networks, rounded to 6 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def timed_network_distance(edges_a: list[tuple[str, str]], times_a: list[float],
                           edges_b: list[tuple[str, str]], times_b: list[float]) -> float:
    '''Distance between two timed ranked networks given their edges and event times.
    Parameters
    ----------
    edges_a, edges_b : list[tuple[str, str]]
        Directed edges written ancestor to descendant. Internal vertices are named
        v1 upwards in rank order; leaf names carry no information.
    times_a, times_b : list[float]
        Strictly increasing event times u_0 to u_N of the respective networks.
    Returns
    -------
    distance : float
        The time-weighted distance, rounded to 6 decimal places.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_timed_network_distance(edges_a, times_a, edges_b, times_b):
    ca = _oracle_classify_vertices(edges_a)
    cb = _oracle_classify_vertices(edges_b)
    if ca["size"] != cb["size"]:
        raise ValueError("encodings differ in size; event alignment would be required")
    out = []
    for edges, c in ((edges_a, ca), (edges_b, cb)):
        F = _oracle_encoding_matrix(_oracle_lineage_sets(edges, c["size"], c["root"]))
        if not _oracle_validate_encoding(F, c["n"], c["m"])["valid"]:
            raise ValueError("network is not admitted to the encoding space")
        if not _oracle_diagonal_profile(F, _oracle_event_sequence(c))["consistent"]:
            raise ValueError("encoding diagonal disagrees with the event sequence")
        out.append(F)
    return _oracle_weighted_distance(out[0], _oracle_weight_matrix(times_a),
                                     out[1], _oracle_weight_matrix(times_b))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''EA = [("v1","v2"),("v1","v3"),("v2","v4"),("v2","v6"),("v3","v5"),("v3","v6"),
      ("v4","L1"),("v4","v7"),("v5","L2"),("v5","v7"),("v6","L3"),("v7","L4")]
EB = [("v1","v2"),("v1","v3"),("v2","L1"),("v2","L2"),("v3","L3"),("v3","v4"),
      ("v4","v5"),("v4","v6"),("v5","v6"),("v5","v7"),("v6","v7"),("v7","L4")]
UA = [0,1,2,3,4,5,10,11,12]
UB = [0,2,4,6,7,8,9,11,12]
ETREE = [("v1","v2"),("v1","L1"),("v2","L2"),("v2","L3")]
'''
    return [
        {   # normal: the answer to the task
            "setup": setup,
            "call": 'timed_network_distance(EA, UA, EB, UB)',
            "gold_call": '_oracle_timed_network_distance(EA, UA, EB, UB)',
        },
        {   # boundary: identical timed networks
            "setup": setup,
            "call": 'timed_network_distance(EA, UA, EA, UA)',
            "gold_call": '_oracle_timed_network_distance(EA, UA, EA, UA)',
        },
        {   # normal: one topology under two clocks, which the distance must separate
            "setup": setup,
            "call": 'timed_network_distance(EA, UA, EA, UB)',
            "gold_call": '_oracle_timed_network_distance(EA, UA, EA, UB)',
        },
        {   # edge: a hybridisation-free network, where the encoding reduces to a ranked tree
            "setup": setup,
            "call": 'timed_network_distance(ETREE, [0,1,2,4], ETREE, [0,2,3,4])',
            "gold_call": '_oracle_timed_network_distance(ETREE, [0,1,2,4], ETREE, [0,2,3,4])',
        },
    ]
