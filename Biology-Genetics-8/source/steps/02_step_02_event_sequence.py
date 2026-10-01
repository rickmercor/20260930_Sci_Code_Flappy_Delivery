"""
Read the ranking of the internal vertices as an ordered sequence of event types.

The ranking of internal vertices is what distinguishes this family of networks from unranked graph shapes: it records the order in which speciations and hybridisations occurred in time, running from the root towards the tips. That ordering is intrinsic to the coalescent and birth-death processes used to infer such histories, and it carries information about the tempo of reticulation that a comparison ignoring time would discard. Reading the vertex classification along the ranking yields the sequence of event types, which determines how the number of ancestral lineages rises and falls from one interval to the next.

Returns
-------
list of ints, one per event in rank order, 0 for a speciation and 1 for a hybridisation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def event_sequence(classification: dict) -> list[int]:
    '''List the event types of the network in rank order.
    Parameters
    ----------
    classification : dict
        The vertex classification as returned by the first step.
    Returns
    -------
    events : list[int]
        One entry per internal vertex, ordered by rank from the root, coded
        0 for a speciation and 1 for a hybridisation.
    '''
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_event_sequence(classification):
    hyb = set(classification["hybridisation"])
    internal = ([classification["root"]] + classification["speciation"]
                + classification["hybridisation"])
    ordered = sorted(internal, key=lambda v: int(v[1:]))
    return [1 if v in hyb else 0 for v in ordered]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """CA = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CB = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CT = {'root': 'v1', 'speciation': ['v2'], 'hybridisation': [], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 0, 'size': 3}
CO = {'root': 'v1', 'speciation': ['v2', 'v3'], 'hybridisation': ['v4'], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 1, 'size': 5}
""",
            "call": "event_sequence(CA)",
            "gold_call": "_oracle_event_sequence(CA)",
        },
        {
            "setup": """CA = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CB = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CT = {'root': 'v1', 'speciation': ['v2'], 'hybridisation': [], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 0, 'size': 3}
CO = {'root': 'v1', 'speciation': ['v2', 'v3'], 'hybridisation': ['v4'], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 1, 'size': 5}
""",
            "call": "event_sequence(CB)",
            "gold_call": "_oracle_event_sequence(CB)",
        },
        {
            "setup": """CA = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CB = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CT = {'root': 'v1', 'speciation': ['v2'], 'hybridisation': [], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 0, 'size': 3}
CO = {'root': 'v1', 'speciation': ['v2', 'v3'], 'hybridisation': ['v4'], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 1, 'size': 5}
""",
            "call": "event_sequence(CT)",
            "gold_call": "_oracle_event_sequence(CT)",
        },
        {
            "setup": """CA = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CB = {'root': 'v1', 'speciation': ['v2', 'v3', 'v4', 'v5'], 'hybridisation': ['v6', 'v7'], 'leaves': ['L1', 'L2', 'L3', 'L4'], 'n': 4, 'm': 2, 'size': 8}
CT = {'root': 'v1', 'speciation': ['v2'], 'hybridisation': [], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 0, 'size': 3}
CO = {'root': 'v1', 'speciation': ['v2', 'v3'], 'hybridisation': ['v4'], 'leaves': ['L1', 'L2', 'L3'], 'n': 3, 'm': 1, 'size': 5}
""",
            "call": "event_sequence(CO)",
            "gold_call": "_oracle_event_sequence(CO)",
        },
    ]
