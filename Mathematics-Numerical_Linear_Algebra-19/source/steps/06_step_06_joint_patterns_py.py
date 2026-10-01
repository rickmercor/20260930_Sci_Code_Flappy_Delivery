"""
Intersect the column partitions at a common parameter pair and retain the distinct combined support patterns that meet the shared storage budget.

This is the benchmark's parameter-search stage, not an additional NRSAI update. A support combination is realizable only on the intersection of its column regions; storage counts active positions, not nonzero fitted values.




$$

\mathcal R=\bigcap_{k=0}^{2n-1}\mathcal R_k,\quad F=\sum_{k=0}^{2n-1}|J_k|\le K.

$$

Returns
-------
tuple of (representative_box, supports, total_size), or an empty tuple when no shared-budget pattern exists.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def joint_patterns(partitions: tuple, budget: int) -> tuple:
    """Intersect column regions and return feasible joint support patterns.

    Parameters
    ----------
    partitions : tuple
        An even, nonzero number 2*n of column partitions. Each partition is a
        nonempty tuple of (box, J), with boxes encoded as in column_regions
        and nonempty increasing integer supports in [0, n). Its boxes must
        not overlap, including at closed boundaries. The first n partitions
        belong to C=A, in column order; the next n belong to C=A.T.
    budget : int
        Nonnegative non-Boolean integer maximum for the sum of support sizes.

    Returns
    -------
    configurations : tuple
        Entries (representative_box, supports, total_size). supports contains
        2*n increasing index tuples in input order. Only nonempty common
        parameter intersections with total_size <= budget are retained.
        Return one representative per distinct supports, choosing the first
        numerical box in lexicographic flattened-interval order. Sort output
        by supports. Return an empty tuple if no configuration is feasible.
        Zero fitted values are not dropped from the active-position count.

    Raises
    ------
    ValueError
        If partitions is empty, has odd length, contains an empty partition,
        malformed/overlapping boxes, or invalid supports; or budget is not
        a nonnegative non-Boolean integer.
    """
    return ()

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_joint_patterns(partitions: tuple, budget: int) -> tuple:
    budget = _nr_integer(budget, 'budget', 0)
    if not isinstance(partitions, (tuple, list)) or not partitions or len(partitions) % 2:
        raise ValueError("partitions must contain a positive even number of columns")
    n = len(partitions) // 2
    decoded = []
    for part in partitions:
        if not isinstance(part, (tuple, list)) or not part:
            raise ValueError("each column partition must be nonempty")
        entries = []
        for entry in part:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("each region needs a box and a support")
            bb, J = _nr_box(entry[0]), _nr_indices(entry[1], n, 'J', nonempty=True)
            if any(_nr_box_intersect(bb, old[0]) is not None for old in entries):
                raise ValueError("boxes within a column partition must be disjoint")
            entries.append((bb, J))
        decoded.append(entries)
    # This lower bound uses supports of the remaining columns only; it cannot
    # discard a feasible common parameter choice.
    minima = [min(len(J) for _, J in part) for part in decoded]
    remaining = sum(minima[1:])
    cells = [(bb, (J,), len(J)) for bb, J in decoded[0] if len(J) + remaining <= budget]
    for index, part in enumerate(decoded[1:], start=1):
        remaining -= minima[index]
        updated = []
        for box, supports, cost in cells:
            for bb, J in part:
                new_cost = cost + len(J)
                if new_cost + remaining > budget:
                    continue
                common = _nr_box_intersect(box, bb)
                if common is not None:
                    updated.append((common, supports + (J,), new_cost))
        cells = updated
        if not cells:
            return ()
    unique = {}
    for bb, supports, cost in cells:
        old = unique.get(supports)
        if old is None or _nr_box_key(bb) < _nr_box_key(old[0]):
            unique[supports] = (bb, cost)
    return tuple((_nr_encode_box(unique[s][0]), s, unique[s][1]) for s in sorted(unique))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: common choices yield three configurations
        {
            "setup": """box = (((1, 10), (1, 5), 1, 1), ((1, 10), (3, 10), 1, 1))
a = (((1,10),(1,5),1,1), ((1,10),(1,5),1,1))
b = (((1,10),(1,5),1,1), ((1,5),(3,10),0,1))
c = (((1,10),(1,5),1,1), ((1,10),(1,5),1,0))
d = (((1,10),(1,5),1,1), ((1,5),(3,10),1,1))
partitions = (((a,(0,)), (b,(0,1))), ((box,(1,)),),
              ((box,(0,)),), ((c,(0,1)), (d,(1,))))

expected = (((((1, 10), (1, 5), 1, 1), ((1, 10), (1, 5), 1, 0)), ((0,), (1,), (0,), (0, 1)), 5),
 ((((1, 10), (1, 5), 1, 1), ((1, 5), (1, 5), 1, 1)), ((0,), (1,), (0,), (1,)), 4),
 ((((1, 10), (1, 5), 1, 1), ((1, 5), (3, 10), 0, 1)), ((0, 1), (1,), (0,), (1,)), 5))
""",
            "call": 'int(joint_patterns(partitions, 5) == expected)',
            "gold_call": 'int(_oracle_joint_patterns(partitions, 5) == expected)',
        },
        # boundary: the only feasible choice is a zero-width boundary
        {
            "setup": """box = (((1, 10), (1, 5), 1, 1), ((1, 10), (3, 10), 1, 1))
a = (((1,10),(1,5),1,1), ((1,10),(1,5),1,1))
b = (((1,10),(1,5),1,1), ((1,5),(3,10),0,1))
c = (((1,10),(1,5),1,1), ((1,10),(1,5),1,0))
d = (((1,10),(1,5),1,1), ((1,5),(3,10),1,1))
partitions = (((a,(0,)), (b,(0,1))), ((box,(1,)),),
              ((box,(0,)),), ((c,(0,1)), (d,(1,))))

expected = (((((1, 10), (1, 5), 1, 1), ((1, 5), (1, 5), 1, 1)), ((0,), (1,), (0,), (1,)), 4),)
""",
            "call": 'int(joint_patterns(partitions, 4) == expected)',
            "gold_call": 'int(_oracle_joint_patterns(partitions, 4) == expected)',
        },
        # edge: infeasible budget returns an empty tuple
        {
            "setup": """box = (((1, 10), (1, 5), 1, 1), ((1, 10), (3, 10), 1, 1))
a = (((1,10),(1,5),1,1), ((1,10),(1,5),1,1))
b = (((1,10),(1,5),1,1), ((1,5),(3,10),0,1))
c = (((1,10),(1,5),1,1), ((1,10),(1,5),1,0))
d = (((1,10),(1,5),1,1), ((1,5),(3,10),1,1))
partitions = (((a,(0,)), (b,(0,1))), ((box,(1,)),),
              ((box,(0,)),), ((c,(0,1)), (d,(1,))))

expected = ()
""",
            "call": 'int(joint_patterns(partitions, 3) == expected)',
            "gold_call": 'int(_oracle_joint_patterns(partitions, 3) == expected)',
        },
        # edge: identical supports in separate regions are scored once
        {
            "setup": """box = (((1, 10), (1, 5), 1, 1), ((1, 10), (3, 10), 1, 1))

a=(((1,10),(1,5),1,1),((1,10),(1,5),1,1))
b=(((1,10),(1,5),1,1),((1,5),(3,10),0,1))
partitions=(((a,(0,)),(b,(0,))), ((box,(1,)),), ((box,(0,)),), ((box,(1,)),))

expected = (((((1, 10), (1, 5), 1, 1), ((1, 10), (1, 5), 1, 1)), ((0,), (1,), (0,), (1,)), 4),)
""",
            "call": 'int(joint_patterns(partitions, 4) == expected)',
            "gold_call": 'int(_oracle_joint_patterns(partitions, 4) == expected)',
        },
        # invalid: overlapping regions within a column
        {
            "setup": """box = (((1, 10), (1, 5), 1, 1), ((1, 10), (3, 10), 1, 1))

partitions=(((box,(0,)),(box,(0,1))), ((box,(1,)),), ((box,(0,)),), ((box,(1,)),))

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: joint_patterns(partitions, 8))',
            "gold_call": '_capture_value_error(lambda: _oracle_joint_patterns(partitions, 8))',
        },
    ]
