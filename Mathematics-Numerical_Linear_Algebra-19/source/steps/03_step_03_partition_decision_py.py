"""
Partition a squared-tolerance rectangle into the next NRSAI decisions at one fixed residual state.

The column fit is independent of the tolerances at a fixed support. Its squared residual norm and squared component ratios therefore locate exact stopping and selection boundaries, including their equality cases.






Let eta = epsilon^2, xi = delta^2, and beta = sum(r_i^2). The column stops when eta >= beta. Otherwise, an unused candidate row i is eligible when r_i^2 >= xi * beta. Equality is included in both tests.

Returns
-------
tuple of (subbox, selected_indices); bounds are exact rational pairs and selected indices are integers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def partition_decision(
    residual: tuple,
    I: tuple[int, ...],
    used: tuple[int, ...],
    box: tuple,
    c: int = 2,
) -> tuple:
    """Split a parameter box into exact next-decision regions.

    Parameters
    ----------
    residual : tuple
        Nonempty full residual; each entry is a reduced integer rational pair.
    I, used : tuple[int, ...]
        Increasing distinct indices in [0, n); I is nonempty and used is a
        subset of I. Residual entries outside I must be zero.
    box : tuple
        (eta_interval, xi_interval) for eta=epsilon**2 and xi=delta**2.
        An interval is (low_pair, high_pair, low_closed, high_closed).
        Pairs are reduced integer rationals; closure flags are integers 0 or 1.
        Require 0 <= eta_low <= eta_high and 0 < xi_low <= xi_high <= 1.
        A singleton is allowed only when both ends are closed.
    c : int
        Positive non-Boolean integer cap on selected rows.

    Returns
    -------
    decisions : tuple
        Entries are (subbox, selected_indices). Empty selected_indices means
        terminate. Otherwise rows are ranked by decreasing residual magnitude,
        then increasing global index. The full residual norm is used.
        Regions are disjoint and cover box exactly. Order: tolerance stop,
        no-eligible stop, then selections of sizes 1 through min(c, available),
        omitting empty regions. Equality stops in eta and is eligible in xi.

    Raises
    ------
    ValueError
        If rational encodings, indices, closure flags, box bounds, or c are
        invalid; if I is empty, used is not a subset of I, the residual is
        empty, or a residual outside I is nonzero.
    """
    return ()

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from fractions import Fraction

import numpy as np


def _nr_interval(value):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("intervals need two rational endpoints and two closure flags")
    lo, hi = _nr_fraction(value[0]), _nr_fraction(value[1])
    lc, hc = (_nr_integer(value[i], 'closure flag', 0, 1) for i in (2, 3))
    if lo > hi or (lo == hi and not (lc and hc)):
        raise ValueError("interval is empty")
    return lo, hi, lc, hc


def _nr_box(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("box must have eta and xi intervals")
    eta, xi = _nr_interval(value[0]), _nr_interval(value[1])
    if eta[0] < 0 or xi[0] <= 0 or xi[1] > 1:
        raise ValueError("box requires eta >= 0 and 0 < xi <= 1")
    return eta, xi


def _nr_encode_box(box):
    return tuple((_nr_pair(b[0]), _nr_pair(b[1]), int(b[2]), int(b[3])) for b in box)


def _nr_intersect(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    lc = (a[2] if lo == a[0] else 1) and (b[2] if lo == b[0] else 1)
    hc = (a[3] if hi == a[1] else 1) and (b[3] if hi == b[1] else 1)
    if lo > hi or (lo == hi and not (lc and hc)):
        return None
    return lo, hi, int(lc), int(hc)


def _nr_box_intersect(a, b):
    eta = _nr_intersect(a[0], b[0])
    if eta is None:
        return None
    xi = _nr_intersect(a[1], b[1])
    return None if xi is None else (eta, xi)


def _nr_box_key(box):
    return tuple(x for interval in box for x in interval)


def _oracle_partition_decision(
    residual: tuple,
    I: tuple[int, ...],
    used: tuple[int, ...],
    box: tuple,
    c: int = 2,
) -> tuple:
    if not isinstance(residual, (tuple, list)) or not residual:
        raise ValueError("residual must be a nonempty tuple or list")
    r = tuple(_nr_fraction(v) for v in residual)
    I = _nr_indices(I, len(r), 'I', nonempty=True)
    used = _nr_indices(used, len(r), 'used')
    if not set(used).issubset(I) or any(r[i] != 0 for i in range(len(r)) if i not in I):
        raise ValueError("history or residual is inconsistent with I")
    c = _nr_integer(c, 'c', 1)
    eta, xi = _nr_box(box)
    beta = sum((v * v for v in r), Fraction(0))
    result = []
    stop = _nr_intersect(eta, (beta, max(beta, eta[1]), 1, 1))
    if stop is not None:
        result.append((_nr_encode_box((stop, xi)), ()))
    continuing = _nr_intersect(eta, (min(eta[0], beta), beta, 1, 0))
    if continuing is None:
        return tuple(result)
    candidates = sorted((i for i in I if i not in used), key=lambda i: (-r[i] * r[i], i))[:c]
    if not candidates:
        result.append((_nr_encode_box((continuing, xi)), ()))
        return tuple(result)
    ratios = [r[i] * r[i] / beta for i in candidates]
    no_row = _nr_intersect(xi, (ratios[0], max(xi[1], ratios[0]), 0, 1))
    if no_row is not None:
        result.append((_nr_encode_box((continuing, no_row)), ()))
    for count in range(1, len(candidates) + 1):
        low = ratios[count] if count < len(candidates) else Fraction(0)
        region = _nr_intersect(xi, (low, ratios[count - 1], int(count == len(candidates)), 1))
        if region is not None:
            result.append((_nr_encode_box((continuing, region)), tuple(candidates[:count])))
    return tuple(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: tolerance, no-selection, one-row and two-row regions
        {
            "setup": """residual=((3,1),(4,1),(0,1))
I=(0,1,2)
used=()
box=(((1,1),(30,1),1,1),((1,10),(9,10),1,1))

expected = (((((25, 1), (30, 1), 1, 1), ((1, 10), (9, 10), 1, 1)), ()),
 ((((1, 1), (25, 1), 1, 0), ((16, 25), (9, 10), 0, 1)), ()),
 ((((1, 1), (25, 1), 1, 0), ((9, 25), (16, 25), 0, 1)), (1,)),
 ((((1, 1), (25, 1), 1, 0), ((1, 10), (9, 25), 1, 1)), (1, 0)))
""",
            "call": 'int(partition_decision(residual, I, used, box, 2) == expected)',
            "gold_call": 'int(_oracle_partition_decision(residual, I, used, box, 2) == expected)',
        },
        # boundary: equality belongs to the tolerance stop
        {
            "setup": """residual=((3,1),(4,1))
I=(0,1)
used=()
box=(((25,1),(25,1),1,1),((1,10),(9,10),1,1))

expected = (((((25, 1), (25, 1), 1, 1), ((1, 10), (9, 10), 1, 1)), ()),)
""",
            "call": 'int(partition_decision(residual, I, used, box, 2) == expected)',
            "gold_call": 'int(_oracle_partition_decision(residual, I, used, box, 2) == expected)',
        },
        # boundary: equality belongs to relative eligibility
        {
            "setup": """residual=((3,1),(4,1))
I=(0,1)
used=()
box=(((1,1),(1,1),1,1),((16,25),(16,25),1,1))

expected = (((((1, 1), (1, 1), 1, 1), ((16, 25), (16, 25), 1, 1)), (1,)),)
""",
            "call": 'int(partition_decision(residual, I, used, box, 2) == expected)',
            "gold_call": 'int(_oracle_partition_decision(residual, I, used, box, 2) == expected)',
        },
        # edge: full norm includes used entries
        {
            "setup": """residual=((1,1),(-1,1),(1,1))
I=(0,1,2)
used=(0,)
box=(((0,1),(0,1),1,1),((2,5),(2,5),1,1))

expected = (((((0, 1), (0, 1), 1, 1), ((2, 5), (2, 5), 1, 1)), ()),)
""",
            "call": 'int(partition_decision(residual, I, used, box, 2) == expected)',
            "gold_call": 'int(_oracle_partition_decision(residual, I, used, box, 2) == expected)',
        },
        # edge: equal magnitudes use global index order
        {
            "setup": """residual=((1,1),(-1,1),(1,1))
I=(0,1,2)
used=(0,)
box=(((0,1),(0,1),1,1),((1,3),(1,3),1,1))

expected = (((((0, 1), (0, 1), 1, 1), ((1, 3), (1, 3), 1, 1)), (1, 2)),)
""",
            "call": 'int(partition_decision(residual, I, used, box, 2) == expected)',
            "gold_call": 'int(_oracle_partition_decision(residual, I, used, box, 2) == expected)',
        },
        # boundary: zero residual does not divide by zero
        {
            "setup": """residual=((0,1),)
I=(0,)
used=()
box=(((0,1),(1,1),1,1),((1,4),(1,2),1,1))

expected = (((((0, 1), (1, 1), 1, 1), ((1, 4), (1, 2), 1, 1)), ()),)
""",
            "call": 'int(partition_decision(residual, I, used, box, 2) == expected)',
            "gold_call": 'int(_oracle_partition_decision(residual, I, used, box, 2) == expected)',
        },
        # edge: all candidate rows have already been used
        {
            "setup": """residual=((1,1),(2,1))
I=(0,1)
used=(0,1)
box=(((0,1),(0,1),1,1),((1,4),(1,2),1,1))

expected = (((((0, 1), (0, 1), 1, 1), ((1, 4), (1, 2), 1, 1)), ()),)
""",
            "call": 'int(partition_decision(residual, I, used, box, 2) == expected)',
            "gold_call": 'int(_oracle_partition_decision(residual, I, used, box, 2) == expected)',
        },
        # invalid: noncanonical rational pair
        {
            "setup": """residual=((2,2),)
I=(0,)
used=()
box=(((0,1),(0,1),1,1),((1,4),(1,2),1,1))

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: partition_decision(residual, I, used, box, 2))',
            "gold_call": '_capture_value_error(lambda: _oracle_partition_decision(residual, I, used, box, 2))',
        },
    ]
