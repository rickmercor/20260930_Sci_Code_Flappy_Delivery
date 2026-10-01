"""
Order a selected rule multiset under prefix moiety availability.

A balanced collection of elementary rules is not necessarily a feasible mechanism because a rule cannot consume an intermediate before that intermediate becomes available. The source's OrderRules formulation therefore requires the initial inventory plus every cumulative rule prefix to remain nonnegative. Labeled catalytic residues and cofactors are exempt from this availability constraint because they may act as reusable sources or sinks.

Returns
-------
One-dimensional integer array containing the lexicographically smallest feasible sequence of one-based rule labels.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def order_rule_multiset(
    rule_counts: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
) -> 'np.ndarray':
    """Find the lexicographically smallest chemically feasible rule ordering.

    Rules are identified by one-based column labels. After every prefix, the
    initial counts plus cumulative rule changes must be nonnegative for every
    non-exempt moiety. Labeled catalytic/cofactor moieties marked ``True`` in
    ``exempt_moieties`` do not participate in this availability test. Every
    selected rule copy must be used exactly once.

    Returns
    -------
    np.ndarray
        One-dimensional integer array of one-based rule labels. Return an empty
        integer array when the balanced multiset has no feasible ordering.

    Raises
    ------
    ValueError
        If shapes, integer counts, or initial availability are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_order_rule_multiset(
    rule_counts: 'np.ndarray',
    rule_matrix: 'np.ndarray',
    initial_counts: 'np.ndarray',
    exempt_moieties: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    counts = np.asarray(rule_counts)
    matrix = np.asarray(rule_matrix)
    initial = np.asarray(initial_counts)
    exempt = np.asarray(exempt_moieties)

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(
            "rule_matrix must be a nonempty two-dimensional array"
        )
    if counts.ndim != 1 or counts.size != matrix.shape[1]:
        raise ValueError("rule_counts has incompatible shape")
    if initial.ndim != 1 or initial.size != matrix.shape[0]:
        raise ValueError("initial_counts has incompatible shape")
    if (
        exempt.ndim != 1
        or exempt.size != matrix.shape[0]
        or exempt.dtype != np.bool_
    ):
        raise ValueError(
            "exempt_moieties must be a boolean vector"
        )

    for value, name in (
        (counts, "rule_counts"),
        (matrix, "rule_matrix"),
        (initial, "initial_counts"),
    ):
        if (
            not np.issubdtype(value.dtype, np.number)
            or not np.isrealobj(value)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(f"{name} must be finite and numeric")
        if np.any(value != np.rint(value)):
            raise ValueError(f"{name} must contain integers")

    counts = counts.astype(int, copy=True)
    matrix = matrix.astype(int, copy=False)
    initial = initial.astype(int, copy=True)

    if np.any(counts < 0) or int(np.sum(counts)) == 0:
        raise ValueError(
            "rule_counts must be nonnegative and select at least one rule"
        )
    if np.any(initial < 0):
        raise ValueError("initial_counts must be nonnegative")

    failed = set()

    def _search(remaining, current, prefix):
        if int(np.sum(remaining)) == 0:
            return prefix

        state = (
            tuple(int(x) for x in remaining),
            tuple(int(x) for x in current[~exempt]),
        )
        if state in failed:
            return None

        for rule in np.flatnonzero(remaining):
            next_counts = current + matrix[:, rule]
            if np.any(next_counts[~exempt] < 0):
                continue

            remaining[rule] -= 1
            ordered = _search(
                remaining,
                next_counts,
                prefix + (int(rule) + 1,),
            )
            remaining[rule] += 1

            if ordered is not None:
                return ordered

        failed.add(state)
        return None

    answer = _search(counts, initial, tuple())

    if answer is None:
        return np.empty(0, dtype=int)

    return np.asarray(answer, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nT=np.array([[-1,-1,1,0],[1,0,0,-1],[0,1,-1,0],[0,0,1,-1],[0,0,0,1]])\ny=np.ones(4,dtype=int)\nc=np.array([1,0,0,0,0])\ne=np.zeros(5,dtype=bool)",
            "call": "order_rule_multiset(y.copy(),T.copy(),c.copy(),e.copy())",
            "gold_call": "_oracle_order_rule_multiset(y.copy(),T.copy(),c.copy(),e.copy())",
        },
        {
            "setup": "import numpy as np\nT=np.array([[0,0,-1],[0,0,1],[-1,0,1],[1,-1,0],[0,1,-1]])\ny=np.ones(3,dtype=int)\nc=np.array([1,0,0,0,0])\ne=np.zeros(5,dtype=bool)",
            "call": "order_rule_multiset(y.copy(),T.copy(),c.copy(),e.copy())",
            "gold_call": "_oracle_order_rule_multiset(y.copy(),T.copy(),c.copy(),e.copy())",
        },
        {
            "setup": "import numpy as np\nT=np.array([[-1,0],[1,-1],[0,1],[-1,1]])\ny=np.array([2,1])\nc=np.array([2,0,0,0])\ne=np.array([False,False,False,True])",
            "call": "order_rule_multiset(y.copy(),T.copy(),c.copy(),e.copy())",
            "gold_call": "_oracle_order_rule_multiset(y.copy(),T.copy(),c.copy(),e.copy())",
        },
        {
            "setup": "import numpy as np\nT=np.eye(2,dtype=int)\ny=np.array([1,-1])\nc=np.zeros(2,dtype=int)\ne=np.zeros(2,dtype=bool)\ndef check(fn):\n try: fn(y.copy(),T.copy(),c.copy(),e.copy())\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(order_rule_multiset)",
            "gold_call": "check(_oracle_order_rule_multiset)",
        },
    ]
