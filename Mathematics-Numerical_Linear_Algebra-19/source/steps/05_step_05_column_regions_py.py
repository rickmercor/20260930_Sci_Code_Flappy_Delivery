"""
Resolve every terminal support of one NRSAI column over a closed or half-open squared-tolerance box.

At a fixed adaptive state, only the tolerance and significance comparisons depend on the parameter pair. Following each nonempty decision region yields an exact partition, including boundary-only cases, without imposing an iteration cap.

Use eta = epsilon^2 and xi = delta^2. The final support J_k(C; eta, xi) is constant on each returned terminal region.

Returns
-------
tuple of (subbox, J) terminal regions with exact numerical bounds and integer supports.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def column_regions(C: np.ndarray, k: int, box: tuple, c: int = 2) -> tuple:
    """Return the exact terminal-support partition for one NRSAI column.

    Parameters
    ----------
    C : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        in exact arithmetic after binary64 conversion.
    k : int
        Non-Boolean integer column index in [0, n).
    box : tuple
        Squared-parameter box as in partition_decision: two intervals, each
        (low_rational, high_rational, low_closed, high_closed), with reduced
        integer rational pairs, integer 0/1 flags, eta >= 0 and 0 < xi <= 1.
        Bounds are ordered and a singleton must be closed at both ends.
    c : int
        Positive non-Boolean integer selection cap.

    Returns
    -------
    regions : tuple
        Entries (subbox, J), with numerical box encodings and increasing
        integer coefficient supports. Regions are disjoint and cover box.
        Different histories ending with the same support are not merged.
        Sort by the numerical flattened interval fields, then by J.
        Selection histories start empty; all active coefficients are refitted;
        no dropping, additional scaling, or iteration cap is applied.

    Raises
    ------
    ValueError
        If C, k, box, or c violates the stated conditions.
    """
    return ()

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_column_regions(C: np.ndarray, k: int, box: tuple, c: int = 2) -> tuple:
    domain = _nr_box(box)
    c = _nr_integer(c, 'c', 1)
    I, J = _oracle_initial_pattern(C, k)
    pending = [(I, J, (), _nr_encode_box(domain))]
    leaves = []
    while pending:
        I, J, used, current_box = pending.pop()
        _, residual, _ = _oracle_fit_column(C, k, J)
        decisions = _oracle_partition_decision(residual, I, used, current_box, c)
        for subbox, selected in decisions:
            if not selected:
                leaves.append((subbox, J))
            else:
                I_new, J_new, used_new = _oracle_expand_pattern(C, I, J, used, selected)
                pending.append((I_new, J_new, used_new, subbox))
    leaves.sort(key=lambda item: (_nr_box_key(_nr_box(item[0])), item[1]))
    return tuple(leaves)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: multiple adaptive regions
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
k=0
box=(((1,200),(1,50),1,1),((1,10),(9,10),1,1))

expected = (((((1, 200), (1, 85), 1, 0), ((1, 10), (16, 85), 1, 1)), (0, 1, 2, 3)),
 ((((1, 200), (1, 85), 1, 0), ((16, 85), (64, 85), 0, 1)), (0, 1, 2, 3)),
 ((((1, 200), (1, 85), 1, 0), ((64, 85), (9, 10), 0, 1)), (0, 2, 3)),
 ((((1, 85), (1, 50), 1, 1), ((1, 10), (9, 10), 1, 1)), (0, 2, 3)))
""",
            "call": 'int(column_regions(C, k, box, 2) == expected)',
            "gold_call": 'int(_oracle_column_regions(C, k, box, 2) == expected)',
        },
        # boundary: exact residual-tolerance singleton
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
k=0
box=(((1,85),(1,85),1,1),((1,10),(9,10),1,1))

expected = (((((1, 85), (1, 85), 1, 1), ((1, 10), (9, 10), 1, 1)), (0, 2, 3)),)
""",
            "call": 'int(column_regions(C, k, box, 2) == expected)',
            "gold_call": 'int(_oracle_column_regions(C, k, box, 2) == expected)',
        },
        # edge: exact eligibility transition adds the missing inverse position
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
k=0
box=(((1,200),(1,200),1,1),((64,85),(64,85),1,1))

expected = (((((1, 200), (1, 200), 1, 1), ((64, 85), (64, 85), 1, 1)), (0, 1, 2, 3)),)
""",
            "call": 'int(column_regions(C, k, box, 2) == expected)',
            "gold_call": 'int(_oracle_column_regions(C, k, box, 2) == expected)',
        },
        # invalid: an open singleton is empty
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
k=0
box=(((1,85),(1,85),0,1),((1,10),(9,10),1,1))

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: column_regions(C, k, box, 2))',
            "gold_call": '_capture_value_error(lambda: _oracle_column_regions(C, k, box, 2))',
        },
    ]
