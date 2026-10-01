"""
Expand the active coefficient and row sets from selected residual rows, retaining every selection in the column history.

NRSAI maps selected residual rows to nonzero columns of those matrix rows. The selected row labels are not themselves coefficient indices; a selection can enlarge only the history and leave the support unchanged.




$$

\widehat J={j:C_{i,j}\ne0,\ i\in\widehat D}\setminus J,\quad J_+=J\cup\widehat J,\quad I_+=\operatorname{rowsupp}(C(:,J_+)),\quad U_+=U\cup\widehat D.

$$

Returns
-------
tuple: (I_new, J_new, used_new), each an increasing tuple of native integer indices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def expand_pattern(
    C: np.ndarray,
    I: tuple[int, ...],
    J: tuple[int, ...],
    used: tuple[int, ...],
    selected: tuple[int, ...],
) -> tuple:
    """Return enlarged active rows, coefficients, and permanent row history.

    Parameters
    ----------
    C : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        in exact arithmetic after binary64 conversion.
    I, J, used : tuple[int, ...]
        Increasing distinct indices in [0, n). I and J are nonempty;
        I must be exactly the nonzero row set of C[:, J]; used is a subset of I.
    selected : tuple[int, ...]
        Distinct indices in I excluding used, in any order. Empty is allowed.

    Returns
    -------
    I_new, J_new, used_new : tuple
        Three increasing integer tuples. J_new adds every nonzero column of
        C[selected, :]; I_new includes all rows touched by J_new; used_new
        records all selections even when J_new equals J. No position is dropped.

    Raises
    ------
    ValueError
        If C fails the matrix conditions; if an index collection is invalid;
        if I or J is empty; if I is not the row closure of J; if used is not
        contained in I; or if selected repeats or includes a used/outside row.
    """
    return (), (), ()

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_expand_pattern(
    C: np.ndarray,
    I: tuple[int, ...],
    J: tuple[int, ...],
    used: tuple[int, ...],
    selected: tuple[int, ...],
) -> tuple:
    data = _nr_matrix(C)
    n = data['n']
    I = _nr_indices(I, n, 'I', nonempty=True)
    J = _nr_indices(J, n, 'J', nonempty=True)
    used = _nr_indices(used, n, 'used')
    selected = _nr_indices(selected, n, 'selected', sorted_required=False)
    closure = tuple(sorted(set().union(*(data['cols'][j] for j in J))))
    if I != closure or not set(used).issubset(I) or not set(selected).issubset(set(I) - set(used)):
        raise ValueError("inconsistent row closure, history, or selection")
    J_new = tuple(sorted(set(J).union(*(data['rows'][i] for i in selected))))
    I_new = tuple(sorted(set().union(*(data['cols'][j] for j in J_new))))
    used_new = tuple(sorted(set(used) | set(selected)))
    return I_new, J_new, used_new

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: selected row introduces a different coefficient
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
I=(0,3)
J=(0,)
used=()
selected=(3,)

expected = ((0, 2, 3), (0, 3), (3,))
""",
            "call": 'int(expand_pattern(C, I, J, used, selected) == expected)',
            "gold_call": 'int(_oracle_expand_pattern(C, I, J, used, selected) == expected)',
        },
        # boundary: empty selection preserves state
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
I=(0,3)
J=(0,)
used=()
selected=()

expected = ((0, 3), (0,), ())
""",
            "call": 'int(expand_pattern(C, I, J, used, selected) == expected)',
            "gold_call": 'int(_oracle_expand_pattern(C, I, J, used, selected) == expected)',
        },
        # edge: history grows although the coefficient support does not
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
I=(0,2,3)
J=(0,3)
used=()
selected=(3,)

expected = ((0, 2, 3), (0, 3), (3,))
""",
            "call": 'int(expand_pattern(C, I, J, used, selected) == expected)',
            "gold_call": 'int(_oracle_expand_pattern(C, I, J, used, selected) == expected)',
        },
        # invalid: previously used row selected again
        {
            "setup": """import numpy as np
n = 4
C = 2.0 * np.eye(n)
C[np.arange(n), (np.arange(n)+1) % n] = -1.0
I=(0,2,3)
J=(0,3)
used=(3,)
selected=(3,)

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: expand_pattern(C, I, J, used, selected))',
            "gold_call": '_capture_value_error(lambda: _oracle_expand_pattern(C, I, J, used, selected))',
        },
    ]
