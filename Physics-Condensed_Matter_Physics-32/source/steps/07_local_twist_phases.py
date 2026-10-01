"""
Construct the site phases of the local twist operators that act within a unit cell, across a bond, or along the next-nearest-neighbour link of the SSH chain.

The global twist operator factorises into local pieces acting on two or four orbitals, each carrying the phase exp(i delta_k x) of its orbital position. Comparing the magnitudes of these local expectation values at the centre of the chain distinguishes where the Wannier charge centres sit, and therefore the topological sector, from local occupation measurements alone. The positions assigned to the two orbitals of a cell and to cells across the periodic boundary decide every phase.

Returns
-------
theta : np.ndarray -- Float array of shape (2N,).
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def local_twist_phases(n_cells: int, kind: str, j: int, delta_x: float) -> "np.ndarray":
    '''Return theta (length 2N) such that the local twist operator is exp(i sum_i theta_i n_i).

    Orbitals are ordered as in step 01. Cell l has position x_l = l, orbital A_l sits at
    x_l - delta_x and B_l at x_l + delta_x, and delta_k = 2 pi / N. Writing u(orbital, x)
    for "add delta_k * x to the entry of that orbital", the operators are

    - "intra":  u(A_j, j - delta_x), u(B_j, j + delta_x)
    - "inter":  u(B_j, j + delta_x), u(A_{j+1}, (j + 1) - delta_x)
    - "nnn":    u(A_j, j - delta_x), u(B_{j+1}, (j + 1) + delta_x)
    - "intra2": u(A_j, j - delta_x), u(B_j, j + delta_x), u(A_{j+1}, (j + 1) - delta_x),
                u(B_{j+1}, (j + 1) + delta_x)

    where the orbital index j + 1 is taken modulo N but the position (j + 1) is not
    reduced (for j = N - 1 it is N). Entries not touched are zero.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2.
    kind : str
        One of "intra", "inter", "nnn", "intra2".
    j : int
        Cell index, 0 <= j <= N - 1.
    delta_x : float
        Intracell half-separation of the orbitals, finite.

    Returns
    -------
    theta : np.ndarray
        Float array of shape (2N,).

    Raises
    ------
    ValueError
        If n_cells is not an integer >= 2, kind is not recognised, j is out of range or
        not an integer, or delta_x is not finite.
    '''
    return theta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_local_twist_phases(n_cells: int, kind: str, j: int, delta_x: float) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 2:
        raise ValueError("n_cells must be an integer >= 2")
    if isinstance(j, (bool, np.bool_)) or not isinstance(j, (int, np.integer)) or not 0 <= j < n_cells:
        raise ValueError("j out of range")
    if not np.isfinite(delta_x):
        raise ValueError("delta_x must be finite")
    n = int(n_cells)
    j = int(j)
    dk = 2.0 * np.pi / n
    theta = np.zeros(2 * n)
    a_j, b_j = 2 * j, 2 * j + 1
    a_next, b_next = 2 * ((j + 1) % n), 2 * ((j + 1) % n) + 1
    terms = {
        "intra": [(a_j, j - delta_x), (b_j, j + delta_x)],
        "inter": [(b_j, j + delta_x), (a_next, j + 1 - delta_x)],
        "nnn": [(a_j, j - delta_x), (b_next, j + 1 + delta_x)],
        "intra2": [(a_j, j - delta_x), (b_j, j + delta_x), (a_next, j + 1 - delta_x), (b_next, j + 1 + delta_x)],
    }
    if kind not in terms:
        raise ValueError("unknown kind")
    for index, position in terms[kind]:
        theta[index] += dk * position
    return theta

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
"""
    guard = setup + """
def raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0
"""
    return [
        {"setup": setup,
         "call": "local_twist_phases(5, 'intra', 2, 0.25)",
         "gold_call": "_oracle_local_twist_phases(5, 'intra', 2, 0.25)"},
        {"setup": setup,
         "call": "local_twist_phases(5, 'inter', 4, 0.25)",
         "gold_call": "_oracle_local_twist_phases(5, 'inter', 4, 0.25)"},
        {"setup": setup,
         "call": "local_twist_phases(5, 'nnn', 4, 0.25)",
         "gold_call": "_oracle_local_twist_phases(5, 'nnn', 4, 0.25)"},
        {"setup": setup,
         "call": "local_twist_phases(5, 'intra2', 4, 0.25)",
         "gold_call": "_oracle_local_twist_phases(5, 'intra2', 4, 0.25)"},
        {"setup": setup,
         "call": "local_twist_phases(2, 'intra2', 1, 0.1)",
         "gold_call": "_oracle_local_twist_phases(2, 'intra2', 1, 0.1)"},
        {"setup": setup,
         "call": "local_twist_phases(7, 'inter', 3, 0.0)",
         "gold_call": "_oracle_local_twist_phases(7, 'inter', 3, 0.0)"},
        {"setup": setup,
         "call": "local_twist_phases(np.int64(9), 'nnn', np.int64(0), -0.3)",
         "gold_call": "_oracle_local_twist_phases(np.int64(9), 'nnn', np.int64(0), -0.3)"},
        {"setup": guard,
         "call": "raises_value_error(local_twist_phases, 5, 'bond', 2, 0.25)",
         "gold_call": "raises_value_error(_oracle_local_twist_phases, 5, 'bond', 2, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(local_twist_phases, 5, 'intra', 5, 0.25)",
         "gold_call": "raises_value_error(_oracle_local_twist_phases, 5, 'intra', 5, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(local_twist_phases, 5, 'intra', 2.0, 0.25)",
         "gold_call": "raises_value_error(_oracle_local_twist_phases, 5, 'intra', 2.0, 0.25)"},
        {"setup": guard,
         "call": "raises_value_error(local_twist_phases, 5, 'intra', 2, np.nan)",
         "gold_call": "raises_value_error(_oracle_local_twist_phases, 5, 'intra', 2, np.nan)"},
    ]
