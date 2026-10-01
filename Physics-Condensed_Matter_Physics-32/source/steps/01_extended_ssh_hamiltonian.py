"""
Build the single-particle Hamiltonian matrix of the extended Su-Schrieffer-Heeger chain in real space.

The Su-Schrieffer-Heeger chain is the minimal one-dimensional topological insulator: spinless fermions hop between two sublattices A and B with alternating intracell and intercell amplitudes, and an inversion-symmetric next-nearest-neighbour hop from A in one cell to B in the next cell adds a third topological sector. Because the many-body Hamiltonian is quadratic, every thermal expectation value used later follows from this 2N-by-2N single-particle matrix, so its basis ordering and boundary treatment fix the layout of everything downstream.

Returns
-------
h : np.ndarray -- Float array of shape (2N, 2N) in the stated basis order.
"""

import numpy as np
import math

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def extended_ssh_hamiltonian(n_cells: int, t1: float, t2: float, t3: float, periodic: bool) -> "np.ndarray":
    '''Return the real symmetric single-particle Hamiltonian of the extended SSH chain.

    The chain has n_cells unit cells j = 0, ..., N - 1, each with orbitals A_j and B_j,
    ordered in the basis as (A_0, B_0, A_1, B_1, ..., A_{N-1}, B_{N-1}). The many-body
    Hamiltonian is

        H = - sum_j ( t1 a_j^dag b_j + t2 b_j^dag a_{j+1} + t3 a_j^dag b_{j+1} + h.c. ),

    so the matrix element between orbitals coupled by an amplitude t is -t. With
    periodic = True the cell index j + 1 is taken modulo N; with periodic = False every
    term that would involve cell N is omitted. Couplings between the same pair of
    orbitals add.

    Parameters
    ----------
    n_cells : int
        Number of unit cells N >= 2 (numpy integers accepted; bool is not).
    t1, t2, t3 : float
        Intracell, intercell and next-nearest-neighbour hopping amplitudes, finite and >= 0.
    periodic : bool
        Periodic (True) or open (False) boundary conditions.

    Returns
    -------
    h : np.ndarray
        Float array of shape (2N, 2N) in the stated basis order.

    Raises
    ------
    ValueError
        If n_cells is a bool, not an integer, or smaller than 2, or any hopping amplitude
        is negative or not finite.
    '''
    return h

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_extended_ssh_hamiltonian(n_cells: int, t1: float, t2: float, t3: float, periodic: bool) -> "np.ndarray":
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 2:
        raise ValueError("n_cells must be an integer >= 2")
    hops = np.array([t1, t2, t3], dtype=float)
    if not np.all(np.isfinite(hops)) or np.any(hops < 0.0):
        raise ValueError("hopping amplitudes must be finite and non-negative")
    n = int(n_cells)
    h = np.zeros((2 * n, 2 * n))
    for j in range(n):
        a, b = 2 * j, 2 * j + 1
        h[a, b] -= t1
        h[b, a] -= t1
        if j + 1 < n or periodic:
            a_next, b_next = 2 * ((j + 1) % n), 2 * ((j + 1) % n) + 1
            h[b, a_next] -= t2
            h[a_next, b] -= t2
            h[a, b_next] -= t3
            h[b_next, a] -= t3
    return h

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
         "call": "extended_ssh_hamiltonian(3, 1.0, 0.6, 1.5, True)",
         "gold_call": "_oracle_extended_ssh_hamiltonian(3, 1.0, 0.6, 1.5, True)"},
        {"setup": setup,
         "call": "extended_ssh_hamiltonian(3, 1.0, 0.6, 1.5, False)",
         "gold_call": "_oracle_extended_ssh_hamiltonian(3, 1.0, 0.6, 1.5, False)"},
        {"setup": setup,
         "call": "extended_ssh_hamiltonian(2, 0.7, 1.3, 0.4, True)",
         "gold_call": "_oracle_extended_ssh_hamiltonian(2, 0.7, 1.3, 0.4, True)"},
        {"setup": setup,
         "call": "extended_ssh_hamiltonian(4, 2.0, 0.0, 0.0, True)",
         "gold_call": "_oracle_extended_ssh_hamiltonian(4, 2.0, 0.0, 0.0, True)"},
        {"setup": setup,
         "call": "extended_ssh_hamiltonian(12, 1.0, 2.0, 0.5, True)[::5, ::3]",
         "gold_call": "_oracle_extended_ssh_hamiltonian(12, 1.0, 2.0, 0.5, True)[::5, ::3]"},
        {"setup": setup,
         "call": "extended_ssh_hamiltonian(np.int64(5), 0.0, 1.0, 0.0, False)",
         "gold_call": "_oracle_extended_ssh_hamiltonian(np.int64(5), 0.0, 1.0, 0.0, False)"},
        {"setup": guard,
         "call": "raises_value_error(extended_ssh_hamiltonian, 1, 1.0, 1.0, 0.0, True)",
         "gold_call": "raises_value_error(_oracle_extended_ssh_hamiltonian, 1, 1.0, 1.0, 0.0, True)"},
        {"setup": guard,
         "call": "raises_value_error(extended_ssh_hamiltonian, True, 1.0, 1.0, 0.0, True)",
         "gold_call": "raises_value_error(_oracle_extended_ssh_hamiltonian, True, 1.0, 1.0, 0.0, True)"},
        {"setup": guard,
         "call": "raises_value_error(extended_ssh_hamiltonian, 3, -1.0, 1.0, 0.0, True)",
         "gold_call": "raises_value_error(_oracle_extended_ssh_hamiltonian, 3, -1.0, 1.0, 0.0, True)"},
        {"setup": guard,
         "call": "raises_value_error(extended_ssh_hamiltonian, 3, 1.0, np.inf, 0.0, True)",
         "gold_call": "raises_value_error(_oracle_extended_ssh_hamiltonian, 3, 1.0, np.inf, 0.0, True)"},
        {"setup": guard,
         "call": "raises_value_error(extended_ssh_hamiltonian, 3.0, 1.0, 1.0, 0.0, True)",
         "gold_call": "raises_value_error(_oracle_extended_ssh_hamiltonian, 3.0, 1.0, 1.0, 0.0, True)"},
    ]
