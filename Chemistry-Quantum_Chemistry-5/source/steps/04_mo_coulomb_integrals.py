"""
Return the two-electron integrals (pq|rs) of the chain's density-density interaction in the orbital basis of step 02, in the chemist's notation of the source's Eq. (1): the charge distribution of orbital pair (p, q) on the sites interacts through W with that of pair (r, s).

The source's self-energy and static potentials are written with Coulomb integrals in the chemist's notation and their antisymmetrised combination (its Eqs. (1)-(2)); for a lattice model the integrals follow from the interaction matrix and the orbital coefficients.

Returns
-------
ndarray of float64 with shape (N, N, N, N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mo_coulomb_integrals(C: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    '''Two-electron integrals of the chain's interaction in the orbital basis, chemist's notation.

    Parameters
    ----------
    C : np.ndarray
        Orthonormal orbitals of step 02, shape (N, N).
    W : np.ndarray
        Interaction matrix of step 01, shape (N, N).

    Returns
    -------
    eri : np.ndarray
        Array of shape (N, N, N, N) with eri[p, q, r, s] = (pq|rs) in chemist's notation for the
        density-density interaction: the site charge of the orbital pair (p, q) interacting through
        W with the site charge of the pair (r, s).

    Raises
    ------
    ValueError
        If C is not a square two-dimensional array or W does not have the matching shape.
    '''
    return [[[[0.0]]]]

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import eigh, solve


def _check_array(a, name, ndim=None, shape=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must have %d dimensions" % (name, ndim))
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    return a


def _oracle_mo_coulomb_integrals(C: "np.ndarray", W: "np.ndarray") -> "np.ndarray":
    C = _check_array(C, "C", ndim=2); N = C.shape[0]
    if C.shape[1] != N:
        raise ValueError("C must be a square matrix of orbital coefficients")
    W = _check_array(W, "W", shape=(N, N))
    D = np.einsum("ip,iq->ipq", C, C)                      # site charge density of the orbital pair (p, q)
    return np.einsum("ipq,ij,jrs->pqrs", D, W, D)           # (pq|rs) = sum_ij C_ip C_iq W_ij C_jr C_js

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(6, 1.0, 1.5, 1.5, np.array([-1.0, -0.5, 0.0, 0.0, 0.0, 0.0]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 6, 0.75)\nW = hW[1]",
            "call": "mo_coulomb_integrals(C.copy(), W.copy())",
            "gold_call": "_oracle_mo_coulomb_integrals(C.copy(), W.copy())",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(4, 0.8, 2.0, 0.7, np.array([-0.6, 0.0, 0.0, 0.3]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 1.0)\nW = hW[1]",
            "call": "mo_coulomb_integrals(C.copy(), W.copy())",
            "gold_call": "_oracle_mo_coulomb_integrals(C.copy(), W.copy())",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nhW = _oracle_chain_hamiltonian(5, 1.2, 1.0, 0.5, np.array([0.5, -0.2, 0.0, 0.1, -0.4]))\nC = _oracle_scaled_exchange_orbitals(hW[0], hW[1], 4, 0.5)\nW = hW[1]",
            "call": "mo_coulomb_integrals(C.copy(), W.copy())",
            "gold_call": "_oracle_mo_coulomb_integrals(C.copy(), W.copy())",
            "tol": 1e-10,
        },
    ]
