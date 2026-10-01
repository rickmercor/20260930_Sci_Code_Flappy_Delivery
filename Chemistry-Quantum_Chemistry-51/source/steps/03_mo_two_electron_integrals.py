"""
Return the four-index array of chemists' two-electron integrals (pq|rs) = sum_{ij} C_ip C_iq V_ij C_jr C_js in the orbital basis C for the density-density site-basis integrals (pq|rs)_site = delta_pq delta_rs V_pr. Raise ValueError if V and C are not both (K, K) matrices.

The pair coupled cluster equations and the generalised Fock matrix need the Coulomb-type integrals (pp|qq), the exchange-type integrals (pq|pq) and, for the orbital gradient and the extended Koopmans' matrix, the full four-index array in the current orbital basis.

Returns
-------
float array of shape (K, K, K, K).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    '''Two-electron integrals (pq|rs) in an orbital basis for the density-density model.

    Parameters
    ----------
    V : np.ndarray
        Site-basis interaction matrix (K, K).
    C : np.ndarray
        Orbital coefficients (K, K), one orbital per column.

    Returns
    -------
    Vm : np.ndarray
        Array (K, K, K, K) of chemists' integrals (pq|rs) in the orbital basis.

    Raises
    ------
    ValueError
        If V and C are not both (K, K) matrices.
    '''
    return Vm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mo_two_electron_integrals(V: "np.ndarray", C: "np.ndarray") -> "np.ndarray":
    """(pq|rs) = sum_ij C_ip C_iq V_ij C_jr C_js in the orbital basis C. Returns (K, K, K, K)."""
    V = np.asarray(V, dtype=float); C = np.asarray(C, dtype=float)
    K = V.shape[0]
    if V.shape != (K, K) or C.shape != (K, K):
        raise ValueError("V and C must be (K, K)")
    X = np.einsum('ip,iq->ipq', C, C)                       # X[i,p,q] = C_ip C_iq
    return np.einsum('ipq,ij,jrs->pqrs', X, V, X, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.15, 6.0, 4.0, -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25]))\nV = hV[1]\nC = _oracle_rhf_orbitals(hV[0], hV[1], 4)",
            "call": "mo_two_electron_integrals(V, C)",
            "gold_call": "_oracle_mo_two_electron_integrals(V, C)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.1, 4.0, 3.0, -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1]))\nV = hV[1]\nC = _oracle_rhf_orbitals(hV[0], hV[1], 3)",
            "call": "mo_two_electron_integrals(V, C)",
            "gold_call": "_oracle_mo_two_electron_integrals(V, C)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(0.8, 0.0, 5.0, 2.5, np.array([-3.0, -3.2, -2.75, -3.15]))\nV = hV[1]\nC = np.eye(4)",
            "call": "mo_two_electron_integrals(V, C)",
            "gold_call": "_oracle_mo_two_electron_integrals(V, C)",
            "tol": 1e-12,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        mo_two_electron_integrals(np.eye(4), np.eye(3))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_mo_two_electron_integrals(np.eye(4), np.eye(3))\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
