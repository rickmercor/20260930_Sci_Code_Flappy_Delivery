"""
Forms the 16×16 real Pauli-transfer matrix of a two-qubit unitary.

A unitary channel is completely described by the coefficients χ_U(α, β) = (1/d) Tr[P_α U P_β U†]. These are the overlaps that direct fidelity estimation importance-samples.

Returns
-------
A float64 ndarray of shape (16, 16).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pauli_transfer_matrix(unitary: np.ndarray) -> np.ndarray:
    """Return the 16×16 real Pauli-transfer matrix of a two-qubit unitary.

    Index rows and columns in the same lexicographic Pauli order as
    two_qubit_pauli. The (α, β) entry is (1/4) Tr[P_α U P_β U†].

    Parameters
    ----------
    unitary : np.ndarray
        Complex array of shape (4, 4), unitary to numerical tolerance.

    Returns
    -------
    chi : np.ndarray
        Real array of shape (16, 16).

    Raises
    ------
    ValueError
        If unitary is not a finite 4×4 array, or is not unitary within
        absolute tolerance 1e-8.
    """
    return np.zeros((16, 16))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pauli_transfer_matrix(unitary: np.ndarray) -> np.ndarray:
    import numpy as np

    U = np.asarray(unitary, dtype=complex)
    if U.shape != (4, 4):
        raise ValueError("unitary must have shape (4, 4)")
    if not np.all(np.isfinite(U)):
        raise ValueError("unitary must be finite")
    if np.linalg.norm(U.conj().T @ U - np.eye(4)) > 1e-8:
        raise ValueError("unitary must be unitary")
    chi = np.zeros((16, 16), dtype=float)
    Ud = U.conj().T
    for alpha in range(16):
        Pa = _oracle_two_qubit_pauli(alpha)
        for beta in range(16):
            Pb = _oracle_two_qubit_pauli(beta)
            chi[alpha, beta] = float(np.real(np.trace(Pa @ U @ Pb @ Ud) / 4.0))
    return chi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
unitary = np.eye(4, dtype=complex)
""",
            "call": "pauli_transfer_matrix(unitary.copy())",
            "gold_call": "_oracle_pauli_transfer_matrix(unitary.copy())",
        },
        {
            "setup": """import numpy as np
unitary = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
""",
            "call": "pauli_transfer_matrix(unitary.copy())",
            "gold_call": "_oracle_pauli_transfer_matrix(unitary.copy())",
        },
        {
            "setup": """import numpy as np
unitary = _oracle_fsim_unitary(0.0, 0.0)
""",
            "call": "pauli_transfer_matrix(unitary.copy())",
            "gold_call": "_oracle_pauli_transfer_matrix(unitary.copy())",
        },
        {
            "setup": """import numpy as np
unitary = np.zeros((4, 4), dtype=complex)
def run_model():
    try:
        pauli_transfer_matrix(unitary.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pauli_transfer_matrix(unitary.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
