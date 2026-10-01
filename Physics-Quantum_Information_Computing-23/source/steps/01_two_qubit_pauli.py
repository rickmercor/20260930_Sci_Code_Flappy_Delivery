"""
Returns the 4×4 matrix of a two-qubit Pauli given its lexicographic code in {I, X, Y, Z}⊗{I, X, Y, Z}.

The phase-free n-qubit Pauli basis is the tensor product of single-qubit Paulis. On two qubits the sixteen operators are indexed by a code from 0 through 15 in lexicographic {I, X, Y, Z}⊗{I, X, Y, Z} order, and they are orthogonal under the Hilbert–Schmidt product Tr(P_α P_β) = 4 δ_{αβ}.

Returns
-------
A complex ndarray of shape (4, 4).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def two_qubit_pauli(code: int) -> np.ndarray:
    """Return the 4×4 matrix of the two-qubit Pauli with the given code.

    Codes run from 0 through 15 in lexicographic tensor order
    {I, X, Y, Z}⊗{I, X, Y, Z}, so code 0 is I⊗I and code 5 is X⊗X.

    Parameters
    ----------
    code : int
        Integer in {0, …, 15}.

    Returns
    -------
    matrix : np.ndarray
        Complex array of shape (4, 4).

    Raises
    ------
    ValueError
        If code is not an integer in {0, …, 15}.
    """
    return np.zeros((4, 4), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_two_qubit_pauli(code: int) -> np.ndarray:
    import numpy as np

    if type(code) is not int:
        raise ValueError("code must be an integer")
    if code < 0 or code > 15:
        raise ValueError("code must lie in {0, ..., 15}")
    I = np.eye(2, dtype=complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    local = (I, X, Y, Z)
    return np.kron(local[code // 4], local[code % 4])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
code = 0
""",
            "call": "two_qubit_pauli(code)",
            "gold_call": "_oracle_two_qubit_pauli(code)",
        },
        {
            "setup": """import numpy as np
code = 5
""",
            "call": "two_qubit_pauli(code)",
            "gold_call": "_oracle_two_qubit_pauli(code)",
        },
        {
            "setup": """import numpy as np
code = 10
""",
            "call": "two_qubit_pauli(code)",
            "gold_call": "_oracle_two_qubit_pauli(code)",
        },
        {
            "setup": """
def run_model():
    try:
        two_qubit_pauli(16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_two_qubit_pauli(16)
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
