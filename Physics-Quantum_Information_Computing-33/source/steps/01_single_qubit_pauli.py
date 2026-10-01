"""
Return a phase-free single-qubit Pauli matrix.

The phase-free single-qubit Paulis {I, X, Y, Z}, labelled 0, 1, 2, 3, generate the n-qubit set P_n used to form the signed Pauli spectrum. Y is the unique complex generator; I, X, and Z are real.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def single_qubit_pauli(kind: int) -> np.ndarray:
    """Return a phase-free single-qubit Pauli matrix.

    The integer ``kind`` selects an element of {I, X, Y, Z} by
    0 -> I, 1 -> X, 2 -> Y, 3 -> Z. These are the generators of the
    n-qubit Pauli set P_n = {I, X, Y, Z}^{otimes n} used to build the
    signed Pauli spectrum.

    Parameters
    ----------
    kind : int
        Pauli label in {0, 1, 2, 3}.

    Returns
    -------
    P : np.ndarray
        Complex array of shape (2, 2).
    """
    return P

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_single_qubit_pauli(kind: int) -> np.ndarray:
    """Reference oracle for the four phase-free single-qubit Paulis."""
    import numpy as np

    tables = (
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    )
    k = int(kind)
    if k not in (0, 1, 2, 3):
        raise ValueError("kind must be 0, 1, 2, or 3 (I, X, Y, Z).")
    return tables[k].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pair = (
        "[np.round(np.asarray({fn}({k})).real, 12).tolist(), "
        "np.round(np.asarray({fn}({k})).imag, 12).tolist()]"
    )
    return [
        {
            "setup": "import numpy as np\nk = 0",
            "call": pair.format(fn="single_qubit_pauli", k="k"),
            "gold_call": pair.format(fn="_oracle_single_qubit_pauli", k="k"),
        },
        {
            "setup": "import numpy as np\nk = 1",
            "call": pair.format(fn="single_qubit_pauli", k="k"),
            "gold_call": pair.format(fn="_oracle_single_qubit_pauli", k="k"),
        },
        {
            "setup": "import numpy as np\nk = 2",
            "call": pair.format(fn="single_qubit_pauli", k="k"),
            "gold_call": pair.format(fn="_oracle_single_qubit_pauli", k="k"),
        },
        {
            "setup": "import numpy as np\nk = 3",
            "call": pair.format(fn="single_qubit_pauli", k="k"),
            "gold_call": pair.format(fn="_oracle_single_qubit_pauli", k="k"),
        },
    ]
