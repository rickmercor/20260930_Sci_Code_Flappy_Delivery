"""
Signed Pauli spectrum of an n-qubit density matrix.

For P running through the canonical enumeration of P_n, Spec(psi) collects the real numbers Tr(psi P). Signs are retained; the identity contribution Tr(psi I) = 1 for a normalized state is kept. Absolute values |Tr(psi P)| are a different convention.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pauli_spectrum(psi: np.ndarray, n: int) -> np.ndarray:
    """Signed Pauli spectrum of an n-qubit density matrix.

    For P running through the canonical enumeration of P_n, return the
    real numbers Tr(psi P). The signed convention is required: do not
    replace Tr(psi P) by |Tr(psi P)|. The identity contribution is
    Tr(psi I) = 1 for a normalized state and is retained.

    Parameters
    ----------
    psi : np.ndarray
        Density matrix of shape (2**n, 2**n).
    n : int
        Number of qubits.

    Returns
    -------
    spec : np.ndarray
        1-D real array of length 4**n in canonical Pauli order.
    """
    return spec

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pauli_spectrum(psi: np.ndarray, n: int) -> np.ndarray:
    """Reference oracle for the signed Pauli spectrum."""
    import numpy as np

    n = int(n)
    if n < 1:
        raise ValueError("n must be an integer >= 1.")
    rho = np.asarray(psi, dtype=complex)
    dim = 2**n
    if rho.ndim != 2 or rho.shape != (dim, dim):
        raise ValueError("psi must have shape (2**n, 2**n).")
    if not np.all(np.isfinite(rho)):
        raise ValueError("psi must be finite.")

    # Step 02 is available in the concatenated Studio namespace; the inline builder is a
    # fallback only for isolated execution of this step.
    _npauli = globals().get("_oracle_n_qubit_pauli")
    if not callable(_npauli):
        tables = (
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
            np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
            np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
        )

        def _npauli(n_loc: int, index: int) -> np.ndarray:
            n_loc = int(n_loc)
            idx = int(index)
            mat = tables[idx % 4]
            idx //= 4
            for _ in range(1, n_loc):
                mat = np.kron(tables[idx % 4], mat)
                idx //= 4
            return mat

    out = np.empty(4**n, dtype=float)
    for k in range(out.size):
        p = _npauli(n, k)
        out[k] = float(np.real(np.trace(rho @ p)))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "n = 1\n"
                "plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)\n"
            ),
            "call": "np.round(pauli_spectrum(plus, n), 12).tolist()",
            "gold_call": "np.round(_oracle_pauli_spectrum(plus, n), 12).tolist()",
        },
        {
            "setup": (
                "import numpy as np\n"
                "n = 1\n"
                "zero = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)\n"
            ),
            "call": "np.round(pauli_spectrum(zero, n), 12).tolist()",
            "gold_call": "np.round(_oracle_pauli_spectrum(zero, n), 12).tolist()",
        },
        {
            "setup": (
                "import numpy as np\n"
                "n = 2\n"
                "plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)\n"
                "psi = np.kron(plus, plus)\n"
            ),
            "call": "np.round(pauli_spectrum(psi, n), 12).tolist()",
            "gold_call": "np.round(_oracle_pauli_spectrum(psi, n), 12).tolist()",
        },
        {
            "setup": (
                "import numpy as np\n"
                "n = 1\n"
                "one = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)\n"
            ),
            "call": "np.round(pauli_spectrum(one, n), 12).tolist()",
            "gold_call": "np.round(_oracle_pauli_spectrum(one, n), 12).tolist()",
        },
    ]
