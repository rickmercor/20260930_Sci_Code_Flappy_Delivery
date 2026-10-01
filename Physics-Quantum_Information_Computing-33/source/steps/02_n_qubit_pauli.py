"""
Return one n-qubit Pauli string in canonical enumeration order.

Index k runs through 0, ..., 4**n - 1. Base-4 digits of k, least-significant digit first, label qubits 0, 1, ..., n-1 with 0->I, 1->X, 2->Y, 3->Z. The operator is the Kronecker product with qubit n-1 as the leftmost factor, so computational-basis bit 0 is the least-significant bit. Index 0 is the identity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def n_qubit_pauli(n: int, index: int) -> np.ndarray:
    """Return one n-qubit Pauli string in canonical enumeration order.

    Index ``k`` runs through 0, ..., 4**n - 1. The base-4 digits of ``k``,
    least-significant digit first, label qubits 0, 1, ..., n-1 with
    0 -> I, 1 -> X, 2 -> Y, 3 -> Z. The operator is the Kronecker product
    with qubit n-1 as the leftmost factor, so computational-basis bit 0 is
    the least-significant bit. In particular ``k = 0`` is the identity and
    ``n = 1`` recovers the four single-qubit Paulis.

    Parameters
    ----------
    n : int
        Number of qubits, n >= 1.
    index : int
        Enumeration index in {0, 1, ..., 4**n - 1}.

    Returns
    -------
    P : np.ndarray
        Complex array of shape (2**n, 2**n).
    """
    return P

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_n_qubit_pauli(n: int, index: int) -> np.ndarray:
    """Reference oracle for the canonical Kronecker Pauli string."""
    import numpy as np

    n = int(n)
    if n < 1:
        raise ValueError("n must be an integer >= 1.")
    nmax = 4**n
    idx = int(index)
    if idx < 0 or idx >= nmax:
        raise ValueError("index must satisfy 0 <= index < 4**n.")

    pauli = globals().get("_oracle_single_qubit_pauli")
    if not callable(pauli):
        try:
            import importlib.util
            from pathlib import Path

            path = Path(__file__).resolve().parent / "01_single_qubit_pauli.py"
            spec = importlib.util.spec_from_file_location("_o1", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            pauli = getattr(mod, "_oracle_single_qubit_pauli")
            globals()["_oracle_single_qubit_pauli"] = pauli
        except Exception:
            tables = (
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
                np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
                np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
            )

            def pauli(kind: int) -> np.ndarray:
                k = int(kind)
                if k not in (0, 1, 2, 3):
                    raise ValueError("kind must be 0, 1, 2, or 3 (I, X, Y, Z).")
                return tables[k].copy()

    mat = pauli(idx % 4)
    idx //= 4
    for _ in range(1, n):
        mat = np.kron(pauli(idx % 4), mat)
        idx //= 4
    return mat

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pair = (
        "[np.round(np.asarray({fn}({n}, {k})).real, 12).tolist(), "
        "np.round(np.asarray({fn}({n}, {k})).imag, 12).tolist()]"
    )
    return [
        {
            "setup": "import numpy as np\nn = 1\nk = 3",
            "call": pair.format(fn="n_qubit_pauli", n="n", k="k"),
            "gold_call": pair.format(fn="_oracle_n_qubit_pauli", n="n", k="k"),
        },
        {
            "setup": "import numpy as np\nn = 2\nk = 0",
            "call": pair.format(fn="n_qubit_pauli", n="n", k="k"),
            "gold_call": pair.format(fn="_oracle_n_qubit_pauli", n="n", k="k"),
        },
        {
            "setup": "import numpy as np\nn = 2\nk = 1",
            "call": pair.format(fn="n_qubit_pauli", n="n", k="k"),
            "gold_call": pair.format(fn="_oracle_n_qubit_pauli", n="n", k="k"),
        },
        {
            "setup": "import numpy as np\nn = 2\nk = 4",
            "call": pair.format(fn="n_qubit_pauli", n="n", k="k"),
            "gold_call": pair.format(fn="_oracle_n_qubit_pauli", n="n", k="k"),
        },
    ]
