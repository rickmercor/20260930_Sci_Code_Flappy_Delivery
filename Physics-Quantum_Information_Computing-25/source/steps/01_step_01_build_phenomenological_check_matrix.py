"""
Build the single-round decoding matrix of a hypergraph-product code, with one column per data-qubit fault and one per check-outcome flip, together with the stabilizers of the opposite type.

The X-type checks of a hypergraph-product code detect Z errors on data qubits, and a flipped check outcome adds one more independent fault per check, so the decoder works with a matrix wider than the code's own check matrix.

Returns
-------
tuple: (D, H_Z) as integer 0/1 arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_phenomenological_check_matrix(
    h1: "np.ndarray",
    h2: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    r"""Return the fault-to-syndrome matrix and the opposite-type check matrix.

    With ``h1`` $=h_1$ of shape $(m_1, n_1)$ and ``h2`` $=h_2$ of shape
    $(m_2, n_2)$, the code has $n_1 n_2 + m_1 m_2$ data qubits and $m_1 n_2$
    $X$-type checks, $H_X = (h_1 \otimes I_{n_2} \mid I_{m_1} \otimes h_2^{T})$
    and $H_Z = (I_{n_1} \otimes h_2 \mid h_1^{T} \otimes I_{m_2})$ (Kronecker
    products, entries reduced modulo 2). The decoding matrix is
    $D = (H_X \mid I_{m_1 n_2})$: its first $n_1 n_2 + m_1 m_2$ columns are the
    data-qubit $Z$ faults in $H_X$ column order and its last $m_1 n_2$ columns
    are the outcome flips of the checks in row order.

    Parameters
    ----------
    h1 : np.ndarray
        Nonempty 2-D binary (0/1) parity-check matrix $h_1$ of the first classical code.
    h2 : np.ndarray
        Nonempty 2-D binary (0/1) parity-check matrix $h_2$ of the second classical code.

    Returns
    -------
    tuple of np.ndarray
        ``(D, H_Z)`` as integer 0/1 arrays of shapes
        $(m_1 n_2,\ n_1 n_2 + m_1 m_2 + m_1 n_2)$ and $(n_1 m_2,\ n_1 n_2 + m_1 m_2)$.

    Raises
    ------
    ValueError
        If either input is not a nonempty 2-D array whose entries are all 0 or 1.
    """
    return d_matrix, hz

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_phenomenological_check_matrix(
    h1: "np.ndarray",
    h2: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference construction of the hypergraph product with outcome-flip columns."""
    import numpy as np

    def _binary_matrix(value, name):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a 2-D binary array") from None
        if array.ndim != 2 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty 2-D array")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"{name} must hold real 0/1 entries")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} entries must be 0 or 1")
        return array.astype(np.int64)

    a = _binary_matrix(h1, "h1")
    b = _binary_matrix(h2, "h2")
    m1, n1 = a.shape
    m2, n2 = b.shape
    hx = np.hstack([
        np.kron(a, np.eye(n2, dtype=np.int64)),
        np.kron(np.eye(m1, dtype=np.int64), b.T),
    ]) % 2
    hz = np.hstack([
        np.kron(np.eye(n1, dtype=np.int64), b),
        np.kron(a.T, np.eye(m2, dtype=np.int64)),
    ]) % 2
    d_matrix = np.hstack([hx, np.eye(m1 * n2, dtype=np.int64)])
    return d_matrix.astype(np.int64), hz.astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return numerical test specifications with elementwise array comparisons."""
    helpers = (
        "import numpy as np\n"
        "def _ring(length):\n"
        "    h = np.zeros((length, length), dtype=int)\n"
        "    for i in range(length):\n"
        "        h[i, i] = 1\n"
        "        h[i, (i + 1) % length] = 1\n"
        "    return h\n"
        "def _sig(a, salt):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    k = np.arange(a.size, dtype=float)\n"
        "    return (1000.0 * a.shape[0] + a.shape[1] + np.abs(a).sum()\n"
        "            + float(a.ravel() @ np.cos(salt * k + 0.3)))\n"
        "def _pair(out):\n"
        "    d, hz = out\n"
        "    return _sig(d, 0.7) + 0.5 * _sig(hz, 1.3)\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": helpers,
            "call": "build_phenomenological_check_matrix(_ring(3), _ring(4))",
            "gold_call": "_oracle_build_phenomenological_check_matrix(_ring(3), _ring(4))",
        },
        {
            "setup": helpers,
            "call": "build_phenomenological_check_matrix(np.array([[1, 1]]), _ring(3))",
            "gold_call": "_oracle_build_phenomenological_check_matrix(np.array([[1, 1]]), _ring(3))",
        },
        {
            "setup": helpers,
            "call": "build_phenomenological_check_matrix(np.array([[1, 1, 0], [0, 1, 1]]), np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0]]))",
            "gold_call": "_oracle_build_phenomenological_check_matrix(np.array([[1, 1, 0], [0, 1, 1]]), np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0]]))",
        },
        {
            "setup": helpers,
            "call": "build_phenomenological_check_matrix(np.array([[1.0]]), np.array([[1.0, 1.0]]))",
            "gold_call": "_oracle_build_phenomenological_check_matrix(np.array([[1.0]]), np.array([[1.0, 1.0]]))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_phenomenological_check_matrix(np.array([[1, 2]]), _ring(3)))",
            "gold_call": "_status(lambda: _oracle_build_phenomenological_check_matrix(np.array([[1, 2]]), _ring(3)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_phenomenological_check_matrix(np.array([1, 1, 0]), _ring(3)))",
            "gold_call": "_status(lambda: _oracle_build_phenomenological_check_matrix(np.array([1, 1, 0]), _ring(3)))",
        },
    ]
