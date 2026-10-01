"""
Find the column permutation that brings the decoding matrix of a hypergraph-product code into the decoupled form used by the hierarchical decoder, with the block count that decoder prescribes for this code family.

The decoder trades one wide decoding problem for several small independent ones plus a sparse remainder; for hypergraph-product codes the split follows from the Kronecker structure of the checks, so no search is needed.

Returns
-------
np.ndarray: integer column permutation of length n1*n2 + m1*m2 + m1*n2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def decouple_hypergraph_product(h1: "np.ndarray", h2: "np.ndarray") -> "np.ndarray":
    r"""Return the column permutation of the decoupled decoding matrix.

    $D$ is the matrix of ``build_phenomenological_check_matrix(h1, h2)``,
    of shape $(m, n)$ with $m = m_1 n_2$. Use the off-diagonal part
    $A = h_1 \otimes I_{n_2}$, the first $n_1 n_2$ columns of $D$, and
    $K = m_1$ diagonal blocks. For each zero-based block index $i$,
    the block occupies rows $i n_2$ through $(i+1) n_2 - 1$. Its
    identity comes from the outcome-flip columns of those rows; its
    $B_i$ consists of the $m_2$ columns of $I_{m_1} \otimes h_2^{T}$ belonging
    to block $i$, restricted to those rows. Thus each
    $D_i = (I_{n_2} \mid B_i)$ has shape $(n_2,\ n_2 + m_2)$.

    Return the integer array ``perm`` of length $n$ such that
    ``D[:, perm]`` $= (\mathrm{diag}(D_1, \dots, D_K) \mid A)$, with no row
    transformation needed. Column $j$ of ``D[:, perm]`` is column ``perm[j]``
    of $D$. Order the blocks by increasing row index. Within each block, place
    the identity columns before $B_i$; inside each identity part,
    each $B_i$ and $A$, keep increasing original column indices.

    Parameters
    ----------
    h1 : np.ndarray
        Nonempty 2-D binary matrix $h_1$ of shape $(m_1, n_1)$.
    h2 : np.ndarray
        Nonempty 2-D binary matrix $h_2$ of shape $(m_2, n_2)$.

    Returns
    -------
    np.ndarray
        Integer permutation of ``range(n)``, $n = n_1 n_2 + m_1 m_2 + m_1 n_2$.

    Raises
    ------
    ValueError
        If either input is not a nonempty 2-D array whose entries are all 0 or 1.
    """
    return perm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_decouple_hypergraph_product(h1: "np.ndarray", h2: "np.ndarray") -> "np.ndarray":
    """Reference split: A = h1 kron I, blocks (I | h2^T) with K = m1."""
    import numpy as np

    def _shape(value, name):
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
        return array.shape

    m1, n1 = _shape(h1, "h1")
    m2, n2 = _shape(h2, "h2")
    n_data = n1 * n2 + m1 * m2
    order = []
    for block in range(m1):
        rows = block * n2 + np.arange(n2)
        order.extend((n_data + rows).tolist())
        order.extend((n1 * n2 + block * m2 + np.arange(m2)).tolist())
    order.extend(range(n1 * n2))
    return np.asarray(order, dtype=np.int64)

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
        "def _sig(perm):\n"
        "    p = np.asarray(perm, dtype=float)\n"
        "    k = np.arange(p.size, dtype=float)\n"
        "    return p.size + float(p @ np.cos(0.37 * k + 0.1)) + float(p @ np.sin(0.11 * k))\n"
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
            "call": "decouple_hypergraph_product(_ring(3), _ring(4))",
            "gold_call": "_oracle_decouple_hypergraph_product(_ring(3), _ring(4))",
        },
        {
            "setup": helpers,
            "call": "decouple_hypergraph_product(_ring(4), _ring(3))",
            "gold_call": "_oracle_decouple_hypergraph_product(_ring(4), _ring(3))",
        },
        {
            "setup": helpers,
            "call": "decouple_hypergraph_product(np.array([[1, 1, 0], [0, 1, 1]]), np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0]]))",
            "gold_call": "_oracle_decouple_hypergraph_product(np.array([[1, 1, 0], [0, 1, 1]]), np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 0]]))",
        },
        {
            "setup": helpers,
            "call": "decouple_hypergraph_product(np.array([[1]]), np.array([[1, 1]]))",
            "gold_call": "_oracle_decouple_hypergraph_product(np.array([[1]]), np.array([[1, 1]]))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: decouple_hypergraph_product(np.zeros((0, 3)), _ring(3)))",
            "gold_call": "_status(lambda: _oracle_decouple_hypergraph_product(np.zeros((0, 3)), _ring(3)))",
        },
    ]
