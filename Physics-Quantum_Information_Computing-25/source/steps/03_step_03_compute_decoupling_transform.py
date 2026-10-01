"""
Compute the full-rank binary row transformation that, together with a given column permutation, puts a decoding matrix into the block-decoupled form, and confirm that the form is reached.

Multiplying the syndrome equation by an invertible binary matrix and relabelling the fault columns leaves the set of solutions unchanged, so the decoupled problem is equivalent to the original one.

Returns
-------
np.ndarray: (m, m) integer 0/1 transformation T.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_decoupling_transform(
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    n_blocks: int,
    block_cols: int,
) -> "np.ndarray":
    r"""Return the row transformation $T$ of the decoupled decoding matrix.

    $D$ has shape $(m, n)$. With $K$ = ``n_blocks``, $b$ = ``block_cols`` and
    $m_D = m / K$ rows per block, return the $(m, m)$ binary matrix $T$ for
    which $D' = T D_{\pi}$ (arithmetic modulo 2), where $D_{\pi}$ = ``D[:, perm]``,
    is decoupled: for every block $i$, the columns $i b, \dots, (i+1) b - 1$
    of $D'$ vanish outside rows $i m_D, \dots, (i+1) m_D - 1$, and inside those
    rows their first $m_D$ columns form the identity matrix. The columns
    after $K b$ are unconstrained.

    Parameters
    ----------
    d_matrix : np.ndarray
        2-D binary (0/1) matrix $D$ of shape $(m, n)$.
    perm : np.ndarray
        Integer permutation of ``range(n)``.
    n_blocks : int
        Positive number of blocks $K$ dividing $m$.
    block_cols : int
        Columns per block $b$, with $m_D \le b$ and $K b \le n$.

    Returns
    -------
    np.ndarray
        Integer 0/1 array of shape $(m, m)$.

    Raises
    ------
    ValueError
        If ``d_matrix`` is not a 2-D binary array, ``perm`` is not a
        permutation of ``range(n)``, ``n_blocks`` or ``block_cols`` violate
        the stated conditions, or no binary matrix $T$ produces the
        decoupled form.
    """
    return t_matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_decoupling_transform(
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    n_blocks: int,
    block_cols: int,
) -> "np.ndarray":
    """Reference transform: invert the identity-position columns over GF(2)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    d = np.asarray(d_matrix)
    if d.ndim != 2 or d.size == 0 or not np.issubdtype(d.dtype, np.number):
        raise ValueError("d_matrix must be a nonempty 2-D binary array")
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("d_matrix entries must be 0 or 1")
    d = d.astype(np.int64)
    m, n = d.shape
    p = np.asarray(perm)
    if p.shape != (n,) or not np.issubdtype(p.dtype, np.integer) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError("perm must be a permutation of range(n)")
    if not (_is_int(n_blocks) and n_blocks > 0 and m % n_blocks == 0):
        raise ValueError("n_blocks must be a positive divisor of m")
    rows = m // n_blocks
    if not (_is_int(block_cols) and rows <= block_cols and n_blocks * block_cols <= n):
        raise ValueError("block_cols must satisfy m_D <= block_cols and n_blocks*block_cols <= n")
    permuted = d[:, p]
    pivots = np.concatenate([i * block_cols + np.arange(rows) for i in range(n_blocks)])
    # Gauss-Jordan inversion of the identity-position columns over GF(2).
    work = np.hstack([permuted[:, pivots], np.eye(m, dtype=np.int64)])
    for col in range(m):
        hits = np.nonzero(work[col:, col])[0]
        if hits.size == 0:
            raise ValueError("the identity-position columns are singular over GF(2)")
        pivot = col + hits[0]
        if pivot != col:
            work[[col, pivot]] = work[[pivot, col]]
        others = np.nonzero(work[:, col])[0]
        others = others[others != col]
        work[others] ^= work[col]
    t_matrix = work[:, m:]
    decoupled = (t_matrix @ permuted) % 2
    for i in range(n_blocks):
        cols = slice(i * block_cols, (i + 1) * block_cols)
        inside = decoupled[i * rows:(i + 1) * rows, cols]
        outside = np.delete(decoupled[:, cols], np.s_[i * rows:(i + 1) * rows], axis=0)
        if outside.any() or not np.array_equal(inside[:, :rows], np.eye(rows, dtype=np.int64)):
            raise ValueError("no row transformation decouples the matrix under this permutation")
    return t_matrix.astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return numerical test specifications with elementwise array comparisons."""
    helpers = (
        "import numpy as np\n"
        "def _sig(t):\n"
        "    t = np.asarray(t, dtype=float)\n"
        "    k = np.arange(t.size, dtype=float)\n"
        "    return 100.0 * t.shape[0] + t.sum() + float(t.ravel() @ np.cos(0.61 * k + 0.2))\n"
        "def _scrambled(target, t0, perm):\n"
        "    t0 = np.asarray(t0)\n"
        "    m = t0.shape[0]\n"
        "    work = np.hstack([t0 % 2, np.eye(m, dtype=int)])\n"
        "    for c in range(m):\n"
        "        r = c + int(np.nonzero(work[c:, c])[0][0])\n"
        "        work[[c, r]] = work[[r, c]]\n"
        "        for q in range(m):\n"
        "            if q != c and work[q, c]:\n"
        "                work[q] ^= work[c]\n"
        "    inv = work[:, m:]\n"
        "    d = np.zeros_like(target)\n"
        "    d[:, perm] = (inv @ target) % 2\n"
        "    return d\n"
        "TARGET = np.array([\n"
        "    [1, 0, 1, 0, 0, 0, 1, 0, 1],\n"
        "    [0, 1, 1, 0, 0, 0, 0, 1, 1],\n"
        "    [0, 0, 0, 1, 0, 1, 1, 1, 0],\n"
        "    [0, 0, 0, 0, 1, 1, 0, 1, 1]])\n"
        "T0 = np.array([[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 1]])\n"
        "PERM = np.array([4, 7, 0, 2, 8, 1, 5, 3, 6])\n"
        "D0 = _scrambled(TARGET, T0, PERM)\n"
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
            "call": "compute_decoupling_transform(D0.copy(), PERM.copy(), 2, 3)",
            "gold_call": "_oracle_compute_decoupling_transform(D0.copy(), PERM.copy(), 2, 3)",
        },
        {
            "setup": helpers,
            "call": "compute_decoupling_transform(TARGET.copy(), np.arange(9), 2, 3)",
            "gold_call": "_oracle_compute_decoupling_transform(TARGET.copy(), np.arange(9), 2, 3)",
        },
        {
            "setup": helpers,
            "call": "compute_decoupling_transform(D0.copy(), PERM[[0, 1, 3, 4, 2, 5, 6, 7, 8]], 1, 5)",
            "gold_call": "_oracle_compute_decoupling_transform(D0.copy(), PERM[[0, 1, 3, 4, 2, 5, 6, 7, 8]], 1, 5)",
        },
        {
            "setup": helpers + (
                "T5 = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0],\n"
                "               [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])\n"
                "BIG = np.hstack([np.eye(5, dtype=int), np.array([[1, 1], [0, 1], [1, 0], [1, 1], [0, 1]])])\n"
                "P5 = np.array([6, 2, 0, 5, 3, 1, 4])\n"
                "D5 = _scrambled(BIG, T5, P5)\n"
            ),
            "call": "compute_decoupling_transform(D5.copy(), P5.copy(), 5, 1)",
            "gold_call": "_oracle_compute_decoupling_transform(D5.copy(), P5.copy(), 5, 1)",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_decoupling_transform(D0.copy(), PERM.copy(), 4, 3))",
            "gold_call": "_status(lambda: _oracle_compute_decoupling_transform(D0.copy(), PERM.copy(), 4, 3))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_decoupling_transform(D0.copy(), PERM[[0, 1, 6, 3, 4, 5, 2, 7, 8]], 2, 3))",
            "gold_call": "_status(lambda: _oracle_compute_decoupling_transform(D0.copy(), PERM[[0, 1, 6, 3, 4, 5, 2, 7, 8]], 2, 3))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_decoupling_transform(D0.copy(), np.array([0, 0, 1, 2, 3, 4, 5, 6, 7]), 2, 3))",
            "gold_call": "_status(lambda: _oracle_compute_decoupling_transform(D0.copy(), np.array([0, 0, 1, 2, 3, 4, 5, 6, 7]), 2, 3))",
        },
    ]
