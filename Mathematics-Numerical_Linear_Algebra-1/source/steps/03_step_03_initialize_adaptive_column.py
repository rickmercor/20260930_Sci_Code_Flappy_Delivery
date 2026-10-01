"""
Start one column of the adaptive sparse approximate inverse from its a-priori pattern by solving the least-squares problem restricted to the rows that pattern reaches.

The columns of the matrix inside the a-priori pattern $J_k^0$ vanish outside the reached rows $I_k^0$, the rows on which $A(:, J_k^0)$ has a nonzero entry, so each column's least-squares problem shrinks to the small dense problem $\min \|A(I_k^0, J_k^0) m_k(J_k^0) - e_k(I_k^0)\|_2$, whose residual norm equals that of the full column residual whenever the column's own row is reached.

Returns
-------
tuple: (sorted integer a-priori pattern $J_k^0$, length-$n$ float least-squares column on it).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def initialize_adaptive_column(matrix: "np.ndarray", column: int) -> tuple:
    r"""Return the a-priori pattern of one column and the least-squares column on it.

    Write $A$ for ``matrix`` and $k$ for ``column``. The a-priori pattern
    $J_k^0$ is the set of positions at which column $k$ of the identity, of
    $A$, or of $A^2$ (formed in floating point) is nonzero. The starting
    column $m$ is the vector of length $n$ whose entries vanish outside
    $J_k^0$ and which minimizes $\lVert A m - e_k \rVert_2$, with $e_k$ the
    unit vector of index $k$. ``matrix[:, pattern]`` is assumed to have full
    column rank, so $m$ is unique; its entries must be accurate to $10^{-12}$
    relative to the largest of them. Column scales may differ by up to
    $10^{198}$: after normalizing each reached column by its 2-norm, the
    reduced matrix has condition number at most $10^{5}$. The full-rank
    minimizer is required even for weakly scaled columns. Its coefficients
    multiplied by the respective column 2-norms must also be accurate to
    $10^{-8}$ relative to $\max(1, \text{the largest such coefficient})$. All
    relevant column norms and coefficients are finite. These are numerical
    accuracy conditions, not a prescribed library routine or rank-truncation
    rule.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    column : int
        Column index ``k`` with ``0 <= k < n``.

    Returns
    -------
    tuple
        ``(pattern, m)``: the sorted 1-D integer array of pattern positions and
        the float column of length ``n``.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers, or
        ``column`` is not an integer in ``[0, n)`` (booleans are rejected).
    """
    return (pattern, column_vector)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _solve_reduced_column(array, k, pattern):
    """Least-squares column on a pattern, solved on the rows the pattern reaches."""
    import numpy as np

    positions = np.asarray(pattern, dtype=int)
    # Rows outside those reached by matrix[:, pattern] cannot be changed, so the
    # minimizer is the solution of the reduced problem on the reached rows.
    rows = np.flatnonzero(np.any(array[:, positions] != 0.0, axis=1))
    target = (rows == k).astype(float)
    vector = np.zeros(array.shape[0])
    reduced = array[np.ix_(rows, positions)]
    scales = np.max(np.abs(reduced), axis=0)
    q, r = np.linalg.qr(reduced / scales, mode="reduced")
    vector[positions] = np.linalg.solve(r, q.T @ target) / scales
    return vector


def _oracle_initialize_adaptive_column(matrix: "np.ndarray", column: int) -> tuple:
    """Reference implementation (a-priori seed, then the reduced least-squares solve)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    if not (_is_integer(column) and 0 <= column < array.shape[0]):
        raise ValueError("column must be an integer index of the matrix")
    k = int(column)
    pattern = _oracle_seed_power_pattern(array, k)
    vector = _solve_reduced_column(array, k, pattern)
    return pattern, vector

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    common = (
        "import numpy as np\n"
        "def _pair(t, n):\n"
        "    if not isinstance(t, tuple) or len(t) != 2:\n"
        "        return -1.0\n"
        "    p = np.asarray(t[0])\n"
        "    m = np.asarray(t[1], dtype=float)\n"
        "    if p.ndim != 1 or (p.size and not np.issubdtype(p.dtype, np.integer)) or m.shape != (n,):\n"
        "        return -2.0\n"
        "    w = np.cos(np.arange(n, dtype=float) + 1.0)\n"
        "    code = 1000.0 * p.size + np.sum((np.arange(p.size) + 1.0) * (p + 1.0))\n"
        "    return float(code / 1000.0 + np.sum(np.abs(m)) + 0.5 * np.sum(m * w))\n"
        "def _band(n):\n"
        "    A = np.zeros((n, n))\n"
        "    for i in range(n):\n"
        "        A[i, i] = 4.0 + 0.2 * (i % 3)\n"
        "        if i >= 1:\n"
        "            A[i, i - 1] = -2.2\n"
        "        if i >= 3:\n"
        "            A[i, i - 3] = 0.6\n"
        "        if i + 1 < n:\n"
        "            A[i, i + 1] = -0.9\n"
        "    return A\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    cases = [
        {
            "setup": common + "A = _band(14)\n",
            "call": "_pair(initialize_adaptive_column(A.copy(), 6), 14)",
            "gold_call": "_pair(_oracle_initialize_adaptive_column(A.copy(), 6), 14)",
        },
        {
            "setup": common + "A = _band(14)\n",
            "call": "_pair(initialize_adaptive_column(A.copy(), 13), 14)",
            "gold_call": "_pair(_oracle_initialize_adaptive_column(A.copy(), 13), 14)",
        },
        {
            "setup": common + (
                "A = np.zeros((10, 10))\n"
                "for i in range(10):\n"
                "    A[i, i] = 4.0 + 0.1 * i\n"
                "    A[(i + 1) % 10, i] = -1.5\n"
                "    A[i, (i + 3) % 10] = 0.75\n"
            ),
            "call": "_pair(initialize_adaptive_column(A.copy(), 4), 10)",
            "gold_call": "_pair(_oracle_initialize_adaptive_column(A.copy(), 4), 10)",
        },
        {
            "setup": common + (
                "A = np.array([[2.0, 0.0, 0.0, 0.0, 0.0],\n"
                "              [1.0, 3.0, 0.0, 0.0, 0.0],\n"
                "              [1.0, 0.0, 3.0, 0.0, 0.0],\n"
                "              [0.0, 1.0, -1.0, 2.0, 0.0],\n"
                "              [0.0, 1.0, 1.0, 0.0, 5.0]])\n"
            ),
            "call": "_pair(initialize_adaptive_column(A.copy(), 0), 5)",
            "gold_call": "_pair(_oracle_initialize_adaptive_column(A.copy(), 0), 5)",
        },
        {
            "setup": common + (
                "A = np.zeros((8, 8))\n"
                "for i in range(8):\n"
                "    A[(i + 1) % 8, i] = 2.0 + 0.1 * i\n"
            ),
            "call": "_pair(initialize_adaptive_column(A.copy(), 3), 8)",
            "gold_call": "_pair(_oracle_initialize_adaptive_column(A.copy(), 3), 8)",
        },
        {
            "setup": common + "A = np.array([[-2.5]])\n",
            "call": "_pair(initialize_adaptive_column(A.copy(), 0), 1)",
            "gold_call": "_pair(_oracle_initialize_adaptive_column(A.copy(), 0), 1)",
        },
        {
            "setup": status,
            "call": "_status(lambda: initialize_adaptive_column(np.eye(3), -1))",
            "gold_call": "_status(lambda: _oracle_initialize_adaptive_column(np.eye(3), -1))",
        },
        {
            "setup": status,
            "call": "_status(lambda: initialize_adaptive_column(np.array([[1.0, np.inf], [0.0, 1.0]]), 0))",
            "gold_call": "_status(lambda: _oracle_initialize_adaptive_column(np.array([[1.0, np.inf], [0.0, 1.0]]), 0))",
        },
    ]

    extra_setup = "import numpy as np\ndef scaled_case(seed, exponents):\n    rng = np.random.default_rng(seed)\n    n = len(exponents)\n    base = 5*np.eye(n) + rng.uniform(-.7,.7,(n,n))\n    scales = np.power(10.,np.array(exponents,dtype=float))\n    return base*scales, scales\ndef check_column(fn, A, scales, k):\n    result = fn(A.copy(),k)\n    if not isinstance(result,tuple) or len(result)!=2:\n        raise AssertionError('Expected pattern and column')\n    J,m = np.asarray(result[0]),np.asarray(result[1])\n    if J.ndim!=1 or not np.issubdtype(J.dtype,np.integer) or m.shape!=(len(A),):\n        raise AssertionError('Incorrect pattern or column shape')\n    residual = A@m\n    residual[k] -= 1.\n    return np.r_[J.astype(float),m*scales,residual]\nA,s = scaled_case(73,[-90.,90.,-45.,45.,0.,15.])\n"
    cases.extend([{'setup': extra_setup,
      'call': 'check_column(initialize_adaptive_column,A,s,0)',
      'gold_call': 'check_column(_oracle_initialize_adaptive_column,A,s,0)',
      'tol': 1e-08},
     {'setup': extra_setup,
      'call': 'check_column(initialize_adaptive_column,A,s,4)',
      'gold_call': 'check_column(_oracle_initialize_adaptive_column,A,s,4)',
      'tol': 1e-08},
     {'setup': extra_setup,
      'call': 'check_column(initialize_adaptive_column,A[:,::-1].copy(),s[::-1].copy(),3)',
      'gold_call': 'check_column(_oracle_initialize_adaptive_column,A[:,::-1].copy(),s[::-1].copy(),3)',
      'tol': 1e-08}])
    return cases
