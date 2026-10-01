"""
Build one column of the adaptive residual-based sparse approximate inverse by solving on an initial pattern and then enlarging the pattern pass by pass under a residual tolerance, a relative admission threshold, a per-pass cap and a pass limit.

The a-priori pattern already captures most of a column of the inverse, and the residual-driven passes add positions only where the remaining residual is concentrated, stopping when $\|r_k\|_2 \le \epsilon$, when no residual entry satisfies $|r_k(i)| \ge \delta \|r_k\|_2$, or when the pass budget is spent.

Returns
-------
tuple: (length-$n$ float column $m_k$, float final residual 2-norm $\|r_k\|_2$, int completed passes).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_adaptive_column(matrix: "np.ndarray", column: int, tolerance: float, threshold: float, cap: int, max_passes: int, *, seed_mode: str = "power") -> tuple:
    r"""Return the terminal least-squares column of the adaptive support process.

    Write $A$ for ``matrix`` and $k$ for ``column``. For any support $J$, let
    $m(J)$ be the unique minimizer of $\lVert A m - e_k \rVert_2$ among
    vectors that vanish outside $J$, where $e_k$ is the $k$-th unit vector.
    Its entries must be accurate to $10^{-12}$ relative to the largest.
    Coefficients multiplied by their matrix-column 2-norms must also be
    accurate to $10^{-8}$ relative to
    $\max(1, \text{the largest such coefficient})$. The column norms may span
    up to $10^{198}$, but each reduced matrix after column normalization has
    condition number at most $10^{5}$. All relevant norms and coefficients
    are finite. Define $r(J) = A \, m(J) - e_k$ over all $n$ rows, including
    rows not reached by $A[:, J]$.

    With ``seed_mode`` equal to ``"power"`` the initial support is the union
    of the nonzero positions of $e_k$, $A e_k$ and $A^2 e_k$, where $A^2$
    denotes the floating-point matrix square, including exact numerical
    cancellations. With ``seed_mode`` equal to ``"diagonal"`` the initial
    support is $\{k\}$ alone. The initial admission history is empty.

    At each state, the eligible rows are the current reduced row support
    excluding the admission history. A row qualifies if
    $\lvert r(J)(i) \rvert \ge \text{threshold} \cdot \lVert r(J) \rVert_2$.
    Admit at most ``cap`` qualifying rows, ranked by decreasing residual
    magnitude with smaller global row index breaking ties. The next support
    is the smallest superset of $J$ containing all column coordinates coupled
    to those admitted residual rows; the next history includes all
    admissions.

    Return the first state with residual norm at most ``tolerance``, with no
    qualifying admission, or with ``max_passes`` completed admissions. One
    completed admission means one nonempty group of admitted rows, even if it
    introduces no new column coordinate. The initial minimization consumes no
    admission, and an empty admission consumes none. The returned vector and
    norm correspond to the least-squares problem on the terminal support,
    after all counted admissions have taken effect.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape $(n, n)$, $n \ge 1$, with finite entries, such
        that ``matrix[:, pattern]`` has full column rank for every pattern met
        (true for any nonsingular matrix).
    column : int
        Column index $k$ with $0 \le k < n$.
    tolerance : float
        Finite positive residual tolerance $\epsilon$.
    threshold : float
        Finite nonnegative relative admission threshold $\delta$.
    cap : int
        Maximum number of rows admitted per pass, at least 1.
    max_passes : int
        Maximum number of completed passes, at least 0.
    seed_mode : str
        Either ``"power"`` for the $e_k$, $A e_k$, $A^2 e_k$ support, or
        ``"diagonal"`` for the single position $k$.

    Returns
    -------
    tuple
        ``(m, residual_norm, passes)``: the final column as a float array of
        length $n$, the float $\lVert A m - e_k \rVert_2$ of that column, and
        the int number of completed passes.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers,
        ``column`` is not an integer in $[0, n)$, ``tolerance`` is not a
        finite positive number, ``threshold`` is not a finite nonnegative
        number, ``cap`` is not an integer of at least 1, ``max_passes`` is not
        an integer of at least 0, or ``seed_mode`` is neither ``"power"`` nor
        ``"diagonal"`` (booleans are rejected throughout).
    """
    return (column_vector, residual_norm, passes)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_adaptive_column(matrix: "np.ndarray", column: int, tolerance: float, threshold: float, cap: int, max_passes: int, *, seed_mode: str = "power") -> tuple:
    """Reference implementation (initial column, then threshold-driven passes)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError("matrix must be a nonempty square 2-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must have finite entries")
    size = array.shape[0]
    if not (_is_integer(column) and 0 <= column < size):
        raise ValueError("column must be an integer index of the matrix")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    if not (_is_number(threshold) and threshold >= 0.0):
        raise ValueError("threshold must be a finite nonnegative number")
    if not (_is_integer(cap) and cap >= 1):
        raise ValueError("cap must be an integer of at least 1")
    if not (_is_integer(max_passes) and max_passes >= 0):
        raise ValueError("max_passes must be an integer of at least 0")
    if seed_mode not in ("power", "diagonal"):
        raise ValueError("seed_mode must be 'power' or 'diagonal'")
    k = int(column)

    def _residual(vector):
        # Full residual: row k keeps its -1 whenever the pattern misses it.
        out = array @ vector
        out[k] -= 1.0
        return out

    def _fresh_state(support):
        # Empty factorization extended by the whole reduced block of `support`.
        rows = np.flatnonzero(np.any(array[:, support] != 0.0, axis=1))
        empty = np.empty(0, dtype=int)
        initial_state = (empty, empty.copy(), np.empty((0, 0)), np.empty((0, 0)))
        return _oracle_extend_reduced_qr(
            initial_state, rows, support, array[np.ix_(rows, support)], k, size
        )

    if seed_mode == "diagonal":
        pattern = np.array([k], dtype=int)
        state, vector, _ = _fresh_state(pattern)
    else:
        pattern, vector = _oracle_initialize_adaptive_column(array, k)
        state = None
    residual = _residual(vector)
    admitted = np.zeros(0, dtype=int)
    passes = 0
    while np.linalg.norm(residual) > tolerance and passes < max_passes:
        grown, recorded = _oracle_advance_column_pattern(
            array, pattern, residual, admitted, threshold, cap
        )
        if recorded.size == admitted.size:
            break  # nothing qualified for admission
        if state is None:
            state, _, _ = _fresh_state(pattern)
        added_columns = np.setdiff1d(grown, state[1])
        reached = np.flatnonzero(np.any(array[:, grown] != 0.0, axis=1))
        added_rows = np.setdiff1d(reached, state[0])
        row_order = np.concatenate((state[0], added_rows))
        state, vector, _ = _oracle_extend_reduced_qr(
            state, added_rows, added_columns,
            array[np.ix_(row_order, added_columns)], k, size
        )
        pattern, admitted = grown, recorded
        residual = _residual(vector)
        passes += 1
    return vector, float(np.linalg.norm(residual)), int(passes)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Original regression cases followed by fitted/admission coverage."""
    return [
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(24, (-1, 2, -5, 7), 11)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 9, 0.05, 0.2, 3, 3), 24)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 9, 0.05, 0.2, 3, 3), 24)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(24, (-1, 2, -5, 7), 11)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 9, 0.05, 0.45, 3, 4), 24)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 9, 0.05, 0.45, 3, 4), 24)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(20, (1, -3, 4), 5)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 0, 0.5, 0.2, 2, 3), 20)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 0, 0.5, 0.2, 2, 3), 20)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(20, (1, -3, 4), 5)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 12, 0.01, 0.1, 2, 0), 20)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 12, 0.01, 0.1, 2, 0), 20)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(30, (-1, 1, 6, -6), 23)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 14, 0.02, 0.15, 1, 5), 30)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 14, 0.02, 0.15, 1, 5), 30)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(12, (2, -3, 7, -8), 141)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 11, 0.005, 0.05, 1, 4), 12)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 11, 0.005, 0.05, 1, 4), 12)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(22, (1, -3, 4), 67)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 19, 0.005, 0.15, 3, 5), 22)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 19, 0.005, 0.15, 3, 5), 22)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = _net(25, (1, -4, 9), 70)\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 24, 0.02, 0.3, 1, 3), 25)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 24, 0.02, 0.3, 1, 3), 25)',
        },
        {
            'setup': 'import numpy as np\ndef _triple(t, n):\n    if not isinstance(t, tuple) or len(t) != 3:\n        return -1.0\n    m = np.asarray(t[0], dtype=float)\n    if m.shape != (n,):\n        return -2.0\n    w = np.cos(np.arange(n, dtype=float) + 1.0)\n    return float(np.sum(np.abs(m)) + np.sum(m * w) + 7.0 * float(t[1]) + 10.0 * int(t[2]))\ndef _net(n, offsets, seed):\n    rng = np.random.default_rng(seed)\n    A = np.diag(3.0 + rng.random(n))\n    for d in offsets:\n        for i in range(n):\n            j = i + d\n            if 0 <= j < n:\n                A[i, j] = rng.uniform(-1.5, 1.5)\n    return A\nA = np.zeros((8, 8))\nfor i in range(8):\n    A[(i + 1) % 8, i] = 2.0 + 0.1 * i\n',
            'call': '_triple(compute_adaptive_column(A.copy(), 3, 0.1, 0.2, 3, 3), 8)',
            'gold_call': '_triple(_oracle_compute_adaptive_column(A.copy(), 3, 0.1, 0.2, 3, 3), 8)',
        },
        {
            'setup': 'import numpy as np\ndef _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': '_status(lambda: compute_adaptive_column(np.eye(4), 1, 0.0, 0.2, 2, 3))',
            'gold_call': '_status(lambda: _oracle_compute_adaptive_column(np.eye(4), 1, 0.0, 0.2, 2, 3))',
        },
        {
            'setup': 'import numpy as np\ndef _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            'call': '_status(lambda: compute_adaptive_column(np.eye(4), 1, 0.1, 0.2, 2, -1))',
            'gold_call': '_status(lambda: _oracle_compute_adaptive_column(np.eye(4), 1, 0.1, 0.2, 2, -1))',
        },
        {
            'setup': "import numpy as np\ndef scaled_network(seed,n,reverse=False):\n    rng = np.random.default_rng(seed)\n    base = np.diag(3. + rng.random(n))\n    for d in (-1,2,-5,7):\n        for i in range(n):\n            j = i+d\n            if 0 <= j < n:\n                base[i,j] = rng.uniform(-1.5,1.5)\n    scales = np.power(10.,np.linspace(-90.,90.,n))\n    if reverse:\n        scales = scales[::-1].copy()\n    return base*scales,scales\ndef check_adaptive(fn,A,s,k,tolerance,threshold,cap,passes):\n    result = fn(A.copy(),k,tolerance,threshold,cap,passes)\n    if not isinstance(result,tuple) or len(result)!=3 or np.asarray(result[0]).shape!=(len(A),):\n        raise AssertionError('Expected full column, residual norm, pass count')\n    m,norm,count = result\n    r = A@m\n    r[k] -= 1.\n    return np.r_[np.asarray(m)*s,r,float(norm),float(count)]\nA,s = scaled_network(11,24)\nB,t = scaled_network(67,22,True)\n",
            'call': 'check_adaptive(compute_adaptive_column,A,s,9,.05,.2,3,3)',
            'gold_call': 'check_adaptive(_oracle_compute_adaptive_column,A,s,9,.05,.2,3,3)',
            'tol': 1e-08,
        },
        {
            'setup': "import numpy as np\ndef scaled_network(seed,n,reverse=False):\n    rng = np.random.default_rng(seed)\n    base = np.diag(3. + rng.random(n))\n    for d in (-1,2,-5,7):\n        for i in range(n):\n            j = i+d\n            if 0 <= j < n:\n                base[i,j] = rng.uniform(-1.5,1.5)\n    scales = np.power(10.,np.linspace(-90.,90.,n))\n    if reverse:\n        scales = scales[::-1].copy()\n    return base*scales,scales\ndef check_adaptive(fn,A,s,k,tolerance,threshold,cap,passes):\n    result = fn(A.copy(),k,tolerance,threshold,cap,passes)\n    if not isinstance(result,tuple) or len(result)!=3 or np.asarray(result[0]).shape!=(len(A),):\n        raise AssertionError('Expected full column, residual norm, pass count')\n    m,norm,count = result\n    r = A@m\n    r[k] -= 1.\n    return np.r_[np.asarray(m)*s,r,float(norm),float(count)]\nA,s = scaled_network(11,24)\nB,t = scaled_network(67,22,True)\n",
            'call': 'check_adaptive(compute_adaptive_column,A,s,9,.05,.45,3,4)',
            'gold_call': 'check_adaptive(_oracle_compute_adaptive_column,A,s,9,.05,.45,3,4)',
            'tol': 1e-08,
        },
        {
            'setup': "import numpy as np\ndef scaled_network(seed,n,reverse=False):\n    rng = np.random.default_rng(seed)\n    base = np.diag(3. + rng.random(n))\n    for d in (-1,2,-5,7):\n        for i in range(n):\n            j = i+d\n            if 0 <= j < n:\n                base[i,j] = rng.uniform(-1.5,1.5)\n    scales = np.power(10.,np.linspace(-90.,90.,n))\n    if reverse:\n        scales = scales[::-1].copy()\n    return base*scales,scales\ndef check_adaptive(fn,A,s,k,tolerance,threshold,cap,passes):\n    result = fn(A.copy(),k,tolerance,threshold,cap,passes)\n    if not isinstance(result,tuple) or len(result)!=3 or np.asarray(result[0]).shape!=(len(A),):\n        raise AssertionError('Expected full column, residual norm, pass count')\n    m,norm,count = result\n    r = A@m\n    r[k] -= 1.\n    return np.r_[np.asarray(m)*s,r,float(norm),float(count)]\nA,s = scaled_network(11,24)\nB,t = scaled_network(67,22,True)\n",
            'call': 'check_adaptive(compute_adaptive_column,B,t,19,.005,.15,3,5)',
            'gold_call': 'check_adaptive(_oracle_compute_adaptive_column,B,t,19,.005,.15,3,5)',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\nS=np.zeros((7,7)); S[(np.arange(7)+1)%7,np.arange(7)]=1\nA=S+S@S\ndef packed(fn,k,tol,cap):\n    m,r,p=fn(A.copy(),k,tol,0.,cap,4)\n    return np.r_[np.asarray(m),float(r),int(p)]\n',
            'call': 'packed(compute_adaptive_column,1,0.5,1)',
            'gold_call': 'packed(_oracle_compute_adaptive_column,1,0.5,1)',
        },
        {
            'setup': 'import numpy as np\nS=np.zeros((7,7)); S[(np.arange(7)+1)%7,np.arange(7)]=1\nA=S+S@S\ndef packed(fn,k,tol,cap):\n    m,r,p=fn(A.copy(),k,tol,0.,cap,4)\n    return np.r_[np.asarray(m),float(r),int(p)]\n',
            'call': 'packed(compute_adaptive_column,2,0.5,1)',
            'gold_call': 'packed(_oracle_compute_adaptive_column,2,0.5,1)',
        },
        {
            'setup': 'import numpy as np\nS=np.zeros((7,7)); S[(np.arange(7)+1)%7,np.arange(7)]=1\nA=S+S@S\ndef packed(fn,k,tol,cap):\n    m,r,p=fn(A.copy(),k,tol,0.,cap,4)\n    return np.r_[np.asarray(m),float(r),int(p)]\n',
            'call': 'packed(compute_adaptive_column,3,0.1,2)',
            'gold_call': 'packed(_oracle_compute_adaptive_column,3,0.1,2)',
        },
        {
            'setup': 'import numpy as np\nrng=np.random.default_rng(613)\nA=6*np.eye(9)+rng.uniform(-.8,.8,(9,9))\nA[np.abs(A)<.35]=0.\nnp.fill_diagonal(A,6.)\ndef packed(fn,k,mode):\n    m,r,p=fn(A.copy(),k,0.05,0.25,3,3,seed_mode=mode)\n    return np.r_[np.asarray(m),float(r),int(p)]\n',
            'call': 'packed(compute_adaptive_column,4,"diagonal")',
            'gold_call': 'packed(_oracle_compute_adaptive_column,4,"diagonal")',
        },
        {
            'setup': 'import numpy as np\nrng=np.random.default_rng(613)\nA=6*np.eye(9)+rng.uniform(-.8,.8,(9,9))\nA[np.abs(A)<.35]=0.\nnp.fill_diagonal(A,6.)\ndef packed(fn,k,mode):\n    m,r,p=fn(A.copy(),k,0.05,0.25,3,3,seed_mode=mode)\n    return np.r_[np.asarray(m),float(r),int(p)]\n',
            'call': 'packed(compute_adaptive_column,0,"power")',
            'gold_call': 'packed(_oracle_compute_adaptive_column,0,"power")',
        },
        {
            'setup': 'import numpy as np\nA=np.eye(4)*3.\ndef status(fn):\n    try:fn(A.copy(),1,.1,.25,3,3,seed_mode="powers")\n    except ValueError:return 1\n    return 0\n',
            'call': 'status(compute_adaptive_column)',
            'gold_call': 'status(_oracle_compute_adaptive_column)',
        },
    ]
