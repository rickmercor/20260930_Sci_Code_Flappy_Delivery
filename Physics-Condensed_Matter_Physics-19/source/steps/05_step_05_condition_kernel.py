"""
Produce the Pfaffian kernel of a discrete Pfaffian point process restricted to the remaining sites, conditioned on a set of sites being all occupied or all empty.

Conditioning a Pfaffian point process on the occupation or emptiness of some sites leaves a Pfaffian point process on the other sites, which is what lets exact samplers and order-statistic formulas proceed one site at a time.

Returns
-------
np.ndarray: the conditioned kernel on the remaining sites.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def condition_kernel(kernel: np.ndarray, sites: list[int], occupied: bool) -> np.ndarray:
    """Return the kernel of a discrete Pfaffian point process after conditioning on sites.

    ``kernel`` is a real skew-symmetric ``(2L, 2L)`` array whose rows and
    columns ``2x`` and ``2x + 1`` belong to site ``x = 0, ..., L - 1``; the
    Pfaffian of its principal submatrix on any set of distinct sites is the
    probability that all of them are occupied, with
    ``Pf([[0, a], [-a, 0]]) = a``. Condition on the event that every site in
    ``sites`` is occupied (``occupied=True``) or that every site in ``sites``
    is empty (``occupied=False``). Return a skew-symmetric kernel for the
    remaining sites, kept in increasing order and relabelled
    ``0, 1, ..., L - len(sites) - 1``, with the same layout and Pfaffian
    convention, whose principal Pfaffians are the conditional probabilities
    that the corresponding remaining sites are all occupied. The input is not
    modified.

    Parameters
    ----------
    kernel : np.ndarray
        Skew-symmetric array of shape ``(2L, 2L)``.
    sites : list[int]
        Distinct site labels in ``0, ..., L - 1``; at least one and fewer than
        ``L``.
    occupied : bool
        ``True`` to condition on all listed sites occupied, ``False`` to
        condition on all of them empty.

    Returns
    -------
    np.ndarray
        Array of shape ``(2(L - len(sites)), 2(L - len(sites)))``.

    Raises
    ------
    ValueError
        If ``kernel`` is not a finite square array of even order, ``sites`` is
        empty, contains repeated or out-of-range labels, non-integers or
        booleans, or lists every site, ``occupied`` is not a bool, or the
        conditioning event has probability zero (the 2 x 2 block system that
        defines the conditioning is singular).
    """
    return conditioned

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_condition_kernel(kernel: np.ndarray, sites: list[int], occupied: bool) -> np.ndarray:
    """Reference implementation (Schur complement with or without the symplectic unit)."""
    import numpy as np

    matrix = np.asarray(kernel, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2
            or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix))):
        raise ValueError("kernel must be a finite square array of even order")
    flag_is_bool = isinstance(occupied, (bool, np.bool_))
    if not flag_is_bool:
        raise ValueError("occupied must be a bool")
    size = matrix.shape[0] // 2
    labels = list(sites)
    if (not labels or len(labels) >= size
            or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in labels)
            or len({int(s) for s in labels}) != len(labels)
            or any(not 0 <= int(s) < size for s in labels)):
        raise ValueError("sites must be distinct in-range integers, not all sites")
    chosen = [int(s) for s in labels]
    rest = [s for s in range(size) if s not in set(chosen)]
    inner = np.ravel([[2 * s, 2 * s + 1] for s in chosen])
    outer = np.ravel([[2 * s, 2 * s + 1] for s in rest])
    block = matrix[np.ix_(inner, inner)].copy()
    if not occupied:
        # Emptiness enters through K_Y - J (J the symplectic unit on the listed sites).
        block -= np.kron(np.eye(len(chosen)), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    try:
        correction = np.linalg.solve(block, matrix[np.ix_(inner, outer)])
    except np.linalg.LinAlgError as error:
        raise ValueError("the conditioning event has probability zero") from error
    conditioned = matrix[np.ix_(outer, outer)] - matrix[np.ix_(outer, inner)] @ correction
    if not np.all(np.isfinite(conditioned)):
        raise ValueError("the conditioning event has probability zero")
    return 0.5 * (conditioned - conditioned.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications graded on Pfaffian minors."""
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
    build = (
        "import numpy as np\n"
        "def _pf(A):\n"
        "    A = np.array(A, dtype=float)\n"
        "    n = A.shape[0]\n"
        "    out = 1.0\n"
        "    for k in range(0, n - 1, 2):\n"
        "        p = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))\n"
        "        if p != k + 1:\n"
        "            A[[k + 1, p], :] = A[[p, k + 1], :]\n"
        "            A[:, [k + 1, p]] = A[:, [p, k + 1]]\n"
        "            out = -out\n"
        "        if A[k, k + 1] == 0.0:\n"
        "            return 0.0\n"
        "        out *= A[k, k + 1]\n"
        "        t = A[k, k + 2:] / A[k, k + 1]\n"
        "        A[k + 2:, k + 2:] += np.outer(t, A[k + 2:, k + 1]) - np.outer(A[k + 2:, k + 1], t)\n"
        "    return float(out)\n"
        "def _blocks(K, sites):\n"
        "    idx = np.ravel([[2 * s, 2 * s + 1] for s in sites]).astype(int)\n"
        "    return K[np.ix_(idx, idx)]\n"
        "def _kern(w, n):\n"
        "    L = w.size\n"
        "    x = np.arange(L, dtype=float)\n"
        "    G = 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n"
        "    cols = []\n"
        "    for d in range(n):\n"
        "        v = ((x - x.mean()) / L) ** d\n"
        "        for k in range(len(cols) // 2):\n"
        "            a, b = cols[2 * k], cols[2 * k + 1]\n"
        "            v = v + (b @ G @ v) * a - (a @ G @ v) * b\n"
        "        if len(cols) % 2:\n"
        "            v = v / (cols[-1] @ G @ v)\n"
        "        cols.append(v)\n"
        "    R = np.array(cols).T\n"
        "    sg = np.sign(x[:, None] - x[None, :])\n"
        "    P = 0.5 * sg @ (R * w[:, None])\n"
        "    A, B, PA, PB = R[:, 0::2], R[:, 1::2], P[:, 0::2], P[:, 1::2]\n"
        "    S = w[:, None] * (B @ PA.T - A @ PB.T)\n"
        "    D = np.outer(w, w) * (A @ B.T - B @ A.T)\n"
        "    K = np.zeros((2 * L, 2 * L))\n"
        "    K[0::2, 0::2] = PB @ PA.T - PA @ PB.T - 0.5 * sg\n"
        "    K[0::2, 1::2] = S.T\n"
        "    K[1::2, 0::2] = -S\n"
        "    K[1::2, 1::2] = -D\n"
        "    return K\n"
        "def _minors(K):\n"
        "    K = np.asarray(K, dtype=float)\n"
        "    L = K.shape[0] // 2\n"
        "    r1 = np.array([K[2 * s, 2 * s + 1] for s in range(L)])\n"
        "    r2 = _pf(_blocks(K, [0, L - 1]))\n"
        "    J = np.kron(np.eye(2), [[0.0, 1.0], [-1.0, 0.0]])\n"
        "    gap = _pf(J - _blocks(K, [L - 2, L - 1]))\n"
        "    asym = float(np.abs(K + K.T).max())\n"
        "    return float(r1 @ np.cos(0.9 * np.arange(L)) + 2.0 * r2 + 5.0 * gap\n"
        "                 + 1.0e3 * asym + 7.0 * L)\n"
    )
    preserve = (
        'import copy\n'
        'def _preserved(fn, *args, **kwargs):\n'
        '    saved_args = copy.deepcopy(args)\n'
        '    saved_kwargs = copy.deepcopy(kwargs)\n'
        '    result = fn(*args, **kwargs)\n'
        '    for actual, before in list(zip(args, saved_args)) + [(kwargs[k], saved_kwargs[k]) for k in kwargs]:\n'
        '        if isinstance(actual, np.ndarray):\n'
        "            assert np.array_equal(actual, before, equal_nan=True), 'Array input was modified'\n"
        '        else:\n'
        "            assert actual == before, 'Input was modified'\n"
        '    return result\n'
    )
    return [
        {
            "setup": (build + "K = _kern(0.3 ** (0.5 * np.arange(10.0)), 4)\n") + preserve,
            "call": "_minors(_preserved(condition_kernel, K.copy(), [4], True))",
            "gold_call": "_minors(_preserved(_oracle_condition_kernel, K.copy(), [4], True))",
        },
        {
            "setup": (build + "K = _kern(0.3 ** (0.5 * np.arange(10.0)), 4)\n") + preserve,
            "call": "_minors(_preserved(condition_kernel, K.copy(), [3], False))",
            "gold_call": "_minors(_preserved(_oracle_condition_kernel, K.copy(), [3], False))",
        },
        {
            "setup": (build + "K = _kern(0.5 ** (0.5 * np.arange(12.0)), 6)\n") + preserve,
            "call": "_minors(_preserved(condition_kernel, K.copy(), [9, 2], True))",
            "gold_call": "_minors(_preserved(_oracle_condition_kernel, K.copy(), [9, 2], True))",
        },
        {
            "setup": (build + "K = _kern(0.5 ** (0.5 * np.arange(12.0)), 6)\n") + preserve,
            "call": "_minors(_preserved(condition_kernel, K.copy(), [0, 5, 11], False))",
            "gold_call": "_minors(_preserved(_oracle_condition_kernel, K.copy(), [0, 5, 11], False))",
        },
        {
            "setup": (build + "K = _kern(0.4 ** (0.5 * np.arange(5.0)), 2)\n") + preserve,
            "call": "_minors(_preserved(condition_kernel, K.copy(), [0, 1, 2], False))",
            "gold_call": "_minors(_preserved(_oracle_condition_kernel, K.copy(), [0, 1, 2], False))",
        },
        {
            "setup": (build + "K = _kern(0.3 ** (0.5 * np.arange(10.0)), 4)\n") + preserve,
            "call": "float(np.trace(_preserved(condition_kernel, K.copy(), [6], True)[0::2, 1::2]))",
            "gold_call": "float(np.trace(_preserved(_oracle_condition_kernel, K.copy(), [6], True)[0::2, 1::2]))",
        },
        {
            "setup": (status + "K = np.zeros((8, 8))\n") + preserve,
            "call": "_status(lambda: _preserved(condition_kernel, K, [1], True))",
            "gold_call": "_status(lambda: _preserved(_oracle_condition_kernel, K, [1], True))",
        },
        {
            "setup": (status + "K = np.zeros((8, 8))\n") + preserve,
            "call": "_status(lambda: _preserved(condition_kernel, K, [2, 2], False))",
            "gold_call": "_status(lambda: _preserved(_oracle_condition_kernel, K, [2, 2], False))",
        },
    ]
