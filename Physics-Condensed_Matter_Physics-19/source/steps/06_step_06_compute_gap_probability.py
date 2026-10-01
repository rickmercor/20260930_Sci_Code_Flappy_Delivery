"""
Compute the probability that a discrete Pfaffian point process leaves every site of a given set empty.

Gap probabilities of Pfaffian point processes control the distribution of extreme particles, such as the largest shifted row of a random shape.

Returns
-------
float: the gap probability of the listed sites.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_gap_probability(kernel: np.ndarray, sites: list[int]) -> float:
    """Return the probability that no site in ``sites`` is occupied.

    ``kernel`` is a real skew-symmetric ``(2L, 2L)`` array whose rows and
    columns ``2x`` and ``2x + 1`` belong to site ``x = 0, ..., L - 1``; the
    Pfaffian of its principal submatrix on any set of distinct sites is the
    probability that all of them are occupied, with
    ``Pf([[0, a], [-a, 0]]) = a``. An empty ``sites`` gives 1.0. The input is
    not modified.

    Parameters
    ----------
    kernel : np.ndarray
        Skew-symmetric array of shape ``(2L, 2L)``.
    sites : list[int]
        Distinct site labels in ``0, ..., L - 1``.

    Returns
    -------
    float
        The probability that every listed site is empty.

    Raises
    ------
    ValueError
        If ``kernel`` is not a finite square array of even order, or
        ``sites`` contains repeated or out-of-range labels, non-integers or
        booleans.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_gap_probability(kernel: np.ndarray, sites: list[int]) -> float:
    """Reference implementation: the rejection branch of the sequential Pfaffian sampler."""
    import numpy as np

    matrix = np.asarray(kernel, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2
            or not np.all(np.isfinite(matrix))):
        raise ValueError("kernel must be a finite square array of even order")
    size = matrix.shape[0] // 2
    labels = list(sites)
    if (any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in labels)
            or len({int(s) for s in labels}) != len(labels)
            or any(not 0 <= int(s) < size for s in labels)):
        raise ValueError("sites must be distinct in-range integers")
    if not labels:
        return 1.0
    index = np.ravel([[2 * int(s), 2 * int(s) + 1] for s in labels])
    current = matrix[np.ix_(index, index)]
    probability = 1.0
    # Visit the sites one at a time, always taking the "empty" branch: the
    # conditional occupation is the (0, 1) entry, and the remaining kernel is
    # conditioned on the visited site being empty.
    while current.shape[0] > 2:
        occupation = current[0, 1]
        probability *= 1.0 - occupation
        if occupation == 1.0:
            return 0.0
        current = _oracle_condition_kernel(current, [0], False)
    probability *= 1.0 - current[0, 1]
    return float(probability)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
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
            "call": "_preserved(compute_gap_probability, K.copy(), [6, 7, 8, 9])",
            "gold_call": "_preserved(_oracle_compute_gap_probability, K.copy(), [6, 7, 8, 9])",
        },
        {
            "setup": (build + "K = _kern(0.5 ** (0.5 * np.arange(12.0)), 6)\n") + preserve,
            "call": "_preserved(compute_gap_probability, K.copy(), [2, 5, 9])",
            "gold_call": "_preserved(_oracle_compute_gap_probability, K.copy(), [2, 5, 9])",
        },
        {
            "setup": (build + "K = _kern(0.5 ** (0.5 * np.arange(12.0)), 6)\n") + preserve,
            "call": "_preserved(compute_gap_probability, K.copy(), [4])",
            "gold_call": "_preserved(_oracle_compute_gap_probability, K.copy(), [4])",
        },
        {
            "setup": (build + "K = _kern(0.4 ** (0.5 * np.arange(8.0)), 4)\n") + preserve,
            "call": "_preserved(compute_gap_probability, K.copy(), [])",
            "gold_call": "_preserved(_oracle_compute_gap_probability, K.copy(), [])",
        },
        {
            "setup": (build + "K = _kern(0.2 ** (0.5 * np.arange(16.0)), 6)\n") + preserve,
            "call": "_preserved(compute_gap_probability, K.copy(), list(range(8, 16)))",
            "gold_call": "_preserved(_oracle_compute_gap_probability, K.copy(), list(range(8, 16)))",
        },
        {
            "setup": (build + "K = _kern(0.6 ** (0.5 * np.arange(9.0)), 4)\n") + preserve,
            "call": "_preserved(compute_gap_probability, K.copy(), [8, 0, 4])",
            "gold_call": "_preserved(_oracle_compute_gap_probability, K.copy(), [8, 0, 4])",
        },
        {
            "setup": ("import numpy as np\nK = np.zeros((6, 6))\nK[0, 1], K[1, 0] = 1.0, -1.0\nK[2, 3], K[3, 2] = 0.3, -0.3\nK[4, 5], K[5, 4] = 0.6, -0.6\n") + preserve,
            "call": "_preserved(compute_gap_probability, K.copy(), [2, 0, 1])",
            "gold_call": "_preserved(_oracle_compute_gap_probability, K.copy(), [2, 0, 1])",
        },
        {
            "setup": (status + "K = np.zeros((8, 8))\n") + preserve,
            "call": "_status(lambda: _preserved(compute_gap_probability, K, [1, 1]))",
            "gold_call": "_status(lambda: _preserved(_oracle_compute_gap_probability, K, [1, 1]))",
        },
        {
            "setup": (status + "K = np.zeros((8, 8))\n") + preserve,
            "call": "_status(lambda: _preserved(compute_gap_probability, K, [4]))",
            "gold_call": "_status(lambda: _preserved(_oracle_compute_gap_probability, K, [4]))",
        },
    ]
