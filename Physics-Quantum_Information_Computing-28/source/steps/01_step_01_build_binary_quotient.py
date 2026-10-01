"""
Construct a canonical binary quotient that separates syndrome and logical coordinates.

Return the canonical binary quotient basis and its inverse for the supplied ordered checks, stabilizers, and logical generators. Work over the binary field. Use left-to-right pivot selection, set nonpivot entries of the chosen dual rows to zero, and order quotient coordinates as syndrome bits followed by two logical bits, most significant bit first. These choices fix the returned basis when the syndrome is nonzero.

Returns
-------
An integer array of shape (2, n, n) contains the ordered basis and its binary inverse.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_binary_quotient(
    checks: "np.ndarray", stabilizers: "np.ndarray", logicals: "np.ndarray"
) -> "np.ndarray":
    r"""Construct a canonical binary quotient that separates syndrome and logical
    coordinates.

    Parameters
    ----------
    checks : np.ndarray
        Binary array $H$ of shape $(m,n)$, $1\le m\le3$ and $m+2\le n\le10$.
    stabilizers : np.ndarray
        Binary array $S$ of shape $(n-m-2,n)$.
    logicals : np.ndarray
        Binary array $L$ of shape $(2,n)$, ordered logical generators.

    Returns
    -------
    result : np.ndarray
        Integer array of shape $(2,n,n)$ containing $B$ and $B^{-1}$.

    Raises
    ------
    ValueError
        If arrays are nonbinary, dimensions disagree, checks are dependent,
        a supplied generator changes syndrome, or the complete basis is singular.

    Notes
    -----
    All quantities are dimensionless. No randomness is used.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _binary(a, ndim, name):
    a = np.asarray(a)
    if a.ndim != ndim or not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must be a binary array with {ndim} axes")
    return a.astype(np.int64, copy=True)


def _real(a, name):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        raise ValueError(f"{name} must be real")
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a.copy()


def _int(a, lo, hi, name):
    if (
        isinstance(a, (bool, np.bool_))
        or not isinstance(a, (int, np.integer))
        or not lo <= a <= hi
    ):
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return int(a)


def _prob(a, hi, name):
    a = _real(a, name)
    if a.ndim or not 0 < a <= hi:
        raise ValueError(f"{name} must be a scalar in (0, {hi}]")
    return float(a)


def _words(n):
    return (np.arange(1 << n)[:, None] >> np.arange(n - 1, -1, -1)) & 1


def _rref(a):
    a = a.copy()
    pivots = []
    row = 0
    for col in range(a.shape[1]):
        found = np.flatnonzero(a[row:, col])
        if not found.size:
            continue
        pivot = row + found[0]
        a[[row, pivot]] = a[[pivot, row]]
        for i in range(a.shape[0]):
            if i != row and a[i, col]:
                a[i] ^= a[row]
        pivots.append(col)
        row += 1
        if row == a.shape[0]:
            break
    return a, pivots


def _inverse(a):
    n = len(a)
    b, pivots = _rref(np.concatenate((a, np.eye(n, dtype=int)), axis=1))
    if pivots[:n] != list(range(n)):
        raise ValueError("binary basis must be invertible")
    return b[:, n:]


def _quotient(q, m):
    q = _binary(q, 3, "quotient")
    m = _int(m, 1, 3, "check_count")
    if q.shape[0] != 2 or q.shape[1] != q.shape[2] or not m + 2 <= q.shape[1] <= 10:
        raise ValueError("quotient must have shape (2,n,n), m+2 <= n <= 10")
    if not np.array_equal((q[0] @ q[1]) % 2, np.eye(q.shape[1], dtype=int)):
        raise ValueError("quotient matrices must be inverses over GF(2)")
    return q, m


def _oracle_build_binary_quotient(
    checks: "np.ndarray", stabilizers: "np.ndarray", logicals: "np.ndarray"
) -> "np.ndarray":
    h = _binary(checks, 2, "checks")
    s = _binary(stabilizers, 2, "stabilizers")
    logical_rows = _binary(logicals, 2, "logicals")
    m, n = h.shape
    if (
        not 1 <= m <= 3
        or not m + 2 <= n <= 10
        or s.shape != (n - m - 2, n)
        or logical_rows.shape != (2, n)
    ):
        raise ValueError("incompatible check, stabilizer and two-logical dimensions")
    if np.any((h @ np.concatenate((s, logical_rows)).T) % 2):
        raise ValueError("stabilizers and logicals must preserve the syndrome")
    _, pivots = _rref(h)
    if len(pivots) != m:
        raise ValueError("checks must be independent")
    pure = np.zeros((m, n), dtype=int)
    pure[:, pivots] = _inverse(h[:, pivots]).T
    basis = np.concatenate((pure, logical_rows, s))
    inverse = _inverse(basis)
    return np.stack((basis, inverse))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                r"""import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array(
    [
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
    ],
    dtype=int,
)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
"""
            ),
            "call": (
                "build_binary_quotient(checks[0].copy(), stabilizers[0].c"
                "opy(), logicals[0].copy())"
            ),
            "gold_call": (
                "_oracle_build_binary_quotient(checks[0].copy(), stabiliz"
                "ers[0].copy(), logicals[0].copy())"
            ),
            "tol": 0,
        },
        {
            "setup": (
                r"""import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array(
    [
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
    ],
    dtype=int,
)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
"""
            ),
            "call": (
                "build_binary_quotient(checks[1].copy(), stabilizers[1].c"
                "opy(), logicals[1].copy())"
            ),
            "gold_call": (
                "_oracle_build_binary_quotient(checks[1].copy(), stabiliz"
                "ers[1].copy(), logicals[1].copy())"
            ),
            "tol": 0,
        },
        {
            "setup": (
                r"""import numpy as np

checks = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
stabilizers = np.array(
    [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], dtype=int
)
logicals = np.array(
    [
        [[1, 1, 0, 0], [1, 0, 1, 0]],
        [[1, 0, 1, 0], [1, 1, 0, 0]],
    ],
    dtype=int,
)
detectors = np.array([[[1], [0]], [[0], [1]]], dtype=int)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
"""
            ),
            "call": (
                "build_binary_quotient(checks[0].copy(), stabilizers[0].c"
                "opy(), logicals[0].copy())"
            ),
            "gold_call": (
                "_oracle_build_binary_quotient(checks[0].copy(), stabiliz"
                "ers[0].copy(), logicals[0].copy())"
            ),
            "tol": 0,
        },
        {
            "setup": (
                r"""import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array(
    [
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
    ],
    dtype=int,
)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
perm = np.array([4, 7, 1, 5, 0, 6, 2, 3])
"""
            ),
            "call": (
                "build_binary_quotient(checks[0][:,perm].copy(), stabiliz"
                "ers[0][:,perm].copy(), logicals[0][:,perm].copy())"
            ),
            "gold_call": (
                "_oracle_build_binary_quotient(checks[0][:,perm].copy(), "
                "stabilizers[0][:,perm].copy(), logicals[0][:,perm].copy("
                "))"
            ),
            "tol": 0,
        },
        {
            "setup": (
                r"""import numpy as np

checks = np.array(
    [
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
    ],
    dtype=int,
)
stabilizers = np.array(
    [
        [
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0],
        ],
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
        ],
    ],
    dtype=int,
)
logicals = np.array(
    [
        [
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
    ],
    dtype=int,
)
detectors = np.array(
    [
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
    ],
    dtype=int,
)
counts = np.array([[930, 70], [30, 970]], dtype=int)
p = 0.08
pm = 0.06
tie_tol = 1e-12
checks[0, 1] = checks[0, 0]


def _raises(fn):
    try:
        fn(
            checks[0].copy(),
            stabilizers[0].copy(),
            logicals[0].copy(),
        )
    except ValueError:
        return 1
    return 0
"""
            ),
            "call": ("_raises(build_binary_quotient)"),
            "gold_call": ("_raises(_oracle_build_binary_quotient)"),
            "tol": 0,
        },
    ]
