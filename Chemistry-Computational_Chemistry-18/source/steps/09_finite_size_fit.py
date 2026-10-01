"""
Measures the actual leading finite-size coefficient of a sample from its direct sums at several sizes, and the corresponding coefficient of the second-order multipole term alone.

The finite lattice sums of a sample approach the macroscopic value of the same proportions with residuals that decay as even inverse powers of the number 2p+1 of cells along an edge. Fitting the residuals against the macroscopic value at several size indices to a (2p+1)^(-2) + b (2p+1)^(-4) measures the actual leading coefficient a without assuming the multipole construction. The same fit applied to the lattice sum over the sample of the second-order multipole term alone, [3 (r . n)^2 - (r . r) |n|^2] / (2 |n|^5) with r and n in units of the lattice constant, summed over every cell of the sample other than the central one and expressed in the same inverse length units as the potentials, isolates how much of that coefficient comes from the term the construction treats as size independent.

Returns
-------
A numpy float64 array of shape (5,) holding the fitted leading and next finite-size coefficients of the sample and the fitted coefficients of its second-order sums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def finite_size_fit(r: "np.ndarray", E: "np.ndarray", sizes: "np.ndarray", p_bulk: int, l: float) -> "np.ndarray":
    """Measures the actual leading finite-size coefficient of a sample from its direct sums at several sizes, and the corresponding coefficient of the second-order multipole term alone.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample.
        sizes: array-like of at least three distinct positive integers, the size indices at which the
            sample is summed.
        p_bulk: positive integer, the size index of the cubic sample from which the bulk part of the
            macroscopic value is extracted.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (5,), in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l): the least-squares coefficients a and b of
        a (2p+1)^(-2) + b (2p+1)^(-4) fitted to the residuals, finite sum minus macroscopic value, over the
        given sizes; the least-squares coefficients q2 and q4 of q0 + q2 (2p+1)^(-2) + q4 (2p+1)^(-4)
        fitted to the second-order lattice sums over the same sizes; and q0.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if E is not a 3x3
            array of integer values with non-zero determinant; if sizes holds fewer than three entries or any
            entry that is not a distinct positive integer; if p_bulk is not a positive integer; or if l is
            not a positive finite real number.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _real(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError("invalid " + name)
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("invalid " + name)
    return v


def _pos_float(v, name):
    v = _real(v, name)
    if v <= 0.0:
        raise ValueError("invalid " + name)
    return v


def _nonneg_int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0:
        raise ValueError("invalid " + name)
    return int(v)


def _pos_int(v, name):
    v = _nonneg_int(v, name)
    if v < 1:
        raise ValueError("invalid " + name)
    return v


def _vec3(r, name):
    a = np.asarray(r, dtype=np.float64)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError("invalid " + name)
    return a


def _edges(E, name):
    """3x3 matrix of integer values (any numeric dtype) whose COLUMNS are the edge directions."""
    A = np.asarray(E, dtype=np.float64)
    if A.shape != (3, 3) or not np.all(np.isfinite(A)):
        raise ValueError("invalid " + name)
    if not np.allclose(A, np.round(A), atol=1e-9):
        raise ValueError("invalid " + name)
    A = np.round(A)
    if abs(float(np.linalg.det(A))) < 0.5:
        raise ValueError("invalid " + name)
    return A


def _lattice_E(p, A):
    """Lattice vectors of the sample {n : max_i |(A^-1 n)_i| <= (2p+1)/2}, origin excluded."""
    K = (2 * p + 1) / 2.0
    Ainv = np.linalg.inv(A)
    ext = np.abs(A) @ np.array([K, K, K])
    rng = [np.arange(-int(np.floor(e)), int(np.floor(e)) + 1, dtype=np.float64) for e in ext]
    I, J, L = np.meshgrid(rng[0], rng[1], rng[2], indexing="ij")
    n = np.stack([I.ravel(), J.ravel(), L.ravel()], axis=1)
    u = np.einsum("ij,kj->ik", n, Ainv)
    n = n[np.max(np.abs(u), axis=1) <= K + 1e-9]
    nn = np.sqrt(np.einsum("ij,ij->i", n, n))
    keep = nn > 0.0
    return n[keep], nn[keep]


def _pair_sum_E(r, p, A, l):
    """Finite lattice sum nu(r, p | sample A) for one +/- pair per cell, in inverse length units."""
    n, nn = _lattice_E(p, A)
    rr = np.asarray(r, dtype=np.float64) / l
    d1 = np.sqrt(np.einsum("ij,ij->i", n + rr, n + rr))
    return (1.0 / np.sqrt(rr @ rr) + float(np.sum(1.0 / d1 - 1.0 / nn))) / l


def _face_grid(A, i, sgn, m):
    """Gauss-Legendre nodes/weights on the face u_i = sgn/2 of the unit sample spanned by A.

    Returns the points x (m*m, 3), the weights (m*m,) that include the face area, and the
    outward unit normal.
    """
    a = [A[:, k] for k in range(3)]
    j, k = [q for q in range(3) if q != i]
    nvec = np.cross(a[j], a[k])
    area = float(np.sqrt(nvec @ nvec))
    nhat = nvec / area
    if float(nhat @ a[i]) < 0.0:
        nhat = -nhat
    nhat = sgn * nhat
    g, w = np.polynomial.legendre.leggauss(m)
    g = 0.5 * g
    w = 0.5 * w
    S, T = np.meshgrid(g, g, indexing="ij")
    WS, WT = np.meshgrid(w, w, indexing="ij")
    x = (sgn * 0.5) * a[i][None, :] + S.ravel()[:, None] * a[j][None, :] + T.ravel()[:, None] * a[k][None, :]
    return x, (WS * WT).ravel() * area, nhat


def _boundary_tensor(A):
    """D_ij = surface integral over the unit sample of n_i x_j / |x|^3 (trace 4 pi)."""
    D = np.zeros((3, 3))
    for i in range(3):
        for sgn in (1.0, -1.0):
            x, w, nh = _face_grid(A, i, sgn, 96)
            x3 = np.sqrt(np.einsum("ij,ij->i", x, x)) ** 3
            D += np.einsum("i,ij->j", w / x3, x)[None, :] * nh[:, None]
    return D


def _sizes(sizes, name):
    s = np.asarray(sizes)
    if s.ndim != 1 or s.shape[0] < 3 or not np.all(np.isfinite(s.astype(np.float64))):
        raise ValueError("invalid " + name)
    if not np.all(s == np.round(s)) or np.any(s < 1) or len(set(int(v) for v in s)) != s.shape[0]:
        raise ValueError("invalid " + name)
    return [int(v) for v in s]


def _quadratic_sum_E(r, p, A, l):
    """Lattice sum over the sample of the second-order (dipole-dipole) multipole term, in inverse length units."""
    n, nn = _lattice_E(p, A)
    rr = np.asarray(r, dtype=np.float64) / l
    rn = np.einsum("ij,j->i", n, rr)
    return float(np.sum((3.0 * rn ** 2 - float(rr @ rr) * nn ** 2) / (2.0 * nn ** 5))) / l


def _oracle_finite_size_fit(r: "np.ndarray", E: "np.ndarray", sizes: "np.ndarray", p_bulk: int, l: float) -> "np.ndarray":
    r = _vec3(r, "r")
    A = _edges(E, "E")
    ps = _sizes(sizes, "sizes")
    p_bulk = _pos_int(p_bulk, "p_bulk")
    l = _pos_float(l, "l")
    inf = _oracle_infinite_crystal_potential(r, A, p_bulk, l)[0]
    K = np.array([2 * p + 1 for p in ps], dtype=np.float64)
    resid = np.array([_pair_sum_E(r, p, A, l) - inf for p in ps])
    X = np.stack([K ** -2, K ** -4], axis=1)
    c2, c4 = np.linalg.lstsq(X, resid, rcond=None)[0]
    quad = np.array([_quadratic_sum_E(r, p, A, l) for p in ps])
    Y = np.stack([np.ones_like(K), K ** -2, K ** -4], axis=1)
    qinf, q2, q4 = np.linalg.lstsq(Y, quad, rcond=None)[0]
    return np.array([c2, c4, q2, q4, qinf], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'finite_size_fit(np.array([0.37, 0.21, 0.13]), E, np.array([4, 6, 8]), 8, 1.0)',
         'gold_call': '_oracle_finite_size_fit(np.array([0.37, 0.21, 0.13]), E, np.array([4, 6, 8]), 8, 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# a box with perpendicular edges, where the second-order sums carry no inverse-square term\n',
         'call': 'finite_size_fit(np.array([0.37, 0.21, 0.13]), np.diag([7, 5, 3]), np.array([4, 6, 8]), 8, 1.0)',
         'gold_call': '_oracle_finite_size_fit(np.array([0.37, 0.21, 0.13]), np.diag([7, 5, 3]), np.array([4, 6, 8]), 8, 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# boundary: the cube with a lattice constant of 2, where the second-order sums vanish identically\n',
         'call': 'finite_size_fit(np.array([0.25, 0.15, 0.05]), np.eye(3), np.array([3, 4, 6, 8]), 8, 2.0)',
         'gold_call': '_oracle_finite_size_fit(np.array([0.25, 0.15, 0.05]), np.eye(3), np.array([3, 4, 6, 8]), 8, 2.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# invalid input: only two sizes cannot support the fit and must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: finite_size_fit(np.array([0.37, 0.21, 0.13]), np.eye(3), np.array([4, 6]), 8, 1.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_finite_size_fit(np.array([0.37, 0.21, 0.13]), np.eye(3), np.array([4, 6]), 8, 1.0))'},
    ]
