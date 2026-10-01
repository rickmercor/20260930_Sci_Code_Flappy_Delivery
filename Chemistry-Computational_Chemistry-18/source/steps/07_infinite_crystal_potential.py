"""
Evaluates the potential of a macroscopic crystal of prescribed proportions from the bulk and boundary parts and compares the finite sum of the same shape with it.

Once the bulk part has been extracted from a cubic sample it can be combined with the boundary term belonging to any chosen proportions, giving the macroscopic value for that shape without summing a large crystal. The finite sum of the same shape at a modest size differs from that value by its finite-size part, which is reported scaled by the square of the number 2p+1 of cells along an edge.

Returns
-------
A numpy float64 array of shape (5,) holding the macroscopic potential of the prescribed proportions, its bulk and boundary parts, the finite sum at size p and the scaled residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infinite_crystal_potential(r: "np.ndarray", E: "np.ndarray", p: int, l: float) -> "np.ndarray":
    """Evaluates the potential of a macroscopic crystal of prescribed proportions from the bulk and boundary parts and compares the finite sum of the same shape with it.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample and so fix the proportions of the macroscopic sample.
        p: positive integer, the size index used both for the bulk extraction from the cubic sample and for
            the finite sum of the prescribed shape.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (5,), in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l): the macroscopic potential
        for the prescribed proportions, the bulk part, the boundary part of that shape, the finite lattice
        sum of the same shape at size index p, and the difference between that sum and the macroscopic value
        multiplied by (2p+1)^2.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if E is not a 3x3
            array of integer values with non-zero determinant; if p is not a positive integer; if l is not a
            positive finite real number; or if the macroscopic potential comes out non-positive.
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


def _oracle_infinite_crystal_potential(r: "np.ndarray", E: "np.ndarray", p: int, l: float) -> "np.ndarray":
    r = _vec3(r, "r")
    A = _edges(E, "E")
    p = _pos_int(p, "p")
    l = _pos_float(l, "l")
    bulk = _oracle_bulk_pair_potential(r, p, l)[0]
    nb = _oracle_boundary_term(r, A, l ** 3)[0]
    inf = bulk + nb
    if inf <= 0.0:
        raise ValueError("the macroscopic potential came out non-positive")
    fin = _oracle_finite_lattice_sum(r, p, A, l)[0]
    return np.array([inf, bulk, nb, fin, (fin - inf) * (2 * p + 1) ** 2], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'infinite_crystal_potential(np.array([0.37, 0.21, 0.13]), E, 6, 1.0)',
         'gold_call': '_oracle_infinite_crystal_potential(np.array([0.37, 0.21, 0.13]), E, 6, 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# a box elongated along the displacement\n',
         'call': 'infinite_crystal_potential(np.array([0.5, 0.0, 0.0]), np.diag([3, 1, 1]), 5, 1.0)',
         'gold_call': '_oracle_infinite_crystal_potential(np.array([0.5, 0.0, 0.0]), np.diag([3, 1, 1]), 5, 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# boundary: the cube itself, where the scaled difference is the closed-form finite-size coefficient\n',
         'call': 'infinite_crystal_potential(np.array([0.25, 0.15, 0.05]), np.eye(3), 4, 2.0)',
         'gold_call': '_oracle_infinite_crystal_potential(np.array([0.25, 0.15, 0.05]), np.eye(3), 4, 2.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# invalid input: a singular edge matrix must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: infinite_crystal_potential(np.array([0.37, 0.21, 0.13]), np.zeros((3, 3)), 6, 1.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_infinite_crystal_potential(np.array([0.37, 0.21, 0.13]), np.zeros((3, 3)), 6, 1.0))'},
    ]
