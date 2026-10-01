"""
Evaluates analytically the coefficient of the leading finite-size term that the termination of a sheared sample adds to the multipole construction.

The second-order (dipole-dipole) term of the multipole series, [3 (r . n)^2 - (r . r) |n|^2] / (2 |n|^5), summed over the cells of a sample, differs from its continuum integral by Euler-Maclaurin corrections at the termination of the sample. For a boundary made entirely of faces lying midway between lattice planes these corrections cancel at order (2p+1)^(-2), because the summand is harmonic away from the origin; for a sample whose edge matrix has the form [[a, b, 0], [0, c, 0], [0, 0, d]] with a, c, d odd and b non-zero, the two faces of the first edge direction cut the lattice in a staircase whose boundary phase, the fractional part of the x limit (b/c) n_y +- a(2p+1)/2, cycles with n_y with period c, and a (2p+1)^(-2) term survives. This step evaluates that coefficient exactly from the iterated Euler-Maclaurin expansion of the three nested index sums: the midpoint second-order terms of the four faces on lattice midplanes, the phase-averaged second-order term of the two sheared faces, the edge terms where the sheared faces meet the faces of the second edge direction (from the n_y dependence of the x limits), and the summation-by-parts boundary terms of the zero-mean part of the first-order boundary weight, all evaluated on the unit-scale shape. The result must agree with the limit of the fitted coefficient of the previous step but is required to 1e-11, which a fit over accessible sizes cannot deliver.

Returns
-------
A numpy float64 array of shape (2,) holding the analytic termination coefficient of the sample and that of the box of the same edge lengths.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def termination_coefficient(r: "np.ndarray", E: "np.ndarray", l: float) -> "np.ndarray":
    """Evaluates analytically the coefficient of the leading finite-size term that the termination of a sheared sample adds to the multipole construction.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values of the form [[a, b, 0], [0, c, 0], [0, 0, d]] with a,
            c and d odd positive integers and b any integer (b = 0 is a box); any other edge matrix is
            rejected.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (2,), in inverse length units, the reciprocal of the unit in which r
        and l are given: the (2p+1)^(-2) coefficient of the second-order lattice sum over the sample minus
        its limit, from the Euler-Maclaurin evaluation, and the same quantity for the box with edge lengths
        a, c, d (b = 0), which vanishes.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if E is not a 3x3
            array of integer values of the stated form with odd positive a, c, d; or if l is not a positive
            finite real number.
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


def _sheared_family(E, name):
    """Edge matrix of the form [[a, b, 0], [0, c, 0], [0, 0, d]] with a, c, d odd positive integers."""
    A = _edges(E, name)
    a, b, c, d = float(A[0, 0]), float(A[0, 1]), float(A[1, 1]), float(A[2, 2])
    off = np.array([A[0, 2], A[1, 0], A[1, 2], A[2, 0], A[2, 1]])
    if np.any(off != 0.0) or min(a, c, d) <= 0.0 or int(a) % 2 == 0 or int(c) % 2 == 0 or int(d) % 2 == 0:
        raise ValueError("invalid " + name)
    return a, b, c, d


def _g2(x, r):
    """Second-order (dipole-dipole) term of the multipole series, [3 (r.x)^2 - r^2 x^2] / (2 |x|^5)."""
    rr = float(r @ r)
    rx = np.einsum("...k,k->...", x, r)
    x2 = np.einsum("...k,...k->...", x, x)
    return (3.0 * rx * rx - rr * x2) / (2.0 * x2 ** 2.5)


def _grad_g2(x, r):
    rr = float(r @ r)
    rx = np.einsum("...k,k->...", x, r)
    x2 = np.einsum("...k,...k->...", x, x)
    t1 = (3.0 * rx[..., None] * r[None, :] - rr * x) / x2[..., None] ** 2.5
    t2 = 5.0 * (3.0 * rx * rx - rr * x2)[..., None] * x / (2.0 * x2[..., None] ** 3.5)
    return t1 - t2


def _gl_nodes(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def _em_q2(a, b, c, d, r, n=160):
    """K^-2 coefficient of the second-order lattice sum minus its limit, unit-scale sheared box.

    Iterated Euler-Maclaurin over n_x (limits (b/c) n_y +- a/2, boundary phase period c), then n_y
    and n_z (midplane faces): midpoint face terms, the phase-averaged second-order term of the
    sheared faces, the edge term from the n_y dependence of the n_x limits, and the summation-by-
    parts boundary term of the zero-mean first-order weight.
    """
    s = b / c
    y, wy = _gl_nodes(n, -c / 2, c / 2)
    z, wz = _gl_nodes(n, -d / 2, d / 2)
    u, wu = _gl_nodes(n, -0.5, 0.5)
    term1 = 0.0
    for sy in (1.0, -1.0):
        yy = sy * c / 2
        X = s * yy + a * u[:, None] + 0.0 * z[None, :]
        Z = 0.0 * u[:, None] + z[None, :]
        P = np.stack([X, np.full_like(X, yy), Z], axis=-1)
        term1 += sy * float(np.sum((a * wu)[:, None] * wz[None, :] * _grad_g2(P, r)[..., 1]))
    for sz in (1.0, -1.0):
        zz = sz * d / 2
        X = s * y[None, :] + a * u[:, None]
        Y = 0.0 * u[:, None] + y[None, :]
        P = np.stack([X, Y, np.full_like(X, zz)], axis=-1)
        term1 += sz * float(np.sum((a * wu)[:, None] * wy[None, :] * _grad_g2(P, r)[..., 2]))
    term1 *= -1.0 / 24.0
    cc = int(round(c))
    k = np.arange(cc)
    phi = np.mod(int(round(b)) * k / cc + 0.5, 1.0)
    b1 = phi - 0.5
    b2mean = float(np.mean(phi * phi - phi + 1.0 / 6.0))
    Y2, Z2 = np.meshgrid(y, z, indexing="ij")
    W2 = wy[:, None] * wz[None, :]
    Pp = np.stack([s * Y2 + a / 2, Y2, Z2], axis=-1)
    Pm = np.stack([s * Y2 - a / 2, Y2, Z2], axis=-1)
    term2 = 0.5 * b2mean * float(np.sum(W2 * (_grad_g2(Pp, r)[..., 0] - _grad_g2(Pm, r)[..., 0])))

    def _f1(yy):
        Pp = np.stack([np.full_like(z, s * yy + a / 2), np.full_like(z, yy), z], axis=-1)
        Pm = np.stack([np.full_like(z, s * yy - a / 2), np.full_like(z, yy), z], axis=-1)
        return float(np.sum(wz * (_g2(Pp, r) - _g2(Pm, r))))
    f_top, f_bot = _f1(c / 2), _f1(-c / 2)
    term3 = -(1.0 / 24.0) * s * (f_top - f_bot)
    start = int((-(cc - 1) // 2) % cc)
    wbar = float(np.mean(np.cumsum(b1[(start + np.arange(cc)) % cc])))
    term4 = wbar * (f_top - f_bot)
    return term1 + term2 + term3 + term4


def _oracle_termination_coefficient(r: "np.ndarray", E: "np.ndarray", l: float) -> "np.ndarray":
    r = _vec3(r, "r")
    a, b, c, d = _sheared_family(E, "E")
    l = _pos_float(l, "l")
    if float(np.sqrt(r @ r)) <= 0.0:
        raise ValueError("invalid r")
    rr = r / l
    q2 = _em_q2(a, b, c, d, rr) / l
    q2box = _em_q2(a, 0.0, c, d, rr) / l
    return np.array([q2, q2box], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'termination_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'gold_call': '_oracle_termination_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# the shear reversed\nE = np.array([[7, -3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'termination_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'gold_call': '_oracle_termination_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# a short sample whose boundary-phase pattern leaves a non-zero first-order boundary term\nE = np.array([[3, 1, 0], [0, 3, 0], [0, 0, 1]])\n',
         'call': 'termination_coefficient(np.array([0.25, 0.15, 0.05]), E, 1.0)',
         'gold_call': '_oracle_termination_coefficient(np.array([0.25, 0.15, 0.05]), E, 1.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# a lattice constant of 2 with the displacement in the same absolute units\nE = np.array([[5, 2, 0], [0, 3, 0], [0, 0, 3]])\n',
         'call': 'termination_coefficient(np.array([0.5, 0.3, 0.1]), E, 2.0)',
         'gold_call': '_oracle_termination_coefficient(np.array([0.5, 0.3, 0.1]), E, 2.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# boundary: a box, where the coefficient vanishes\n',
         'call': 'termination_coefficient(np.array([0.37, 0.21, 0.13]), np.diag([7, 5, 3]), 1.0)',
         'gold_call': '_oracle_termination_coefficient(np.array([0.37, 0.21, 0.13]), np.diag([7, 5, 3]), 1.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# invalid input: a shear outside the stated form must raise ValueError\nE = np.array([[3, 0, 0], [0, 3, 0], [1, 0, 3]])\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: termination_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_termination_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0))'},
    ]
