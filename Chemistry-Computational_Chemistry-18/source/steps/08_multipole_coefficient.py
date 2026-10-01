"""
Evaluates the coefficient of the leading finite-size term that the multipole construction predicts for a sample of prescribed proportions.

The finite-size term is built by expanding the contribution of every cell omitted from the finite sample, 1/|r + n| - 1/|n|, in multipoles about the origin and replacing the sum over the omitted cells by an integral over the region outside the sample, one cell per unit-cell volume V. The first-order term cancels between opposite cells, the second-order term integrates to a size-independent quantity that belongs to the boundary term, and the leading size-dependent term is the fourth-order one, whose exterior integral falls as the inverse square of the number 2p+1 of cells along an edge. The finite-size term is the finite sum minus the macroscopic value, nu(r, p | Omega) - nu(r, infinity | Omega), so the coefficient carries that sign: applied to the cube the construction must reproduce the closed form of the finite-size step, sign included, to the accuracy required here. This step returns the coefficient of (2p+1)^(-2) that the construction predicts for the parallelepiped spanned by the columns of E, accurate to better than 1e-9 in absolute terms.

Returns
-------
A numpy float64 array of shape (4,) holding the multipole prediction of the leading finite-size coefficient for the sample, the same for the cube, the cube closed form and their difference.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def multipole_coefficient(r: "np.ndarray", E: "np.ndarray", V: float) -> "np.ndarray":
    """Evaluates the coefficient of the leading finite-size term that the multipole construction predicts for a sample of prescribed proportions.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as the cube root of V.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample; only the proportions matter.
        V: positive float, the volume of the unit cell.

    Returns:
        A numpy float64 array of shape (4,), in inverse length units, the reciprocal of the unit in which r is given: the predicted coefficient
        of (2p+1)^(-2) for the sample spanned by the columns of E, the same construction applied to the cube,
        the closed-form cube coefficient of the finite-size step (its correction multiplied by (2p+1)^2), and
        the difference of the last two.

    Raises:
        ValueError: if r is not a finite real vector of length three; if E is not a 3x3 array of integer
            values with non-zero determinant; or if V is not a positive finite real number.
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


def _multipole_coefficient(A, r, V):
    """K^-2 coefficient of the continuum hexadecapole exterior integral, sample spanned by A.

    -(1/V) int_ext r^4 P4(cos) / |x|^5 d^3x over the exterior of the size-K sample (boundary at
    K l R(xhat)) equals t2 / K^2 with t2 = -(1/(2 V l^2)) sum_faces int h(xhat) (xhat.n) / |x|^4 dS
    over the faces of the unit-size sample, h = r^4 P4(rhat.xhat).
    """
    rr = float(r @ r)
    l = V ** (1.0 / 3.0)
    tot = 0.0
    for i in range(3):
        for sgn in (1.0, -1.0):
            x, w, nh = _face_grid(A, i, sgn, 96)
            nx = np.sqrt(np.einsum("ij,ij->i", x, x))
            xh = x / nx[:, None]
            c = np.einsum("ij,j->i", xh, r) / np.sqrt(rr)
            h = rr ** 2 * (35.0 * c ** 4 - 30.0 * c ** 2 + 3.0) / 8.0
            tot += float(np.sum(w * h * np.einsum("ij,j->i", xh, nh) / nx ** 4))
    # the size-K sample of a lattice of constant l has its boundary at K * l * R(xhat): one factor 1/l^2
    # from the radial integral on top of the 1/V of the cell density, i.e. r^4 / l^5 overall
    return -tot / (2.0 * V * l * l)


def _oracle_multipole_coefficient(r: "np.ndarray", E: "np.ndarray", V: float) -> "np.ndarray":
    r = _vec3(r, "r")
    A = _edges(E, "E")
    V = _pos_float(V, "V")
    l = V ** (1.0 / 3.0)
    t2 = _multipole_coefficient(A, r, V)
    t2c = _multipole_coefficient(np.eye(3), r, V)
    closed = _oracle_finite_size_term(r, 0, l)[1]
    return np.array([t2, t2c, closed, t2c - closed], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'multipole_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'gold_call': '_oracle_multipole_coefficient(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# a box with perpendicular edges of the same nominal lengths\n',
         'call': 'multipole_coefficient(np.array([0.37, 0.21, 0.13]), np.diag([7, 5, 3]), 1.0)',
         'gold_call': '_oracle_multipole_coefficient(np.array([0.37, 0.21, 0.13]), np.diag([7, 5, 3]), 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# boundary: the cube itself with a unit cell of volume 8, where the first entry equals the closed form\n',
         'call': 'multipole_coefficient(np.array([0.25, 0.1, 0.0]), np.eye(3), 8.0)',
         'gold_call': '_oracle_multipole_coefficient(np.array([0.25, 0.1, 0.0]), np.eye(3), 8.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# a sample sheared in the y-z plane with an axial displacement\nE = np.array([[1, 0, 0], [0, 3, 2], [0, 0, 3]])\n',
         'call': 'multipole_coefficient(np.array([0.0, 0.0, 0.4]), E, 1.0)',
         'gold_call': '_oracle_multipole_coefficient(np.array([0.0, 0.0, 0.4]), E, 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# invalid input: a vanishing unit-cell volume must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: multipole_coefficient(np.array([0.37, 0.21, 0.13]), np.eye(3), 0.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_multipole_coefficient(np.array([0.37, 0.21, 0.13]), np.eye(3), 0.0))'},
    ]
