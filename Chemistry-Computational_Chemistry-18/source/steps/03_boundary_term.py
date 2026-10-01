"""
Evaluates the non-periodic boundary contribution that the shape of a macroscopic crystal imposes on the reference ion.

In the limit of a macroscopic crystal the charges of the outermost layer no longer recede fast enough for their effect to vanish. What survives is fixed entirely by the proportions of the sample: it is the interaction of the reference dipole with the uniform polarization of the sample, one dipole per unit cell, nu_b(r|Omega) = (1/2V) times the integral over the region Omega occupied by the sample of (r . grad)^2 (1/|x|) d^3x, with V the volume of the unit cell. The value does not depend on the size of the sample, only on its proportions, and the second derivative is understood in the sense that includes its point term at the origin, so that for a sample with three equal perpendicular edges the integral evaluates to -2 pi (r . r)/(3V). No closed form is assumed for other shapes: the step must evaluate the defining integral for the parallelepiped spanned by the columns of E, accurate to at least ten significant digits.

Returns
-------
A numpy float64 array of shape (3,) holding the boundary term of the sample, the equal-edge value, and their difference.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def boundary_term(r: "np.ndarray", E: "np.ndarray", V: float) -> "np.ndarray":
    """Evaluates the non-periodic boundary contribution that the shape of a macroscopic crystal imposes on the reference ion.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as the cube root of V.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample; only the proportions of the parallelepiped they span matter.
        V: positive float, the volume of the unit cell.

    Returns:
        A numpy float64 array of shape (3,), in inverse length units, the reciprocal of the unit in which r is
        given: the boundary term for the sample spanned by the columns of E, the value -2 pi (r . r)/(3V) of a
        sample with three equal perpendicular edges, and their difference.

    Raises:
        ValueError: if r is not a finite real vector of length three; if E is not a 3x3 array of integer
            values with non-zero determinant; if V is not a positive finite real number; or if the boundary
            term comes out positive.
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


def _oracle_boundary_term(r: "np.ndarray", E: "np.ndarray", V: float) -> "np.ndarray":
    r = _vec3(r, "r")
    A = _edges(E, "E")
    V = _pos_float(V, "V")
    D = _boundary_tensor(A)
    nb = -float(r @ D @ r) / (2.0 * V)
    iso = -2.0 * np.pi * float(r @ r) / (3.0 * V)
    if nb > 0.0:
        raise ValueError("the boundary term must not be positive")
    return np.array([nb, iso, nb - iso], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\n',
         'call': 'boundary_term(np.array([0.5, 0.5, 0.5]), np.eye(3), 1.0)',
         'gold_call': '_oracle_boundary_term(np.array([0.5, 0.5, 0.5]), np.eye(3), 1.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'boundary_term(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'gold_call': '_oracle_boundary_term(np.array([0.37, 0.21, 0.13]), E, 1.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# a box with perpendicular edges and a unit cell of volume 8\n',
         'call': 'boundary_term(np.array([0.25, 0.1, 0.0]), np.diag([3, 1, 1]), 8.0)',
         'gold_call': '_oracle_boundary_term(np.array([0.25, 0.1, 0.0]), np.diag([3, 1, 1]), 8.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# boundary: a displacement along one axis of a sample sheared in the other two\nE = np.array([[1, 0, 0], [0, 3, 2], [0, 0, 3]])\n',
         'call': 'boundary_term(np.array([0.0, 0.0, 0.4]), E, 1.0)',
         'gold_call': '_oracle_boundary_term(np.array([0.0, 0.0, 0.4]), E, 1.0)',
         'tol': 1e-10},
        {'setup': 'import numpy as np\n# invalid input: a vanishing unit-cell volume must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: boundary_term(np.array([0.5, 0.5, 0.5]), np.eye(3), 0.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_boundary_term(np.array([0.5, 0.5, 0.5]), np.eye(3), 0.0))'},
    ]
