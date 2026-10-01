"""
Evaluates the Madelung constant of the caesium chloride structure from the corrected direct summation.

For a body-centred arrangement the pair displacement runs along the body diagonal of the cube, r = (0.5, 0.5, 0.5) in units of the lattice constant, and the Madelung constant is the bulk pair potential expressed in units of the nearest-neighbour separation sqrt(3)/2. Because the displacement is equally inclined to the three axes, the boundary contribution of a cubic sample is -pi/2 in units of the lattice constant whatever the size, and the published reference value of the constant is 1.76267477307098.

Returns
-------
A numpy float64 array of shape (5,) holding the caesium chloride Madelung constant, its deviation from the reference and its three contributions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def madelung_constant(p: int) -> "np.ndarray":
    """Evaluates the Madelung constant of the caesium chloride structure from the corrected direct summation.

    Args:
        p: positive integer, the size index of the cubic sample used for the extraction.

    Returns:
        A numpy float64 array of shape (5,), all in units of the nearest-neighbour separation: the Madelung
        constant, its deviation from the reference value 1.76267477307098, and the three signed contributions
        that add up to the constant, namely the finite lattice sum, the negative of the boundary term (a
        positive number here) and the negative of the finite-size term.

    Raises:
        ValueError: if p is not a positive integer.
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


def _oracle_madelung_constant(p: int) -> "np.ndarray":
    p = _pos_int(p, "p")
    r = np.array([0.5, 0.5, 0.5])
    s = np.sqrt(3.0) / 2.0          # nearest-neighbour separation in units of l
    ref = 1.76267477307098
    b = _oracle_bulk_pair_potential(r, p, 1.0)
    M = b[0] * s
    return np.array([M, M - ref, b[1] * s, -b[2] * s, -b[3] * s], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\n# boundary: the minimal 3x3x3 supercell\n',
         'call': 'madelung_constant(1)',
         'gold_call': '_oracle_madelung_constant(1)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n',
         'call': 'madelung_constant(5)',
         'gold_call': '_oracle_madelung_constant(5)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n',
         'call': 'madelung_constant(8)',
         'gold_call': '_oracle_madelung_constant(8)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# invalid input: a zero size index must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: madelung_constant(0))',
         'gold_call': '_catches_value_error(lambda: _oracle_madelung_constant(0))'},
    ]
