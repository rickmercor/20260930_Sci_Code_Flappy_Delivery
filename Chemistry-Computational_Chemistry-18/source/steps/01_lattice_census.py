"""
Enumerates the lattice vectors of a finite crystal of a given size index and edge matrix and reports its census.

A finite sample of the simple cubic lattice is fixed by a size index p and by an integer matrix E whose columns are the three edge directions of the sample. A cell with lattice vector n belongs to the sample when the solution u of E u = n has every component within (2p+1)/2 in magnitude, so the sample is a parallelepiped centred on the central cell that holds |det E| (2p+1)^3 cells, the central cell included, and every cell has a partner at -n. For E = I it is the cube of (2p+1)^3 cells; for a diagonal E it is a box with E[i, i] (2p+1) cells along axis i; for a non-diagonal E two edges are not perpendicular and the sample is sheared.

Returns
-------
A numpy float64 array of shape (6,) holding the census of the finite sample.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lattice_census(p: int, E: "np.ndarray") -> "np.ndarray":
    """Enumerates the lattice vectors of a finite crystal of a given size index and edge matrix and reports its census.

    Args:
        p: positive integer, the size index of the sample; the census needs at least one cell besides the
            central one, so p = 0 is rejected.
        E: array-like of shape (3, 3) holding integer values in any numeric dtype, with non-zero
            determinant, whose COLUMNS are the edge directions of the sample; a cell n belongs to the sample
            when every component of the solution u of E u = n satisfies |u_i| <= (2p+1)/2.

    Returns:
        A numpy float64 array of shape (6,): the number of lattice vectors of the sample excluding the origin,
        the cell count |det E| (2p+1)^3 predicted by the volume of the sample (origin included), the largest
        vector norm, the sum of the inverse norms, and the largest absolute cell index along x and along z.

    Raises:
        ValueError: if p is not a positive integer; or if E is not a 3x3 array of integer values with
            non-zero determinant.
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


def _oracle_lattice_census(p: int, E: "np.ndarray") -> "np.ndarray":
    p = _pos_int(p, "p")
    A = _edges(E, "E")
    n, nn = _lattice_E(p, A)
    if float(np.max(np.abs(np.sum(n, axis=0)))) > 0.5:
        raise ValueError("the finite sample is not centrosymmetric about the origin")
    expected = abs(float(np.linalg.det(A))) * (2 * p + 1) ** 3
    return np.array([float(n.shape[0]), expected, float(np.max(nn)), float(np.sum(1.0 / nn)),
                     float(np.max(np.abs(n[:, 0]))), float(np.max(np.abs(n[:, 2])))], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\n',
         'call': 'lattice_census(1, np.eye(3))',
         'gold_call': '_oracle_lattice_census(1, np.eye(3))',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'lattice_census(1, E)',
         'gold_call': '_oracle_lattice_census(1, E)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# boundary: a box elongated along x, three times as many cells along x as along y and z\n',
         'call': 'lattice_census(2, np.diag([3, 1, 1]))',
         'gold_call': '_oracle_lattice_census(2, np.diag([3, 1, 1]))',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# edge case: the same sheared sample passed as a float array of integer values\nE = np.array([[7.0, 3.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0]])\n',
         'call': 'lattice_census(2, E)',
         'gold_call': '_oracle_lattice_census(2, E)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# invalid input: a singular edge matrix must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: lattice_census(1, np.zeros((3, 3))))',
         'gold_call': '_catches_value_error(lambda: _oracle_lattice_census(1, np.zeros((3, 3))))'},
    ]
