"""
Evaluates the electrostatic potential at the reference ion of a finite crystal of oppositely charged pairs.

Each unit cell of the crystal carries one positive and one negative unit charge separated by a fixed displacement, the negative charge at the cell origin. The quantity is the electrostatic potential at the position of the negative ion of the central cell (not a potential energy), in units where the Coulomb prefactor is one: a positive unit charge at distance d contributes +1/d and a negative unit charge -1/d. It is the sum over every cell of the sample of +1/|r + n| for that cell's positive charge and -1/|n| for the negative charge at its origin n, the central cell (n = 0) contributing only +1/|r|. The sample is the parallelepiped of the previous step.

Returns
-------
A numpy float64 array of shape (3,) holding the finite lattice sum at size p, at size p-1, and their difference, in inverse length units (the reciprocal of the unit in which r and l are given).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def finite_lattice_sum(r: "np.ndarray", p: int, E: "np.ndarray", l: float) -> "np.ndarray":
    """Evaluates the electrostatic potential at the reference ion of a finite crystal of oppositely charged pairs.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, an
            absolute length in the same units as l, not a multiple of the lattice constant (with l = 2 and
            r = (0.25, 0, 0) the pair separation is 0.25 and the displacement is one eighth of the lattice
            constant); must be non-zero.
        p: non-negative integer, the size index of the sample; p = 0 is the central cell alone.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample, with the membership rule of the previous step.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (3,), in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l): the sum for size index p, the sum for size index p-1
        (equal to the first entry when p is zero), and their difference.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if p is not a
            non-negative integer; if E is not a 3x3 array of integer values with non-zero determinant; or if l
            is not a positive finite real number.
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


def _oracle_finite_lattice_sum(r: "np.ndarray", p: int, E: "np.ndarray", l: float) -> "np.ndarray":
    r = _vec3(r, "r")
    p = _nonneg_int(p, "p")
    A = _edges(E, "E")
    l = _pos_float(l, "l")
    if float(np.sqrt(r @ r)) <= 0.0:
        raise ValueError("invalid r")
    cur = _pair_sum_E(r, p, A, l)
    prev = _pair_sum_E(r, p - 1, A, l) if p >= 1 else cur
    return np.array([cur, prev, cur - prev], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\n',
         'call': 'finite_lattice_sum(np.array([0.5, 0.5, 0.5]), 2, np.eye(3), 1.0)',
         'gold_call': '_oracle_finite_lattice_sum(np.array([0.5, 0.5, 0.5]), 2, np.eye(3), 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'finite_lattice_sum(np.array([0.37, 0.21, 0.13]), 3, E, 1.0)',
         'gold_call': '_oracle_finite_lattice_sum(np.array([0.37, 0.21, 0.13]), 3, E, 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# a lattice constant of 2 with the displacement given in the same absolute units\n',
         'call': 'finite_lattice_sum(np.array([0.25, 0.0, 0.0]), 2, np.diag([7, 5, 3]), 2.0)',
         'gold_call': '_oracle_finite_lattice_sum(np.array([0.25, 0.0, 0.0]), 2, np.diag([7, 5, 3]), 2.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# boundary: the central cell alone, where the sum is the bare pair term\n',
         'call': 'finite_lattice_sum(np.array([0.37, 0.21, 0.13]), 0, np.eye(3), 1.0)',
         'gold_call': '_oracle_finite_lattice_sum(np.array([0.37, 0.21, 0.13]), 0, np.eye(3), 1.0)',
         'tol': 1e-09},
        {'setup': 'import numpy as np\n# invalid input: a vanishing displacement must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: finite_lattice_sum(np.array([0.0, 0.0, 0.0]), 2, np.eye(3), 1.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_finite_lattice_sum(np.array([0.0, 0.0, 0.0]), 2, np.eye(3), 1.0))'},
    ]
