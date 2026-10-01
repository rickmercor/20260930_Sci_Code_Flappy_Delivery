"""
Return the weak stress-divergence term of the source's momentum equation, tested against every velocity basis function on every element. It combines a volume contribution pairing the element stress with the gradient of the test function and a boundary contribution pairing the traction flux with the test function on each face. The sign each contribution carries follows from integrating the divergence by parts once; recover the arrangement from the paper. Use two-point Gauss-Legendre quadrature on the faces and the tensor-product two-by-two rule in the element. Raise ValueError unless nx and the element size are positive. Assemble by calling the earlier sub-problem functions.

This is the only term through which stress drives the velocity, so its face contributions are where neighbouring elements exchange momentum.

Returns
-------
return (nx, nx, 4, 2) float64: the weak divergence of the stress against each velocity basis function
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ldg_divergence(U, S, nx, h, a, b):
    """Arguments as for the strain sub-problem. Returns (nx, nx, 4, 2) float64,
    one two-component value per velocity node per element."""
    return np.zeros((nx, nx, 4, 2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: ldg_divergence."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GW = np.array([1.0, 1.0])

_CACHE = {}


def _key(*a):
    return tuple(x.tobytes() if isinstance(x, np.ndarray) else x for x in a)



def _q1(xi, et):
    N = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                         (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
    dN = 0.25 * np.array([[-(1 - et), -(1 - xi)], [(1 - et), -(1 + xi)],
                          [(1 + et), (1 + xi)], [-(1 + et), (1 - xi)]])
    return N, dN


def _mirror(xi, et, f):
    return {0: (xi, 1.0), 1: (-1.0, et), 2: (xi, -1.0), 3: (1.0, et)}[f]


def _nb(i, j, f, nx):
    d = [(0, -1), (1, 0), (0, 1), (-1, 0)][f]
    ii, jj = i + d[0], j + d[1]
    return (ii, jj) if 0 <= ii < nx and 0 <= jj < nx else None


def _tables():
    """Face normals, face quadrature points and the reference basis values there."""
    fn = [np.array([0.0, -1.0]), np.array([1.0, 0.0]),
          np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    fp = [[(x, -1.0) for x in GP], [(1.0, x) for x in GP],
          [(x, 1.0) for x in GP], [(-1.0, x) for x in GP]]
    fb = [[_q1(xi, et)[0] for (xi, et) in pts] for pts in fp]
    fbo = [[_q1(*_mirror(xi, et, f))[0] for (xi, et) in fp[f]] for f in range(4)]
    vol = [_q1(xi, et) for xi in GP for et in GP]
    return fn, fp, fb, fbo, vol


_FACE_N, _FACE_PTS, _FB, _FBO, _VOL = _tables()


def _oracle_ldg_divergence(U, S, nx, h, a, b):
    """The weak (div sigma, v) of Eq 46a: -(sigma, grad v)_K plus the traction fluxes."""
    if nx < 1 or h <= 0:
        raise ValueError("need nx >= 1 and h > 0")
    U = np.asarray(U, dtype=np.float64); S = np.asarray(S, dtype=np.float64)
    R = np.zeros((nx, nx, 4, 2)); J = h / 2.0
    for j in range(nx):
        for i in range(nx):
            r = np.zeros((4, 2))
            for (_N, dN) in _VOL:
                r -= (dN / J) @ S[i, j].T * J * J
            for f in range(4):
                n = _FACE_N[f]; nb = _nb(i, j, f, nx)
                for q, w in enumerate(GW):
                    N = _FB[f][q]; uK = N @ U[i, j]
                    if nb is None:
                        fl = _oracle_ldg_numerical_fluxes(uK, None, S[i, j], None, n, a, b, True)
                    else:
                        ii, jj = nb
                        fl = _oracle_ldg_numerical_fluxes(uK, _FBO[f][q] @ U[ii, jj], S[i, j], S[ii, jj], n, a, b, False)
                    r += np.outer(N, fl[1]) * w * J
            R[i, j] = r
    return R

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nnx = 3\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(5)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\n', "call": 'ldg_divergence(U, S, nx, h, a, b)', "gold_call": '_oracle_ldg_divergence(U, S, nx, h, a, b)', "tol": 1e-06},
        {"setup": 'import numpy as np\nnx = 4\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(13)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\n', "call": 'ldg_divergence(U, S, nx, h, a, b)', "gold_call": '_oracle_ldg_divergence(U, S, nx, h, a, b)', "tol": 1e-06},
        {"setup": 'import numpy as np\nnx = 3\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(5)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\na = 0.1\nb = 25.0\n', "call": 'ldg_divergence(U, S, nx, h, a, b)', "gold_call": '_oracle_ldg_divergence(U, S, nx, h, a, b)', "tol": 1e-06},
    ]
