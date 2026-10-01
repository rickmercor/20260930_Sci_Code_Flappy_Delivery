"""
Recover the element-wise strain rate on a uniform nx by nx quadrilateral mesh of the square domain from the source's mixed formulation. Because the stress and strain spaces are element-constant, the volume term of that equation vanishes and each element's strain rate is determined entirely by the velocity fluxes on its four faces: integrate the symmetric part of the outer product of the velocity flux with the outward normal over the element boundary and divide by the element area. Velocities are bilinear on each element with the four nodal values ordered anticlockwise from the lower-left corner; integrate each face with two-point Gauss-Legendre quadrature. Raise ValueError unless nx and the element size are positive. Assemble by calling the earlier sub-problem functions.

Writing the first-order system this way keeps the strain rate local to each element, which is what allows the constitutive law to be applied element by element without a global solve.

Returns
-------
return (nx, nx, 2, 2) float64: the element-wise strain rate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ldg_strain(U, S, nx, h, a, b):
    """U: (nx, nx, 4, 2) nodal velocities; S: (nx, nx, 2, 2) element stresses;
    nx: elements per side; h: element size; a, b: flux parameters.
    Returns (nx, nx, 2, 2) float64."""
    return np.zeros((nx, nx, 2, 2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: ldg_strain."""

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


def _oracle_ldg_strain(U, S, nx, h, a, b):
    """Eq 41 with element-constant stress: eps_K = (1/|K|) sum_F int_F sym(u_hat (x) n)."""
    if nx < 1 or h <= 0:
        raise ValueError("need nx >= 1 and h > 0")
    U = np.asarray(U, dtype=np.float64); S = np.asarray(S, dtype=np.float64)
    E = np.zeros((nx, nx, 2, 2))
    for j in range(nx):
        for i in range(nx):
            acc = np.zeros((2, 2))
            for f in range(4):
                n = _FACE_N[f]; nb = _nb(i, j, f, nx)
                for q, w in enumerate(GW):
                    uK = _FB[f][q] @ U[i, j]
                    if nb is None:
                        fl = _oracle_ldg_numerical_fluxes(uK, None, S[i, j], None, n, a, b, True)
                    else:
                        ii, jj = nb
                        fl = _oracle_ldg_numerical_fluxes(uK, _FBO[f][q] @ U[ii, jj], S[i, j], S[ii, jj], n, a, b, False)
                    uh = fl[0]
                    acc += 0.5 * (np.outer(uh, n) + np.outer(n, uh)) * w * (h / 2.0)
            E[i, j] = acc / (h * h)
    return E

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nnx = 3\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(5)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\n', "call": 'ldg_strain(U, S, nx, h, a, b)', "gold_call": '_oracle_ldg_strain(U, S, nx, h, a, b)', "tol": 1e-16},
        {"setup": 'import numpy as np\nnx = 2\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(9)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\n', "call": 'ldg_strain(U, S, nx, h, a, b)', "gold_call": '_oracle_ldg_strain(U, S, nx, h, a, b)', "tol": 1e-16},
        {"setup": 'import numpy as np\nnx = 3\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(5)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\na = 0.1\nb = 25.0\n', "call": 'ldg_strain(U, S, nx, h, a, b)', "gold_call": '_oracle_ldg_strain(U, S, nx, h, a, b)', "tol": 1e-16},
    ]
