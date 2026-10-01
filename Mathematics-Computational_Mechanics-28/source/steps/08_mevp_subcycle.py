"""
Advance the source's modified elastic-viscous-plastic system by ONE sub-iteration and return the new nodal velocity, the new element stress and the viscous-plastic stress target that the stress relaxes towards, flattened and concatenated in that order. The stress is relaxed towards the target evaluated at the CURRENT velocity, with the relaxation weight set by the first stabilisation parameter. The velocity is then updated from the momentum equation, which carries two separate inertia terms, one referred to the previous sub-iteration and weighted by the second stabilisation parameter and one referred to the previous physical time level, together with the stress divergence evaluated at the NEW stress and the given forcing. The exact weights, and which state the stress target and the stress divergence are evaluated at, are the source's conventions; recover them from the paper. The velocity that enters the traction fluxes of that divergence, including the velocity-jump penalty, is taken at the PREVIOUS sub-iterate, so every velocity update is an element-local mass-matrix solve as the problem statement's direct local solves require. The stress closure is evaluated with the source's viscous regularisation threshold 2e-9 per second and yield-curve eccentricity 2. Raise ValueError on a non-positive stabilisation parameter or time step. Assemble by calling the earlier sub-problem functions.

Relaxing an artificial elastic stress in pseudo-time turns the strongly nonlinear implicit problem into a sequence of explicit local updates, and the iteration converges to the solution of the original problem.

Returns
-------
return (nx*nx*16,) float64: the new velocity, the new stress and the stress target, concatenated
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b):
    """U, S: current velocity and stress; Un: velocity at the previous physical
    time level; P: (nx, nx) ice strength; H: (nx, nx) thickness;
    Fv: (nx, nx, 4, 2) forcing; alpha, beta: stabilisation parameters;
    dt: physical time step. Returns a flat float64 array."""
    return np.zeros(nx * nx * 16)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: mevp_subcycle."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

RHO_ICE, RHO_A, RHO_O = 900.0, 1.3, 1026.0
CA, CO, FC = 1.2e-3, 5.5e-3, 1.46e-4
PSTAR, CCONC, ECC, DMIN = 27.5e3, 20.0, 2.0, 2e-9
LDOM, ALPHA, BETA, DTP, NSUB, AFLX, BFLX = 512e3, 500.0, 500.0, 600.0, 800, 0.25, 3.0
BASE = {1: (3, 1, 8.0), 2: (3, 2, 11.0), 3: (4, 1, 9.5)}
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


def _oracle_mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b):
    """One mEVP sub-iteration, Eqs 9-10.

    Returns a flat float64 array holding, in order, the new velocity
    (nx*nx*4*2 entries), the new stress (nx*nx*2*2) and the VP stress target
    (nx*nx*2*2) that the stress relaxes towards."""
    if alpha <= 0 or beta <= 0 or dt <= 0:
        raise ValueError("need alpha > 0, beta > 0 and dt > 0")
    E = _oracle_ldg_strain(U, S, nx, h, a, b)
    tgt = np.zeros_like(np.asarray(S, dtype=np.float64))
    for j in range(nx):
        for i in range(nx):
            tgt[i, j] = _oracle_vp_stress(E[i, j], P[i, j], DMIN, ECC)
    Snew = (alpha * np.asarray(S, dtype=np.float64) + tgt) / (alpha + 1.0)
    R = _oracle_ldg_divergence(U, Snew, nx, h, a, b)
    M = np.zeros((4, 4)); J = h / 2.0
    for xi in GP:
        for et in GP:
            N, _ = _q1(xi, et)
            M += np.outer(N, N) * J * J
    Minv = np.linalg.inv(M)
    Unew = np.zeros_like(np.asarray(U, dtype=np.float64))
    for j in range(nx):
        for i in range(nx):
            m = RHO_ICE * H[i, j]
            rhs = ((beta * m / dt) * (M @ U[i, j]) + (m / dt) * (M @ Un[i, j])
                   + R[i, j] + M @ Fv[i, j])
            Unew[i, j] = Minv @ rhs / (beta * m / dt + m / dt)
    return np.concatenate([Unew.ravel(), Snew.ravel(), tgt.ravel()])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nnx = 3\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(5)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\nUn = np.zeros_like(U)\nP = np.full((nx, nx), 2.0e4)\nH = np.full((nx, nx), 0.3)\nFv = np.zeros((nx, nx, 4, 2))\nalpha = 500.0\nbeta = 500.0\ndt = 600.0\n', "call": 'mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b)', "gold_call": '_oracle_mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b)', "tol": 1e-06},
        {"setup": 'import numpy as np\nnx = 3\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(5)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\nUn = U * 0.5\nP = np.full((nx, nx), 1.2e4)\nH = np.full((nx, nx), 0.8)\nFv = np.full((nx, nx, 4, 2), 0.02)\nalpha = 300.0\nbeta = 900.0\ndt = 60.0\n', "call": 'mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b)', "gold_call": '_oracle_mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b)', "tol": 1e-06},
        {"setup": 'import numpy as np\nnx = 3\nh = 512e3 / nx\na = 0.25\nb = 3.0\nrng = np.random.default_rng(5)\nU = rng.standard_normal((nx, nx, 4, 2)) * 0.05\nS = np.zeros((nx, nx, 2, 2))\nfor _j in range(nx):\n    for _i in range(nx):\n        _m = rng.standard_normal((2, 2)) * 1e3\n        S[_i, _j] = 0.5 * (_m + _m.T)\nUn = np.zeros_like(U)\nP = np.full((nx, nx), 5.0e3)\nH = np.full((nx, nx), 0.2)\nFv = np.zeros((nx, nx, 4, 2))\nalpha = 500.0\nbeta = 500.0\ndt = 600.0\na = 0.1\nb = 25.0\n', "call": 'mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b)', "gold_call": '_oracle_mevp_subcycle(U, S, Un, P, H, Fv, nx, h, alpha, beta, dt, a, b)', "tol": 1e-06},
    ]
