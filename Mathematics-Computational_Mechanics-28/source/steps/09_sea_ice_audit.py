"""
Run the source's scheme on the three declared configurations and return one row each. A configuration is (nx, field index v, wind speed): (3, 1, 8.0), (3, 2, 11.0) and (4, 1, 9.5). The domain is the square of side 512 km. Element (i, j) carries thickness 0.3 + 0.05 sin((3+v) pi xc) cos((2+v) pi yc) and concentration 0.90 + 0.05 sin((2+v) pi (xc+yc)) with xc, yc the element-centre coordinates divided by the domain side. The forcing at each velocity node is the source's external force: the concentration-weighted sum of an air drag and an ocean drag, each quadratic in the velocity of the driving fluid RELATIVE to the ice, plus a Coriolis term proportional to the ice density, the local thickness and the Coriolis parameter that acts on the ocean velocity minus the ice velocity; use each element's thickness and concentration at all four of its nodes, with air velocity wind*[sin(pi x) cos(pi y) + 0.8 (x - 0.5), -cos(pi x) sin(pi y) + 0.8 (y - 0.5)] and ocean velocity 0.01*[2y - 1, -2x + 1] at normalised coordinates and the densities and drag coefficients of the source's parameter table. Start from rest with zero stress and run exactly 800 sub-iterations at alpha = beta = 500, physical time step 600 s, and flux parameters a = 0.25 and b = 3.0, with the wind multiplied by wind_scale. Each row is [largest shear deformation, mean shear deformation, the largest absolute value taken by any scalar velocity component over all nodes, the ratio of the final stress residual to the immediately preceding one, summed shear deformation], where the shear deformation is the source's own diagnostic, the three deformation entries are expressed in units of 1e-6 per second, and the residual is the largest absolute difference between the relaxed stress and its target. Raise ValueError unless wind_scale is positive and finite. Assemble by calling the earlier sub-problem functions, every one of them.

The shear deformation is the field in which the linear kinematic features of sea ice appear, so it is the natural diagnostic for judging whether a discretisation resolves them.

Returns
-------
return (3, 5) float64: one audit row per declared configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sea_ice_audit(wind_scale):
    """wind_scale: positive multiplier on the declared wind speed.
    Returns (3, 5) float64 with one row per configuration."""
    return np.zeros((3, 5))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: sea_ice_audit."""

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


def _fields(nx, v):
    h = LDOM / nx
    H = np.zeros((nx, nx)); A = np.zeros((nx, nx))
    for j in range(nx):
        for i in range(nx):
            xc, yc = (i + 0.5) / nx, (j + 0.5) / nx
            H[i, j] = 0.3 + 0.05 * np.sin((3 + v) * np.pi * xc) * np.cos((2 + v) * np.pi * yc)
            A[i, j] = 0.90 + 0.05 * np.sin((2 + v) * np.pi * (xc + yc))
    return H, A


def _forcing(U, A, H, nx, wind):
    """Eqs 2-3: F = A (tau_a + tau_o) + rho_ice H f k x (u_o - u), both drags quadratic
    in the velocity RELATIVE to the ice and both weighted by the concentration; the
    Coriolis term acts on the ocean-minus-ice velocity, which carries the -k x u sign."""
    F = np.zeros((nx, nx, 4, 2))
    for j in range(nx):
        for i in range(nx):
            for c, (dx, dy) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
                x, y = (i + dx) / nx, (j + dy) / nx
                ua = wind * np.array([np.sin(np.pi * x) * np.cos(np.pi * y) + 0.8 * (x - 0.5),
                                      -np.cos(np.pi * x) * np.sin(np.pi * y) + 0.8 * (y - 0.5)])
                uo = 0.01 * np.array([2.0 * y - 1.0, -2.0 * x + 1.0])
                u = U[i, j, c]
                rel_a = ua - u; rel_o = uo - u
                drag = (RHO_A * CA * np.linalg.norm(rel_a) * rel_a
                        + RHO_O * CO * np.linalg.norm(rel_o) * rel_o)
                F[i, j, c] = (A[i, j] * drag
                              + RHO_ICE * H[i, j] * FC * np.array([-rel_o[1], rel_o[0]]))
    return F


def _shear_deformation(E):
    E = np.asarray(E, dtype=np.float64)
    return np.sqrt((E[..., 0, 0] - E[..., 1, 1]) ** 2 + 4.0 * E[..., 0, 1] ** 2)


_FACE_N, _FACE_PTS, _FB, _FBO, _VOL = _tables()


def _oracle_sea_ice_audit(wind_scale):
    if isinstance(wind_scale, bool) or not np.isfinite(wind_scale) or wind_scale <= 0:
        raise ValueError("wind_scale must be positive and finite")
    rows = []
    for v in (1, 2, 3):
        nx, fv, wind = BASE[v]
        h = LDOM / nx
        H, A = _fields(nx, fv)
        P = _oracle_ice_strength(H, A, PSTAR, CCONC)
        U = np.zeros((nx, nx, 4, 2)); Sg = np.zeros((nx, nx, 2, 2)); Un = U.copy()
        r0 = rlast = rprev = 0.0
        for k in range(NSUB):
            Fv = _forcing(U, A, H, nx, wind * wind_scale)
            out = _oracle_mevp_subcycle(U, Sg, Un, P, H, Fv, nx, h, ALPHA, BETA, DTP, AFLX, BFLX)
            nu = nx * nx * 8; ns = nx * nx * 4
            U = out[:nu].reshape(nx, nx, 4, 2)
            Sg = out[nu:nu + ns].reshape(nx, nx, 2, 2)
            tgt = out[nu + ns:].reshape(nx, nx, 2, 2)
            r = float(np.max(np.abs(Sg - tgt)))
            if k == 0:
                r0 = r
            rprev, rlast = rlast, r
        E = _oracle_ldg_strain(U, Sg, nx, h, AFLX, BFLX)
        # diagnostics at the converged state, each an invariant the source implies
        D = _oracle_effective_deformation(E, DMIN, ECC)
        vis = _oracle_vp_viscosities(P, D, ECC)
        sig_vp = _oracle_vp_stress(E, P, DMIN, ECC)
        Rmom = _oracle_ldg_divergence(U, Sg, nx, h, AFLX, BFLX)
        fl_b = _oracle_ldg_numerical_fluxes(U[0, 0, 0], None, Sg[0, 0], None,
                                    np.array([-1.0, 0.0]), AFLX, BFLX, True)
        if not (np.all(D >= DMIN) and np.all(vis[0] >= vis[1])
                and np.all(np.isfinite(sig_vp)) and np.all(np.isfinite(Rmom))
                and np.all(np.isfinite(fl_b)) and np.allclose(fl_b[0], 0.0)):
            raise ValueError("the converged state violates the source's invariants")
        eII = _shear_deformation(E)
        rows.append([float(np.max(eII) * 1e6), float(np.mean(eII) * 1e6),
                     float(np.max(np.abs(U))), float(rlast / rprev),
                     float(np.sum(eII) * 1e6)])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": '', "call": 'sea_ice_audit(1.0)', "gold_call": '_oracle_sea_ice_audit(1.0)', "tol": 1e-06},
        {"setup": '', "call": 'sea_ice_audit(0.6)', "gold_call": '_oracle_sea_ice_audit(0.6)', "tol": 1e-06},
        {"setup": '', "call": 'sea_ice_audit(1.4)', "gold_call": '_oracle_sea_ice_audit(1.4)', "tol": 1e-06},
    ]
