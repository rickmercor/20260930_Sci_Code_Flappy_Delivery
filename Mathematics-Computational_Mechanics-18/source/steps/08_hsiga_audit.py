"""
Run the full hybrid s-version isogeometric dynamic-fracture audit for a crack running at the given speed, on the declared fixed configuration (a quadratic global B-spline patch on [-1.5, 1.5]^2 with an overlaid 3x3 local Q4 mesh, the declared displacement and velocity fields with the velocity scaled by the crack speed, and E = 1.0, nu = 0.3, rho = 1.0). Return the audit vector [A_I, V1, V2, J_dynamic, J_static, K_dynamic, K_static, coupling_norm, q_count, W_ref, K_ref, partition] where: A_I, V1, V2 come from the wave factor; J_dynamic and J_static are the magnitudes of the dynamic and quasi-static J-integrals over the local mesh; K_dynamic and K_static are the corresponding mode-I stress intensity factors; coupling_norm is the Frobenius norm of the global-local coupling stiffness; q_count is the total number of local element-nodes inside the J-domain; W_ref and K_ref are the strain- and kinetic-energy densities at the declared reference state; partition is the sum of the global basis values at parameter 0.3. Assemble by calling the earlier sub-problem functions. The stress intensity factors must use the dynamic conversion for K_dynamic and the quasi-static conversion for K_static.

The audit contrasts, for one crack speed, the dynamic and quasi-static crack-tip measures produced by the same displacement state, and reports the coupling and weight-function bookkeeping that the s-version isogeometric assembly depends on.

Returns
-------
return (12,) float64: the hybrid s-version IGA dynamic-fracture audit vector
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hsiga_audit(crack_velocity):
    """crack_velocity: positive crack-tip speed below the shear wave speed. Builds the
    declared fixed configuration (velocity field scaled by crack_velocity) and returns a
    float64 array (12,): [A_I, V1, V2, J_dynamic, J_static, K_dynamic, K_static,
    coupling_norm, q_count, W_ref, K_ref, partition]. Assembled by calling the earlier
    sub-problem functions. Raises ValueError on a non-positive or nonfinite crack_velocity."""
    return np.zeros(12)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 (final orchestrator): hybrid s-version IGA dynamic-fracture audit."""

import numpy as np

_E, _NU, _RHO, _P, _L, _HL = 1.0, 0.3, 1.0, 2, 1.5, 1.0
_V_REF = 0.4


def _mesh():
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    loc = []
    for j in range(3):
        for i in range(3):
            loc.append(np.array([[xs[i], xs[j]], [xs[i + 1], xs[j]],
                                 [xs[i + 1], xs[j + 1]], [xs[i], xs[j + 1]]], dtype=np.float64))
    return loc


def _disp(x, y):
    return np.array([0.30 * x + 0.12 * x * y + 0.06 * y * y,
                     0.18 * y + 0.15 * x * y + 0.09 * x * x])


def _vfield(x, y):
    return np.array([0.20 * y - 0.10 * x, 0.16 * x + 0.05 * y])


def _oracle_hsiga_audit(crack_velocity):
    if isinstance(crack_velocity, bool) or not np.isfinite(crack_velocity) or crack_velocity <= 0:
        raise ValueError("crack_velocity must be positive and finite")
    V = float(crack_velocity)
    vs = V / _V_REF
    loc = _mesh()
    ne = len(loc)
    u = np.zeros((ne, 4, 2)); ud = np.zeros((ne, 4, 2)); udd = np.zeros((ne, 4, 2)); qn = np.zeros((ne, 4))
    for e, xy in enumerate(loc):
        for k in range(4):
            x, y = xy[k]
            u[e, k] = _disp(x, y)
            vv = _vfield(x, y)
            ud[e, k] = vs * vv
            udd[e, k] = -0.6 * vs * vv
        qn[e] = _oracle_q_weight(xy, _HL)
    gknx = np.array([-_L, -_L, -_L, 0.0, _L, _L, _L])
    gkny = gknx.copy()
    wf = _oracle_wave_factor(_E, _NU, _RHO, V)
    A_I, V1, V2 = float(wf[0]), float(wf[1]), float(wf[2])
    Jd = abs(_oracle_dynamic_j_integral(loc, u, ud, udd, qn, _E, _NU, _RHO, False))
    Js = abs(_oracle_dynamic_j_integral(loc, u, ud, udd, qn, _E, _NU, _RHO, True))
    Kd = _oracle_dsif(Jd, A_I, _E, _NU, True)
    Ks = _oracle_dsif(Js, A_I, _E, _NU, False)
    Kgl = _oracle_coupling_matrix(gknx, gkny, _P, loc, _E, _NU)
    coupfro = float(np.linalg.norm(Kgl))
    qcount = float(qn.sum())
    eps0 = np.array([0.36, 0.255, 0.285])
    c0 = _E / ((1.0 + _NU) * (1.0 - 2.0 * _NU))
    D0 = c0 * np.array([[1.0 - _NU, _NU, 0.0], [_NU, 1.0 - _NU, 0.0], [0.0, 0.0, (1.0 - 2.0 * _NU) / 2.0]])
    sig0 = D0 @ eps0
    vel0 = vs * _vfield(0.5, 0.5)
    en = _oracle_energy_densities(sig0, eps0, _RHO, vel0)
    Nb = _oracle_bspline_basis(gknx, _P, 0.3)
    Npart = float(Nb[0].sum())
    return np.array([A_I, V1, V2, Jd, Js, Kd, Ks, coupfro, qcount, float(en[0]), float(en[1]), Npart],
                    dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'crack_velocity = 0.4', "call": 'hsiga_audit(crack_velocity)', "gold_call": '_oracle_hsiga_audit(crack_velocity)', "tol": 1e-08},
        {"setup": 'crack_velocity = 0.3', "call": 'hsiga_audit(crack_velocity)', "gold_call": '_oracle_hsiga_audit(crack_velocity)', "tol": 1e-08},
        {"setup": 'crack_velocity = 0.5', "call": 'hsiga_audit(crack_velocity)', "gold_call": '_oracle_hsiga_audit(crack_velocity)', "tol": 1e-08},
    ]
