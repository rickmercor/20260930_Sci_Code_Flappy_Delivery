"""
Evaluate the domain (equivalent-domain-integral) form of the crack-tip J-integral over the local mesh, given nodal displacement, velocity and acceleration fields and the nodal weight function, integrated element by element with the standard 3x3 Gauss rule (the paper's Eq. 17 for static=False, its quasi-static reduction Eq. 20 for static=True). The crack extends in the +x direction, so x1 is the x axis. In the dynamic form the crack-tip energy flux carries the kinetic-energy density inside the Eshelby term AND an explicit inertia term in acceleration and velocity gradient; the quasi-static form keeps only the strain-energy density and drops the inertia term. Stresses follow from the plane-strain elasticity matrix. Recover the exact integrand from the paper; omitting the kinetic and inertia contributions gives the wrong (quasi-static) value.

For a running crack the path-area J-integral must include the kinetic energy and the material acceleration to remain a valid measure of the crack-tip energy flux; the difference from the quasi-static integrand is exactly the dynamic correction the paper is about.

Returns
-------
return float: the domain-form J-integral over the local mesh
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dynamic_j_integral(loc_elems, u, ud, udd, qnod, E, nu, rho, static=False):
    """loc_elems: list of (4, 2) local Q4 node arrays; u, ud, udd: (n_elem, 4, 2) nodal
    displacement, velocity and acceleration; qnod: (n_elem, 4) nodal weight function; E,
    nu, rho: material constants; static: if True use the quasi-static integrand (paper
    Eq. 20), else the dynamic one (paper Eq. 17). Returns float: the J-integral over the
    local domain (x1 is the +x crack direction), 3x3 Gauss per element."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: dynamic J-integral (Eq. 17); quasi-static reduction is Eq. 20."""

import numpy as np


def _psD(E, nu):
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]], dtype=np.float64)


_GPT = np.array([-(3.0 / 5.0) ** 0.5, 0.0, (3.0 / 5.0) ** 0.5], dtype=np.float64)
_GWT = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)


def _q4b(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)], dtype=np.float64)
    dNr = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=np.float64)
    dNs = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=np.float64)
    return N, dNr, dNs


def _oracle_dynamic_j_integral(loc_elems, u, ud, udd, qnod, E, nu, rho, static=False):
    n_el = len(loc_elems)
    if n_el < 1:
        raise ValueError("no local elements supplied")
    if not np.isfinite(E) or E <= 0 or not (-1.0 < float(nu) < 0.5):
        raise ValueError("invalid plane-strain constants")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    for _nm, _a in (("u", u), ("ud", ud), ("udd", udd), ("qnod", qnod)):
        if len(_a) != n_el:
            raise ValueError("field " + _nm + " does not match the element count")
    D = _psD(E, nu)
    J = 0.0
    for e, xy in enumerate(loc_elems):
        xy = np.asarray(xy, dtype=np.float64)
        ue = np.asarray(u[e], dtype=np.float64); ude = np.asarray(ud[e], dtype=np.float64)
        udde = np.asarray(udd[e], dtype=np.float64); qe = np.asarray(qnod[e], dtype=np.float64)
        for a in range(3):
            for b in range(3):
                xi, eta, w = _GPT[a], _GPT[b], _GWT[a] * _GWT[b]
                N, dNr, dNs = _q4b(xi, eta)
                J00 = dNr @ xy[:, 0]; J01 = dNr @ xy[:, 1]
                J10 = dNs @ xy[:, 0]; J11 = dNs @ xy[:, 1]
                detJ = J00 * J11 - J01 * J10
                dNx = (J11 * dNr - J01 * dNs) / detJ
                dNy = (-J10 * dNr + J00 * dNs) / detJ
                du = np.zeros((2, 2))
                for k in range(4):
                    du[0, 0] += dNx[k] * ue[k, 0]; du[0, 1] += dNy[k] * ue[k, 0]
                    du[1, 0] += dNx[k] * ue[k, 1]; du[1, 1] += dNy[k] * ue[k, 1]
                eps = np.array([du[0, 0], du[1, 1], du[0, 1] + du[1, 0]])
                sig = D @ eps
                sigt = np.array([[sig[0], sig[2]], [sig[2], sig[1]]])
                W = 0.5 * (sig @ eps)
                vel = N @ ude
                acc = N @ udde
                Kd = 0.5 * rho * (vel[0] ** 2 + vel[1] ** 2)
                dqdx = dNx @ qe; dqdy = dNy @ qe
                qv = N @ qe
                EW = W + (0.0 if static else Kd)
                term1 = 0.0
                for i in range(2):
                    s = sigt[i, 0] * du[0, 0] + sigt[i, 1] * du[1, 0]
                    s -= EW * (1.0 if i == 0 else 0.0)
                    term1 += s * (dqdx if i == 0 else dqdy)
                integ = term1
                if not static:
                    dvx = dNx @ ude[:, 0]; dvy = dNx @ ude[:, 1]
                    integ += rho * (acc[0] * du[0, 0] + acc[1] * du[1, 0]
                                    - (vel[0] * dvx + vel[1] * dvy)) * qv
                J += integ * detJ * w
    return float(J)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\n_xs=_n.array([-1.5,-0.5,0.5,1.5])\nloc=[]\nfor _j in range(3):\n    for _i in range(3):\n        loc.append(_n.array([[_xs[_i],_xs[_j]],[_xs[_i+1],_xs[_j]],[_xs[_i+1],_xs[_j+1]],[_xs[_i],_xs[_j+1]]]))\nVS=1.0\nu=_n.zeros((9,4,2)); ud=_n.zeros((9,4,2)); udd=_n.zeros((9,4,2)); qnod=_n.zeros((9,4))\nfor _e,_xy in enumerate(loc):\n    for _k in range(4):\n        _x,_y=_xy[_k]\n        u[_e,_k]=[0.30*_x+0.12*_x*_y+0.06*_y*_y, 0.18*_y+0.15*_x*_y+0.09*_x*_x]\n        _vx=0.20*_y-0.10*_x; _vy=0.16*_x+0.05*_y\n        ud[_e,_k]=[VS*_vx, VS*_vy]; udd[_e,_k]=[-0.6*VS*_vx,-0.6*VS*_vy]\n    _r=_n.sqrt(_xy[:,0]**2+_xy[:,1]**2); qnod[_e]=(_r<=1.5).astype(float)\nE=1.0;nu=0.3;rho=1.0', "call": 'dynamic_j_integral(loc, u, ud, udd, qnod, E, nu, rho, False)', "gold_call": '_oracle_dynamic_j_integral(loc, u, ud, udd, qnod, E, nu, rho, False)', "tol": 1e-08},
        {"setup": 'import numpy as _n\n_xs=_n.array([-1.5,-0.5,0.5,1.5])\nloc=[]\nfor _j in range(3):\n    for _i in range(3):\n        loc.append(_n.array([[_xs[_i],_xs[_j]],[_xs[_i+1],_xs[_j]],[_xs[_i+1],_xs[_j+1]],[_xs[_i],_xs[_j+1]]]))\nVS=1.2\nu=_n.zeros((9,4,2)); ud=_n.zeros((9,4,2)); udd=_n.zeros((9,4,2)); qnod=_n.zeros((9,4))\nfor _e,_xy in enumerate(loc):\n    for _k in range(4):\n        _x,_y=_xy[_k]\n        u[_e,_k]=[0.30*_x+0.12*_x*_y+0.06*_y*_y, 0.18*_y+0.15*_x*_y+0.09*_x*_x]\n        _vx=0.20*_y-0.10*_x; _vy=0.16*_x+0.05*_y\n        ud[_e,_k]=[VS*_vx, VS*_vy]; udd[_e,_k]=[-0.6*VS*_vx,-0.6*VS*_vy]\n    _r=_n.sqrt(_xy[:,0]**2+_xy[:,1]**2); qnod[_e]=(_r<=1.5).astype(float)\nE=1.5;nu=0.35;rho=0.8', "call": 'dynamic_j_integral(loc, u, ud, udd, qnod, E, nu, rho, False)', "gold_call": '_oracle_dynamic_j_integral(loc, u, ud, udd, qnod, E, nu, rho, False)', "tol": 1e-08},
        {"setup": 'import numpy as _n\n_xs=_n.array([-1.5,-0.5,0.5,1.5])\nloc=[]\nfor _j in range(3):\n    for _i in range(3):\n        loc.append(_n.array([[_xs[_i],_xs[_j]],[_xs[_i+1],_xs[_j]],[_xs[_i+1],_xs[_j+1]],[_xs[_i],_xs[_j+1]]]))\nVS=1.0\nu=_n.zeros((9,4,2)); ud=_n.zeros((9,4,2)); udd=_n.zeros((9,4,2)); qnod=_n.zeros((9,4))\nfor _e,_xy in enumerate(loc):\n    for _k in range(4):\n        _x,_y=_xy[_k]\n        u[_e,_k]=[0.30*_x+0.12*_x*_y+0.06*_y*_y, 0.18*_y+0.15*_x*_y+0.09*_x*_x]\n        _vx=0.20*_y-0.10*_x; _vy=0.16*_x+0.05*_y\n        ud[_e,_k]=[VS*_vx, VS*_vy]; udd[_e,_k]=[-0.6*VS*_vx,-0.6*VS*_vy]\n    _r=_n.sqrt(_xy[:,0]**2+_xy[:,1]**2); qnod[_e]=(_r<=1.5).astype(float)\nE=1.0;nu=0.3;rho=1.0', "call": 'dynamic_j_integral(loc, u, ud, udd, qnod, E, nu, rho, True)', "gold_call": '_oracle_dynamic_j_integral(loc, u, ud, udd, qnod, E, nu, rho, True)', "tol": 1e-08},
    ]
