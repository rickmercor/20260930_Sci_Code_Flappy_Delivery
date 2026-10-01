"""
Loop the elements and their quadrature points, evaluate the midpoint deformation gradient from the averaged displacement field, obtain the stress from the constitutive step, and scatter the internal force contributions into a global vector; then subtract the external contribution of the dead traction on the right edge. The residual is returned for every degree of freedom, before any Dirichlet reduction, so that the caller decides which rows to keep. Inertia and body forces are absent, so what remains is the quasi-static balance.

The weak form of the momentum balance in the reference configuration pairs the first Piola-Kirchhoff stress with the gradient of the virtual displacement. Discretising with nodal shape functions turns that pairing into a sum over quadrature points, and the nodal residual is the difference between what the material carries and what the boundary applies.

Plane strain enters through the kinematics rather than the constitution. The in-plane deformation gradient is embedded in a three-dimensional tensor with a unit out-of-plane entry, so the three-dimensional energy, stress and viscosity relations apply verbatim and the out-of-plane stress simply never meets a degree of freedom.

The midpoint convention appears again here, and it is not the same rule as the one used for the internal variable. The kinematic quantity is built by averaging the configuration first and only then forming the right Cauchy-Green tensor from the averaged deformation gradient. Averaging the two endpoint tensors instead gives a different, also plausible, scheme. Separately, the traction is a dead load: it is fixed in direction and magnitude and integrated over the reference edge, not over the deformed one. At the deformations reached here, roughly a quarter of the span, that distinction is not a rounding effect.

Returns
-------
np.ndarray of shape (2*(n+1)**2,), the residual for every displacement degree of freedom. In the undeformed, unloaded state it vanishes identically. Under load the residual components sum to minus the applied resultant, which for load factor lam is lam times 160 times (-750, 1000).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_momentum_residual(n, u, u_n, ci, ci_n, lam):
    """Global midpoint momentum residual, internal minus external.

    Plane strain: the 2x2 in-plane deformation gradient is embedded in 3x3
    with a unit out-of-plane entry.  All constitutive quantities are taken
    at the midpoint configuration.  Quadrature is 2x2 Gauss per element and
    unit thickness.  The dead traction lam * (-750, 1000) acts on the whole
    right edge X1 = 480, integrated over the reference edge.  Inertia and
    body forces are absent.

    Args:
        n (int): elements per side.
        u, u_n: (2*(n+1)**2,) nodal displacements at t_{n+1} and t_n.
        ci, ci_n: (24*n**2,) packed quadrature-point internal variables.
        lam (float): current load factor.

    Expected return:
        np.ndarray of shape (2*(n+1)**2,), the residual for every degree of
        freedom, before any Dirichlet reduction.
    """
    return np.zeros(2 * (n + 1) ** 2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_momentum_residual(n, u, u_n, ci, ci_n, lam):
    LAM, MU, LAMV, MUV = 30000.0, 7500.0, 30000.0, 7500.0
    V_DEV, V_VOL = 10000.0, 50000.0
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    def _vec_to_sym(v):
        A = np.zeros((3, 3))
        for k, (i, j) in enumerate(SYM):
            A[i, j] = v[k]
            A[j, i] = v[k]
        return A

    def _sym_to_vec(A):
        return np.array([A[i, j] for (i, j) in SYM])
    def _d_psi(A, m, l):
        Jd = np.sqrt(np.linalg.det(A))
        AinvT = np.linalg.inv(A).T
        return 0.5 * (m * (np.eye(3) - AinvT)
                      + l * (np.log(Jd) + Jd * (Jd - 1.0)) * AinvT)
    def _constitutive(C, Ci):
        S_eq = 2.0 * _d_psi(C, MU, LAM)
        Ci_inv = np.linalg.inv(Ci)
        S_neq = 2.0 * _d_psi(C @ Ci_inv, MUV, LAMV) @ Ci_inv
        return S_eq + S_neq, S_neq @ C
    def _mesh(n):
        out = np.zeros(2 * (n + 1) ** 2)
        for j in range(n + 1):
            for i in range(n + 1):
                xi, eta = i / n, j / n
                nd = j * (n + 1) + i
                out[2 * nd] = 480.0 * xi
                out[2 * nd + 1] = 440.0 * xi + eta * (440.0 - 280.0 * xi)
        return out

    def _conn(n):
        c = []
        for j in range(n):
            for i in range(n):
                n0 = j * (n + 1) + i
                c.append([n0, n0 + 1, n0 + n + 2, n0 + n + 1])
        return np.array(c)
    def _grads(xe, xi, eta):
        dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)],
                              [(1 - eta), -(1 + xi)],
                              [(1 + eta), (1 + xi)],
                              [-(1 + eta), (1 - xi)]])
        J = np.asarray(xe).T @ dN
        return float(np.linalg.det(J)), dN @ np.linalg.inv(J)

    T_BAR = np.array([-750.0, 1000.0])
    GP1D = np.array([-1.0, 1.0]) / np.sqrt(3.0)

    coords = _mesh(n).reshape(-1, 2)
    conn = _conn(n)
    nn = coords.shape[0]
    u = np.asarray(u, dtype=float).reshape(nn, 2)
    u_n = np.asarray(u_n, dtype=float).reshape(nn, 2)
    ci = np.asarray(ci, dtype=float)
    ci_n = np.asarray(ci_n, dtype=float)
    u_mid = 0.5 * (u + u_n)

    R = np.zeros(2 * nn)
    g = 0
    for e in range(conn.shape[0]):
        nodes = conn[e]
        xe = coords[nodes]
        for a in GP1D:
            for b in GP1D:
                detJ, dNdX = _grads(xe, a, b)
                F = np.eye(3)
                F[:2, :2] = np.eye(2) + u_mid[nodes].T @ dNdX
                C_mid = F.T @ F
                Ci_mid = _vec_to_sym(0.5 * (ci[6 * g:6 * g + 6]
                                            + ci_n[6 * g:6 * g + 6]))
                S, _ = _constitutive(C_mid, Ci_mid)
                P = F @ S
                for k, nd in enumerate(nodes):
                    R[2 * nd:2 * nd + 2] += P[:2, :2] @ dNdX[k] * detJ
                g += 1

    t = lam * T_BAR
    for j in range(n):
        na = (j + 1) * (n + 1) - 1
        nb = (j + 2) * (n + 1) - 1
        L = float(np.linalg.norm(coords[nb] - coords[na]))
        for gp in GP1D:
            Nv = np.array([0.5 * (1 - gp), 0.5 * (1 + gp)])
            for k, nd in enumerate((na, nb)):
                R[2 * nd:2 * nd + 2] -= Nv[k] * t * (L / 2.0)
    return R

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": (
                "n = 2\n"
                "nn = (n + 1) ** 2\n"
                "z = np.zeros(2 * nn)\n"
                "e = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 4 * n * n)"
            ),
            "call": "assemble_momentum_residual(n, z, z, e, e, 0.0)",
            "gold_call": "np.zeros(2 * (2 + 1) ** 2)",
        },
        {
            "setup": (
                "n = 2\n"
                "nn = (n + 1) ** 2\n"
                "z = np.zeros(2 * nn)\n"
                "e = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 4 * n * n)\n"
                "R = assemble_momentum_residual(n, z, z, e, e, 1.0).reshape(nn, 2)"
            ),
            "call": "R.sum(axis=0)",
            "gold_call": "-160.0 * np.array([-750.0, 1000.0])",
        },
        {
            "setup": (
                "n = 2\n"
                "nn = (n + 1) ** 2\n"
                "z = np.zeros(2 * nn)\n"
                "u = np.zeros(2 * nn)\n"
                "u[2 * (nn - 1)] = 5.0\n"
                "e = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 4 * n * n)\n"
                "R = assemble_momentum_residual(n, u, z, e, e, 0.0)"
            ),
            "call": "float(np.linalg.norm(R) > 0.0)",
            "gold_call": "1.0",
        },
        {
            "setup": (
                "n = 1\n"
                "nn = (n + 1) ** 2\n"
                "z = np.zeros(2 * nn)\n"
                "e = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 4 * n * n)"
            ),
            "call": "int(assemble_momentum_residual(n, z, z, e, e, 0.5).shape[0])",
            "gold_call": "8",
        },
        {
            "setup": (
                "n = 2\n"
                "nn = (n + 1) ** 2\n"
                "z = np.zeros(2 * nn)\n"
                "short = np.ones(6)\n"
                "def run_model():\n"
                "    try:\n"
                "        assemble_momentum_residual(n, z, z, short, short, 0.0)\n"
                "        return 0\n"
                "    except Exception:\n"
                "        return 1\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_assemble_momentum_residual(n, z, z, short, short, 0.0)\n"
                "        return 0\n"
                "    except Exception:\n"
                "        return 1"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
