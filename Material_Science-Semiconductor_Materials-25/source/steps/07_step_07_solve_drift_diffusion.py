"""
Solve the stationary drift-diffusion problem for the symmetric silicon PN junction on the distorted mesh with Newton's method and voltage ramping.

The device is the unit square in micrometres with net doping +N0 for y < 0.5, -N0 for y > 0.5 and 0 on the junction row. The bottom contact is grounded and the top contact is at the applied voltage Va; both are Ohmic, so their boundary-edge and vertex nodes take the charge-neutral equilibrium values, with Va added to the potential on the top contact. The sides are insulating. Because the variables span fourteen orders of magnitude, the equations are scaled: the potential in thermal voltages, densities in units of N* = 1e16 cm^-3 and lengths in micrometres, which makes the Poisson coefficient lam2 = eps V_T / (q N* (1e-4 cm)^2), while the diffusion coefficients D = V_T mu stay in cm^2/s because each continuity equation is homogeneous. Newton's method starts from the charge-neutral state at 0 V. Stopping on a small absolute residual is not enough, since minority densities are 1e-13 to 1e-4 in scaled units and a loose test leaves them, and therefore the current, inaccurate; the update itself must converge, to 1e-10 in the potential and 1e-6 relative in the densities. Negative densities occasionally produced by an early Newton step are reset to 1e-20 cm^-3, and the contact values are re-imposed exactly after every update, because round-off in the sparse solve otherwise lets the tiny minority densities on the contacts drift. A forward bias cannot be reached in one Newton solve, so the voltage is ramped from the 0 V solution, doubling the step after each converged solve and halving it after a failure.

Returns
-------
np.ndarray of shape (3, Nn), float: psi (V), n (cm^-3), p (cm^-3) on all DDFV nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_drift_diffusion(N0: float, Va: float, M: int) -> "np.ndarray":
    """Stationary DDFV-HA solution of the symmetric abrupt silicon PN junction.

    Parameters
    ----------
    N0 : float
        Doping magnitude in cm^-3: N = +N0 for y < 0.5, -N0 for y > 0.5, 0 at
        y = 0.5 (y in micrometres, evaluated at each node's position:
        barycentre, boundary-edge midpoint or vertex).
    Va : float
        Voltage on the top contact (y = 1) in volts; the bottom contact (y = 0)
        is at 0 V. Va >= 0.
    M : int
        Mesh parameter for build_mesh.

    Returns
    -------
    state : "np.ndarray"
        Shape (3, Nn): psi (V), n (cm^-3), p (cm^-3) on all nodes in the node
        order of ddfv_geometry (triangles, boundary edges, vertices).

    Notes
    -----
    Silicon at 300 K: q = 1.602192e-19 C, eps = 1.035941e-12 C/(V cm),
    V_T = 0.025852 V, mu_n = 1417.0 and mu_p = 470.5 cm^2/(V s), D = V_T mu.
    Scaling: u = psi / V_T, n and p divided by 1e16 cm^-3, lengths in
    micrometres, lam2 = eps V_T / (q 1e16 (1e-4)^2).
    Dirichlet nodes: boundary edges with both ends on y = 0 or both on y = 1,
    and vertices with y = 0 or y = 1, set by contact_densities (plus Va on top).
    Newton: start from contact_densities of the local doping at 0 V; after each
    update clip scaled n, p below at 1e-36 and reset the Dirichlet nodes to
    their exact contact values; stop when, over the non-Dirichlet nodes,
    max|du| < 1e-10 and the largest relative update of n and p is < 1e-6
    (at most 60 iterations).
    Ramp: solve at 0 V, then step the voltage starting at 0.05 V, doubling the
    step after each success and halving it after a failure, until Va.
    Include every import your implementation needs inside the function body.
    """
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse.linalg as spla


def _newton_solve(u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                  ub: "np.ndarray", nb: "np.ndarray", pb: "np.ndarray", args: tuple):
    """Newton iteration at one applied voltage; returns (u, n, p) or None if it fails."""
    geom, Nd, dirichlet, free, lam2, Dn, Dp = args
    u, n, p = u.copy(), n.copy(), p.copy()
    u[dirichlet], n[dirichlet], p[dirichlet] = ub[dirichlet], nb[dirichlet], pb[dirichlet]
    for _ in range(60):
        R, J = _oracle_assemble_system(geom, u, n, p, Nd, dirichlet, lam2, Dn, Dp)
        d = spla.spsolve(J.tocsc(), -R)
        if not np.all(np.isfinite(d)):
            return None
        du, dn, dp = d[0::3], d[1::3], d[2::3]
        u = u + du
        n = np.maximum(n + dn, 1e-36)
        p = np.maximum(p + dp, 1e-36)
        # re-impose contact values exactly: roundoff in the linear solve must not move them
        u[dirichlet], n[dirichlet], p[dirichlet] = ub[dirichlet], nb[dirichlet], pb[dirichlet]
        if np.max(np.abs(du[free])) < 1e-10 and max(np.max(np.abs(dn[free]) / n[free]),
                                                     np.max(np.abs(dp[free]) / p[free])) < 1e-6:
            return u, n, p
    return None


def _oracle_solve_drift_diffusion(N0: float, Va: float, M: int) -> "np.ndarray":
    q, eps, VT = 1.602192e-19, 1.035941e-12, 0.025852
    Nstar = 1e16
    lam2 = eps * VT / (q * Nstar * 1e-8)
    Dn, Dp = VT * 1417.0, VT * 470.5
    pts, tri = _oracle_build_mesh(M)
    geom = _oracle_ddfv_geometry(pts, tri)
    nT = len(tri)
    xy = np.vstack([geom["tri_centroid"], geom["bnd_mid"], pts])
    y = xy[:, 1]
    Nphys = np.where(np.abs(y - 0.5) < 1e-12, 0.0, np.where(y < 0.5, N0, -N0))
    ey = pts[geom["bnd_edges"]][:, :, 1]
    on_e = np.all(np.abs(ey) < 1e-12, axis=1) | np.all(np.abs(ey - 1.0) < 1e-12, axis=1)
    on_v = (np.abs(pts[:, 1]) < 1e-12) | (np.abs(pts[:, 1] - 1.0) < 1e-12)
    dirichlet = np.concatenate([np.zeros(nT, dtype=bool), on_e, on_v])
    top = dirichlet & (y > 0.5)
    eq = _oracle_contact_densities(Nphys, 0.0)
    Nd = Nphys / Nstar

    free = ~dirichlet
    nb, pb = eq[1] / Nstar, eq[2] / Nstar
    args = (geom, Nd, dirichlet, free, lam2, Dn, Dp)

    sol = _newton_solve(eq[0] / VT, nb, pb, eq[0] / VT, nb, pb, args)
    if sol is None:
        raise RuntimeError("Newton failed at 0 V")
    V, dV = 0.0, 0.05
    while V < Va - 1e-15:
        Vn = min(V + dV, Va)
        trial = _newton_solve(*sol, eq[0] / VT + (Vn / VT) * top, nb, pb, args)
        if trial is None:
            dV *= 0.5
            if dV < 1e-6:
                raise RuntimeError("voltage ramp failed")
            continue
        sol, V, dV = trial, Vn, 2.0 * dV
    u, n, p = sol
    return np.stack([u * VT, n * Nstar, p * Nstar])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    view = "(lambda s: np.vstack([s[0, -81:], np.log10(s[1, -81:]), np.log10(s[2, -81:])]))"
    full = "(lambda s: np.vstack([s[0], np.log10(s[1]), np.log10(s[2])]))"
    return [
        # Normal: equilibrium junction at the highest doping on the 8x8 test mesh (vertex values).
        {"setup": "import numpy as np\n",
         "call": view + "(solve_drift_diffusion(1e17, 0.0, 8))",
         "gold_call": view + "(_oracle_solve_drift_diffusion(1e17, 0.0, 8))"},
        # Normal: forward bias 0.5 V at the lowest doping, reached by voltage ramping.
        {"setup": "import numpy as np\n",
         "call": view + "(solve_drift_diffusion(1e16, 0.5, 8))",
         "gold_call": view + "(_oracle_solve_drift_diffusion(1e16, 0.5, 8))"},
        # Boundary: coarsest useful mesh, all primal and dual nodes compared.
        {"setup": "import numpy as np\n",
         "call": full + "(solve_drift_diffusion(3e16, 0.3, 4))",
         "gold_call": full + "(_oracle_solve_drift_diffusion(3e16, 0.3, 4))"},
    ]
