"""
Compute the terminal current density through the top Ohmic contact from the primal DDFV-HA electron and hole fluxes.

The DDFV-HA fluxes are antisymmetric, so the scheme conserves electrons and holes cell by cell on the primal mesh, and with no recombination the total current is the same through every horizontal cut of the device, including each contact. The terminal current is therefore the sum of the primal fluxes from the triangles into the boundary-edge nodes on the top contact. Electrons leaving the device carry negative charge out, which is conventional current in; holes leaving carry conventional current out. With the fluxes computed in the solver's scaling (densities in units of 1e16 cm^-3, dimensionless geometric coefficients, D in cm^2/s), the summed flux is a particle flow per unit depth, which becomes a current density after multiplying by q and 1e16 and dividing by the 1 micrometre (1e-4 cm) device width. At 0 V the scheme reproduces equilibrium exactly, so the current must vanish to solver precision, which is a direct check of the implementation.

Returns
-------
float: terminal current density at the top contact in A/cm^2, positive entering the device.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def terminal_current(psi: "np.ndarray", n: "np.ndarray", p: "np.ndarray", M: int) -> float:
    """Current density through the top contact (y = 1) in A/cm^2.

    Parameters
    ----------
    psi, n, p : "np.ndarray"
        Node values in V, cm^-3, cm^-3, shape (Nn,), in the node order of
        ddfv_geometry(*build_mesh(M)), as returned by solve_drift_diffusion.
    M : int
        Mesh parameter.

    Returns
    -------
    J : float
        q * 1e16 * (sum F - sum G) / 1e-4, where F and G are the primal electron
        and hole fluxes of ddfv_ha_fluxes (u = psi/V_T, n and p divided by 1e16,
        lam2 as in solve_drift_diffusion, Dn = V_T*1417.0, Dp = V_T*470.5) over
        the boundary diamonds whose edge lies on y = 1. Positive for
        conventional current entering the device through the top contact.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return J  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_terminal_current(psi: "np.ndarray", n: "np.ndarray", p: "np.ndarray", M: int) -> float:
    q, eps, VT, Nstar = 1.602192e-19, 1.035941e-12, 0.025852, 1e16
    lam2 = eps * VT / (q * Nstar * 1e-8)
    pts, tri = _oracle_build_mesh(M)
    geom = _oracle_ddfv_geometry(pts, tri)
    fl = _oracle_ddfv_ha_fluxes(geom, np.asarray(psi) / VT, np.asarray(n) / Nstar,
                                np.asarray(p) / Nstar, lam2, VT * 1417.0, VT * 470.5)
    dia = geom["diamonds"]
    nT = len(tri)
    top = np.zeros(len(dia), dtype=bool)
    b = np.where(dia[:, 1] >= nT)[0]
    ey = pts[geom["bnd_edges"][dia[b, 1] - nT]][:, :, 1]
    top[b] = np.all(np.abs(ey - 1.0) < 1e-12, axis=1)
    return float(q * Nstar * (fl[2][top].sum() - fl[4][top].sum()) / 1e-4)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: forward current at 0.5 V, lowest doping.
        {"setup": "import numpy as np\ns = _oracle_solve_drift_diffusion(1e16, 0.5, 8)\n",
         "call": "terminal_current(s[0], s[1], s[2], 8)",
         "gold_call": "_oracle_terminal_current(s[0], s[1], s[2], 8)"},
        # Normal: forward current at 0.3 V, highest doping.
        {"setup": "import numpy as np\ns = _oracle_solve_drift_diffusion(1e17, 0.3, 8)\n",
         "call": "terminal_current(s[0], s[1], s[2], 8)",
         "gold_call": "_oracle_terminal_current(s[0], s[1], s[2], 8)"},
        # Boundary: thermal equilibrium, where the current must vanish.
        {"setup": "import numpy as np\ns = _oracle_solve_drift_diffusion(3e16, 0.0, 8)\n",
         "call": "bool(abs(terminal_current(s[0], s[1], s[2], 8)) < 1e-6)",
         "gold_call": "bool(abs(_oracle_terminal_current(s[0], s[1], s[2], 8)) < 1e-6)"},
    ]
