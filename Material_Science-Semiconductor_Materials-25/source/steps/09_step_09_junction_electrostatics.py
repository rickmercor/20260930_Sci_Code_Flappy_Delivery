"""
Extract the depletion width and the peak electric field of the junction from the equilibrium vertex solution.

Because the doping and contacts depend only on y, the solution is independent of x, so the average over each horizontal vertex row gives the potential and density profiles along y, and the scatter within a row measures the error introduced by the distorted mesh. The depletion edges are taken where the majority carrier density has fallen to half the doping: the row-averaged electron density on the n-side and the hole density on the p-side, each located by linear interpolation between the two rows that bracket N0/2. This half-density point sits slightly inside the smooth edge of the real depletion region, whose width is set by the Debye length. The peak field is harder than it looks. The difference of row-averaged potentials divided by h gives the field midway between two rows, which is h/2 away from the junction rather than on it. In the fully depleted material next to the junction Gauss's law gives dE/dy = q N0 / eps, so the field on the junction is the largest row difference plus q N0 h/(2 eps). On the 64-cell mesh at 1e17 cm^-3 this correction is about 11% of the peak field, larger than the physical deviation from the depletion approximation that the study is designed to measure.

Returns
-------
np.ndarray of shape (2,), float: depletion width W (micrometres) and peak field E_max (V/cm).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def junction_electrostatics(psi_v: "np.ndarray", n_v: "np.ndarray", p_v: "np.ndarray",
                            N0: float, M: int) -> "np.ndarray":
    """Depletion width and peak field from the vertex solution.

    Parameters
    ----------
    psi_v, n_v, p_v : "np.ndarray"
        Vertex values (V, cm^-3, cm^-3), shape ((M+1)**2,), vertex (i, j) at
        index j*(M+1) + i as in build_mesh.
    N0 : float
        Doping magnitude in cm^-3.
    M : int
        Mesh parameter; row spacing h = 1/M micrometres.

    Returns
    -------
    result : "np.ndarray"
        Shape (2,): [W in micrometres, E_max in V/cm]. With row averages
        psi_j, n_j, p_j and y_j = j h: y_n is where n_j first drops below N0/2
        scanning up from j = 0, and y_p is where p_j first exceeds N0/2 scanning
        up from j = M/2, each by linear interpolation between the bracketing rows;
        W = y_p - y_n. E_max = max_j |psi_{j+1} - psi_j| / h_cm + q N0 h_cm/(2 eps)
        with h_cm = 1e-4/M, q = 1.602192e-19 C, eps = 1.035941e-12 C/(V cm).

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _row_means(field: "np.ndarray", M: int) -> "np.ndarray":
    """Average a vertex field over each horizontal vertex row of the (M+1) x (M+1) grid."""
    return np.asarray(field, dtype=float).reshape(M + 1, M + 1).mean(axis=1)


def _oracle_junction_electrostatics(psi_v: "np.ndarray", n_v: "np.ndarray", p_v: "np.ndarray",
                                    N0: float, M: int) -> "np.ndarray":
    q, eps = 1.602192e-19, 1.035941e-12
    M = int(M)
    h_um = 1.0 / M
    h_cm = h_um * 1e-4

    ps, nb, pb = (_row_means(f, M) for f in (psi_v, n_v, p_v))
    y = np.arange(M + 1) * h_um
    E = np.max(np.abs(np.diff(ps))) / h_cm + q * N0 * h_cm / (2.0 * eps)
    half = 0.5 * N0
    j = int(np.argmax(nb < half))
    yn = y[j - 1] + (half - nb[j - 1]) / (nb[j] - nb[j - 1]) * h_um
    k = M // 2 + int(np.argmax(pb[M // 2:] > half))
    yp = y[k - 1] + (half - pb[k - 1]) / (pb[k] - pb[k - 1]) * h_um
    return np.array([yp - yn, E])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pick = "V = (M + 1) ** 2\npsi, n, p = s[0, -V:], s[1, -V:], s[2, -V:]\n"
    return [
        # Normal: lowest doping, wide depletion region, 16x16 mesh.
        {"setup": "import numpy as np\nM = 16\ns = _oracle_solve_drift_diffusion(1e16, 0.0, M)\n" + pick,
         "call": "junction_electrostatics(psi, n, p, 1e16, M)",
         "gold_call": "_oracle_junction_electrostatics(psi, n, p, 1e16, M)",
         "tol": 1e-6},
        # Edge: highest doping, depletion edge only a few rows from the junction.
        {"setup": "import numpy as np\nM = 16\ns = _oracle_solve_drift_diffusion(1e17, 0.0, M)\n" + pick,
         "call": "junction_electrostatics(psi, n, p, 1e17, M)",
         "gold_call": "_oracle_junction_electrostatics(psi, n, p, 1e17, M)",
         "tol": 1e-6},
        # Boundary: synthetic y-only profiles with known crossings, independent of the solver.
        {"setup": "import numpy as np\nM = 8\ny = np.repeat(np.arange(M + 1) / M, M + 1)\npsi = 0.4 * np.tanh(6 * (0.5 - y))\nn = 1e16 / (1 + np.exp((y - 0.3) / 0.05))\np = 1e16 / (1 + np.exp((0.7 - y) / 0.05))\n",
         "call": "junction_electrostatics(psi, n, p, 1e16, M)",
         "gold_call": "_oracle_junction_electrostatics(psi, n, p, 1e16, M)",
         "tol": 1e-6},
    ]
