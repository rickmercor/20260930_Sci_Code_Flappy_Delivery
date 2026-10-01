"""
Step 05: Lowest self-consistent LSDA energy of a one-electron system.

Lowest-energy self-consistent exchange-only local spin-density energy of one electron bound to soft-Coulomb protons.

For a single spin-up electron the exchange-only local spin-density energy of a normalized real orbital phi is
E[phi] = <phi| T + v |phi> + J[phi^2] + E_x[phi^2, 0], with v the soft-Coulomb nuclear attraction of step 03 and J + E_x
the Hartree plus exchange energy of step 04. The exact functional would make J + E_x vanish for any one-electron density;
the local approximation does not, so the minimizing orbital is a genuine self-consistent Kohn-Sham solution whose energy
differs from the exact eigenvalue. Stationary orbitals satisfy the Kohn-Sham equation
[T + v + v_H + v_x] phi = e phi, where v_H(x) = integral n(x') w(x - x') dx' and v_x = d[n eps_x]/dn of step 02, and
the ground state occupies the lowest orbital of that equation.

For stretched two-center systems this nonlinear problem has more than one stationary solution and simple fixed-point
iteration can oscillate between the two wells or settle on a solution that is not the minimum. The quantity required here
is the lowest total energy among the stationary solutions. Only nuclear configurations that are symmetric under
x -> -x are treated.

Everything is discretized exactly as in steps 03 and 04: the grid x_j = -L + j*h_x, the three-point kinetic operator with
the orbital zero outside the grid, rectangle-rule integrals, and h_x sum_j phi_j^2 = 1. The energy excludes the
repulsion between nuclei and is converged to better than 10^-10 hartree.

Returns
-------
float, lowest self-consistent exchange-only LSDA electronic energy in hartree of one spin-up electron
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lsda_ground_state_energy(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float) -> float:
    '''Lowest self-consistent exchange-only LSDA energy of one spin-up electron in a symmetric soft-Coulomb field.

    Parameters
    ----------
    nuclei : np.ndarray
        Shape (P,), P >= 1, positions of unit point charges in bohr; the set of positions must be symmetric under
        x -> -x within 1e-9 bohr.
    softening : float
        Softening length b > 0 used for both the electron-nucleus and the electron-electron interaction, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x an integer.

    Returns
    -------
    result : float
        The lowest total electronic energy in hartree among the self-consistent solutions, without nuclear repulsion.

    Raises
    ------
    ValueError
        If the nuclear configuration is not symmetric under x -> -x, or for any invalid grid or softening as in
        soft_coulomb_states.
    RuntimeError
        If the self-consistent iteration fails to converge within 5000 cycles.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh_tridiagonal


def _oracle_lsda_ground_state_energy(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import eigh_tridiagonal
    centers = np.sort(np.asarray(nuclei, dtype=float).ravel())
    if centers.size == 0 or not np.allclose(centers, -centers[::-1], rtol=0.0, atol=1e-9):
        raise ValueError("the nuclear configuration must be symmetric under x -> -x")
    guess = _oracle_soft_coulomb_states(centers, softening, spacing, half_width, 1)
    phi = guess[0, 1:]
    m = phi.size - 1
    x = -half_width + spacing * np.arange(m + 1)
    v = np.zeros_like(x)
    for c in centers:
        v -= 1.0 / np.sqrt((x - c) ** 2 + softening ** 2)
    idx = np.arange(m + 1)
    kernel = 1.0 / np.sqrt(((idx[:, None] - idx[None, :]) * spacing) ** 2 + softening ** 2)
    off = np.full(m, -0.5 / spacing ** 2)
    n = phi ** 2
    hist = []
    for _ in range(5000):
        veff = v + spacing * (kernel @ n) + _oracle_polarized_exchange_potential(n, softening)
        veff = 0.5 * (veff + veff[::-1])
        _, u = eigh_tridiagonal(1.0 / spacing ** 2 + veff, off, select="i", select_range=(0, 0))
        phi = u[:, 0] / np.sqrt(spacing)
        new = phi ** 2
        res = new - n
        if spacing * np.abs(res).sum() < 1e-12:
            break
        hist.append((n.copy(), res.copy()))
        hist = hist[-6:]
        if len(hist) >= 2:
            d_res = np.array([hist[i + 1][1] - hist[i][1] for i in range(len(hist) - 1)]).T
            d_n = np.array([hist[i + 1][0] - hist[i][0] for i in range(len(hist) - 1)]).T
            coef = np.linalg.lstsq(d_res, res, rcond=None)[0]
            n = n + 0.3 * res - (d_n + 0.3 * d_res) @ coef
            n = np.maximum(n, 0.0)
            n = n / (spacing * n.sum())
        else:
            n = n + 0.3 * res
    else:
        raise RuntimeError("self-consistent field did not converge")
    padded = np.concatenate(([0.0], phi, [0.0]))
    kinetic = -0.5 * spacing * float(np.sum(phi * (padded[2:] - 2.0 * phi + padded[:-2]))) / spacing ** 2
    potential = spacing * float(np.sum(v * phi ** 2))
    zero = np.zeros_like(phi)
    return float(kinetic + potential + _oracle_hartree_exchange_energy(phi ** 2, zero, spacing, softening))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the one-dimensional hydrogen atom ---
        {
            "setup": "import numpy as np\n",
            "call": "lsda_ground_state_energy(np.array([0.0]), 1.0, 0.1, 20.0)",
            "gold_call": "_oracle_lsda_ground_state_energy(np.array([0.0]), 1.0, 0.1, 20.0)",
            "tol": 1e-8,
        },
        # --- Normal: H2+ near its equilibrium separation ---
        {
            "setup": "import numpy as np\n",
            "call": "lsda_ground_state_energy(np.array([-1.1, 1.1]), 1.0, 0.1, 20.0)",
            "gold_call": "_oracle_lsda_ground_state_energy(np.array([-1.1, 1.1]), 1.0, 0.1, 20.0)",
            "tol": 1e-8,
        },
        # --- Edge: stretched H2+ where the charge sloshes between the wells under plain iteration ---
        {
            "setup": "import numpy as np\n",
            "call": "lsda_ground_state_energy(np.array([-2.6, 2.6]), 1.0, 0.1, 20.0)",
            "gold_call": "_oracle_lsda_ground_state_energy(np.array([-2.6, 2.6]), 1.0, 0.1, 20.0)",
            "tol": 1e-8,
        },
        # --- Edge: strongly stretched H2+ with competing stationary solutions ---
        {
            "setup": "import numpy as np\n",
            "call": "lsda_ground_state_energy(np.array([-4.25, 4.25]), 1.0, 0.1, 24.0)",
            "gold_call": "_oracle_lsda_ground_state_energy(np.array([-4.25, 4.25]), 1.0, 0.1, 24.0)",
            "tol": 1e-8,
        },
        # --- Normal: a symmetric three-center one-electron ion with a shorter softening ---
        {
            "setup": "import numpy as np\n",
            "call": "lsda_ground_state_energy(np.array([-3.0, 0.0, 3.0]), 0.8, 0.08, 16.0)",
            "gold_call": "_oracle_lsda_ground_state_energy(np.array([-3.0, 0.0, 3.0]), 0.8, 0.08, 16.0)",
            "tol": 1e-8,
        },
        # --- Error: an asymmetric nuclear configuration must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([-1.0, 1.5]), 1.0, 0.1, 10.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(lsda_ground_state_energy)",
            "gold_call": "_probe(_oracle_lsda_ground_state_energy)",
        },
    ]
