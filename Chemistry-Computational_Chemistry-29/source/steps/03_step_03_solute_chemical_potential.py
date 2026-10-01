"""
Excess chemical potential, in the HNC closure, of single solute beads inserted at infinite dilution into a DPD solvent.

A solute bead s interacts with solvent species mu through the same DPD form with amplitude a_mu_s, and the solute beads are present at vanishing concentration. Treat each solute bead as an additional species of zero mole fraction in the Ornstein-Zernike and HNC equations of the previous step; its correlations with the solvent, h_mu_s and c_mu_s, then follow from the converged solvent structure.

Return, for each bead, the excess chemical potential beta mu_s, the reversible work of inserting one bead into the solvent in units of k_B T. Within the HNC closure this quantity has an exact closed form in the converged solute-solvent correlation functions (the Kirkwood coupling-parameter integral taken along the HNC path gives the same number), and that exact HNC value is required. Radial integrals are rectangle-rule sums over the grid points r_i.

Returns
-------
numpy.ndarray of shape (n_s,): the HNC excess chemical potential beta mu_s of each infinitely dilute solute bead, in units of k_B T
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solute_chemical_potential(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray, n_grid: int, dr: float) -> np.ndarray:
    '''HNC excess chemical potentials of infinitely dilute solute beads.

    Parameters
    ----------
    rho : float
        Total bead number density of the solvent, in beads per cubed range.
    x : numpy.ndarray
        Mole fractions of the m solvent species, shape (m,), non-negative and
        summing to one.
    A : numpy.ndarray
        Symmetric (m, m) array of non-negative solvent-solvent repulsion
        amplitudes A_mu_nu, in units of k_B T.
    a_solute : numpy.ndarray
        Array of shape (m, n_s) of non-negative solute-solvent amplitudes;
        column s holds the amplitudes a_mu_s of solute bead s towards the m
        solvent species.
    n_grid : int
        Number of grid intervals N. Radial functions are sampled at r_i = i * dr
        for i = 1, ..., N - 1 and their transforms at q_j = j * pi / (N * dr).
    dr : float
        Grid spacing in units of the interaction range.

    Returns
    -------
    mu_ex : numpy.ndarray
        Array of shape (n_s,), the HNC excess chemical potential beta mu_s of
        each solute bead in units of k_B T.

    Raises
    ------
    ValueError
        If rho is not positive and finite, x is not a one dimensional array of
        non-negative finite mole fractions summing to one within 1e-12, A is not
        a finite, non-negative, exactly symmetric array of shape (m, m),
        a_solute is not a finite non-negative array of shape (m, n_s) with
        n_s >= 1,
        n_grid is not an integer of at least 16, dr is not positive and finite,
        or (n_grid - 1) * dr is smaller than 4,
        or if an iteration fails to converge.
    '''
    return mu_ex

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_solute(a_solute, m):
    import numpy as np
    a = np.asarray(a_solute, dtype=float)
    if a.ndim != 2 or a.shape[0] != m or a.shape[1] < 1 or not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("a_solute must be a finite non-negative array of shape (m, n_solute)")
    return a


def _solute_correlations(rho, x, A, a, n_grid, dr):
    """Infinite-dilution solute-solvent HNC correlations h, c of shape (m, n_s, n_grid - 1), with the solvent h, c."""
    import numpy as np
    hc = _oracle_solvent_structure(rho, x, A, n_grid, dr)
    h00q = np.moveaxis(_oracle_radial_fourier_transform(hc[0], dr, False), -1, 0)   # (nq, m, m)
    r = dr * np.arange(1, n_grid)
    bu = _dpd_potential(a, r)
    dens = rho * x

    def _update(gam):
        c = np.expm1(-bu + gam) - gam
        cq = np.moveaxis(_oracle_radial_fourier_transform(c, dr, False), -1, 0)        # (nq, m, ns)
        g = np.einsum('qmn,n,qns->qms', h00q, dens, cq)
        return _oracle_radial_fourier_transform(np.moveaxis(g, 0, -1), dr, True)

    gam = _damped_iteration(_update, np.zeros(a.shape + (n_grid - 1,)))
    c = np.expm1(-bu + gam) - gam
    return gam + c, c, hc


def _oracle_solute_chemical_potential(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray,
                                      n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    a = _check_solute(a_solute, x.size)
    h, c, _ = _solute_correlations(rho, x, A, a, n_grid, dr)
    r = dr * np.arange(1, n_grid)
    return rho * np.einsum('m,msr,r->s', x, 0.5 * h * (h - c) - c, 4.0 * np.pi * r * r * dr)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: two unlike beads in the water-like solvent ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([1.0])
A = np.array([[25.0]])
a = np.array([[38.0, 24.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Normal: the same two beads in the wet oil ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.10, 0.675, 0.225])
A = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a = np.array([[38.0, 24.0], [26.0, 34.0], [33.0, 22.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Boundary: a solute bead identical to the solvent bead ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([1.0])
A = np.array([[25.0]])
a = np.array([[25.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Edge: a bead with no solvent interaction next to a repulsive one in a binary solvent ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.75, 0.25])
A = np.array([[25.0, 30.0], [30.0, 25.0]])
a = np.array([[0.0, 30.0], [0.0, 20.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Edge: a solvent species at zero mole fraction ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.8, 0.2, 0.0])
A = np.array([[25.0, 28.0, 40.0], [28.0, 25.0, 22.0], [40.0, 22.0, 25.0]])
a = np.array([[30.0], [20.0], [50.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solute_chemical_potential(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-08,
        },
    ]
