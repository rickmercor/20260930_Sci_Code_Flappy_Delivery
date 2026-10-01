"""
Solvent-mediated potential of mean force between every pair of solute beads at infinite dilution in a DPD solvent, in the HNC closure.

For two solute beads s and t held at separation r in the solvent, the solvent-mediated potential of mean force beta W_st(r) is the reversible work, over and above the two single-bead insertion free energies, of bringing them together from infinite separation, in units of k_B T.

In the HNC closure, with both beads at infinite dilution, it equals minus the solute-solute indirect correlation function, beta W_st(r) = -[h_st(r) - c_st(r)]. No direct solute-solute potential enters, beta W_st vanishes at large separation, and it follows from the solute-solvent correlations and the solvent structure alone. Return it for every ordered pair of beads, s = t included, on the radial grid.

Returns
-------
numpy.ndarray of shape (n_s, n_s, n_grid - 1): the HNC solvent-mediated potential of mean force beta W_st of every bead pair on the radial grid, in units of k_B T
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solvent_mediated_pmf(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray, n_grid: int, dr: float) -> np.ndarray:
    '''HNC solvent-mediated potential of mean force between infinitely dilute beads.

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
        Array of shape (m, n_s) of non-negative solute-solvent amplitudes, one
        column per solute bead.
    n_grid : int
        Number of grid intervals N. Radial functions are sampled at r_i = i * dr
        for i = 1, ..., N - 1 and their transforms at q_j = j * pi / (N * dr).
    dr : float
        Grid spacing in units of the interaction range.

    Returns
    -------
    pmf : numpy.ndarray
        Array of shape (n_s, n_s, n_grid - 1); pmf[s, t] is beta W_st at
        r_i = i * dr, i = 1, ..., n_grid - 1, in units of k_B T.

    Raises
    ------
    ValueError
        If rho is not positive and finite, x is not a one dimensional array of
        non-negative finite mole fractions summing to one within 1e-12, A is not
        a finite, non-negative, exactly symmetric array of shape (m, m),
        any mole fraction is zero, a_solute is not a finite non-negative array
        of shape (m, n_s) with n_s >= 1,
        n_grid is not an integer of at least 16, dr is not positive and finite,
        or (n_grid - 1) * dr is smaller than 4,
        or if an iteration fails to converge.
    '''
    return pmf

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solvent_mediated_pmf(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray,
                                 n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    psi = _oracle_excluded_volume_functions(rho, x, A, a_solute, n_grid, dr)
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    bp = _virial_pressure(rho, x, A, _oracle_solvent_structure(rho, x, A, n_grid, dr), dr)
    psiq = _oracle_radial_fourier_transform(psi, dr, False)
    return _oracle_radial_fourier_transform(-bp * np.einsum('msq,mtq->stq', psiq, psiq), dr, True)

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
            "call": "solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
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
            "call": "solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Boundary: two identical beads in a binary solvent ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.75, 0.25])
A = np.array([[25.0, 30.0], [30.0, 25.0]])
a = np.array([[30.0, 30.0], [20.0, 20.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Edge: one bead with no solvent interaction ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([1.0])
A = np.array([[25.0]])
a = np.array([[0.0, 35.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_mediated_pmf(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
    ]
