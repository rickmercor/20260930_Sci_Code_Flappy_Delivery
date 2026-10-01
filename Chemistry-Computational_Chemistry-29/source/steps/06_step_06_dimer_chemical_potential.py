"""
Excess chemical potential of a flexible two-bead molecule at infinite dilution in a DPD solvent.

The two beads, 0 and 1, are joined by the harmonic bond beta phi_b(r) = k (r - l0)^2, which is their only intramolecular interaction. The excess chemical potential of the molecule is the reversible work of inserting it into the solvent, in units of k_B T, measured relative to the same molecule in the ideal gas, so that it vanishes when the beads do not interact with the solvent.

At a fixed bead separation the insertion work is the sum of the two single-bead excess chemical potentials and the solvent-mediated potential of mean force at that separation, and in the ideal gas the molecule samples separations with the bond distribution. Averages over the bond are three-dimensional integrals over the separation vector, evaluated as rectangle-rule sums over the grid points.

Returns
-------
float: the excess chemical potential of the flexible dimer relative to the ideal-gas dimer, in units of k_B T
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dimer_chemical_potential(rho: float, x: np.ndarray, A: np.ndarray, a_pair: np.ndarray, k_bond: float, l0: float, n_grid: int, dr: float) -> float:
    '''Excess chemical potential of a flexible dimer at infinite dilution.

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
    a_pair : numpy.ndarray
        Array of shape (m, 2); column 0 holds the solvent amplitudes of bead 0
        and column 1 those of bead 1.
    k_bond : float
        Bond constant k in beta phi_b(r) = k (r - l0)^2, positive.
    l0 : float
        Bond rest length, positive and at least 2 inside the end of the grid.
    n_grid : int
        Number of grid intervals N. Radial functions are sampled at r_i = i * dr
        for i = 1, ..., N - 1 and their transforms at q_j = j * pi / (N * dr).
    dr : float
        Grid spacing in units of the interaction range.

    Returns
    -------
    mu_dimer : float
        Excess chemical potential of the dimer relative to the ideal-gas dimer,
        in units of k_B T.

    Raises
    ------
    ValueError
        If rho is not positive and finite, x is not a one dimensional array of
        non-negative finite mole fractions summing to one within 1e-12, A is not
        a finite, non-negative, exactly symmetric array of shape (m, m),
        any mole fraction is zero, a_pair is not a finite non-negative array of
        shape (m, 2), k_bond is not positive and finite, l0 is not positive or
        exceeds (n_grid - 1) * dr - 2,
        n_grid is not an integer of at least 16, dr is not positive and finite,
        or (n_grid - 1) * dr is smaller than 4,
        or if an iteration fails to converge.
    '''
    return mu_dimer

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dimer_chemical_potential(rho: float, x: np.ndarray, A: np.ndarray, a_pair: np.ndarray,
                                     k_bond: float, l0: float, n_grid: int, dr: float) -> float:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    a = _check_solute(a_pair, x.size)
    if a.shape[1] != 2:
        raise ValueError("a_pair must have exactly two columns, one per bead")
    k_bond, l0 = float(k_bond), float(l0)
    if not np.isfinite(k_bond) or k_bond <= 0.0:
        raise ValueError("k_bond must be positive and finite")
    if not np.isfinite(l0) or l0 <= 0.0 or l0 > (n_grid - 1) * dr - 2.0:
        raise ValueError("l0 must be positive and at least 2 inside the end of the grid")
    mu = _oracle_solute_chemical_potential(rho, x, A, a, n_grid, dr)
    w = _oracle_solvent_mediated_pmf(rho, x, A, a, n_grid, dr)[0, 1]
    r = dr * np.arange(1, n_grid)
    bond = r * r * np.exp(-k_bond * (r - l0) ** 2)
    return float(mu[0] + mu[1] - np.log(np.sum(bond * np.exp(-w)) / np.sum(bond)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the flexible dimer in the water-like solvent ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([1.0])
A = np.array([[25.0]])
a = np.array([[38.0, 24.0]])
k_bond = 150.0
l0 = 0.5
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Normal: the flexible dimer in the wet oil ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.10, 0.675, 0.225])
A = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a = np.array([[38.0, 24.0], [26.0, 34.0], [33.0, 22.0]])
k_bond = 150.0
l0 = 0.5
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Boundary: a very stiff bond, close to the rigid dimer ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([1.0])
A = np.array([[25.0]])
a = np.array([[38.0, 24.0]])
k_bond = 5000.0
l0 = 0.6
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Edge: beads with no solvent interaction ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.10, 0.675, 0.225])
A = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a = np.zeros((3, 2))
k_bond = 150.0
l0 = 0.5
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-08,
        },
        # --- Normal: a soft bond longer than the interaction range in a binary solvent ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.75, 0.25])
A = np.array([[25.0, 30.0], [30.0, 25.0]])
a = np.array([[30.0, 22.0], [20.0, 28.0]])
k_bond = 60.0
l0 = 1.3
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_chemical_potential(rho, x.copy(), A.copy(), a.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-08,
        },
    ]
