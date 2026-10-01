"""
Oil/water partition coefficient of the flexible two-bead molecule; the final orchestrator step of the pipeline.

The water phase and the oil phase are each described by their own solvent composition, repulsion matrix and bead amplitudes; the bond is the same in both. At infinite dilution the molecule distributes between coexisting phases so that its chemical potential is the same in both, and the partition coefficient P is the ratio of its concentration in the oil to its concentration in the water. Return the base-10 logarithm of P.

Returns
-------
float: log10 of the oil/water partition coefficient of the flexible dimer at infinite dilution
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dimer_partition_coefficient(rho_water: float, x_water: np.ndarray, A_water: np.ndarray, a_water: np.ndarray, rho_oil: float, x_oil: np.ndarray, A_oil: np.ndarray, a_oil: np.ndarray, k_bond: float, l0: float, n_grid: int, dr: float) -> float:
    '''Base-10 logarithm of the oil/water partition coefficient of a flexible dimer.

    Parameters
    ----------
    rho_water, rho_oil : float
        Total bead densities of the two solvents.
    x_water, x_oil : numpy.ndarray
        Mole fractions of the solvent species of each phase, shapes (m_w,) and
        (m_o,), each positive and summing to one.
    A_water, A_oil : numpy.ndarray
        Symmetric solvent-solvent repulsion amplitudes of each phase, shapes
        (m_w, m_w) and (m_o, m_o).
    a_water, a_oil : numpy.ndarray
        Solvent amplitudes of the two beads in each phase, shapes (m_w, 2) and
        (m_o, 2); column 0 is bead 0 and column 1 is bead 1.
    k_bond : float
        Bond constant k in beta phi_b(r) = k (r - l0)^2.
    l0 : float
        Bond rest length.
    n_grid : int
        Number of grid intervals N. Radial functions are sampled at r_i = i * dr
        for i = 1, ..., N - 1 and their transforms at q_j = j * pi / (N * dr).
    dr : float
        Grid spacing in units of the interaction range.

    Returns
    -------
    log10_p : float
        log10 of P = (concentration in the oil) / (concentration in the water)
        at infinite dilution.

    Raises
    ------
    ValueError
        If either phase or the dimer parameters are invalid in the sense of the
        earlier steps, or if an iteration fails to converge.
    '''
    return log10_p

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dimer_partition_coefficient(rho_water: float, x_water: np.ndarray, A_water: np.ndarray,
                                        a_water: np.ndarray, rho_oil: float, x_oil: np.ndarray,
                                        A_oil: np.ndarray, a_oil: np.ndarray, k_bond: float, l0: float,
                                        n_grid: int, dr: float) -> float:
    import numpy as np
    mu_water = _oracle_dimer_chemical_potential(rho_water, x_water, A_water, a_water, k_bond, l0, n_grid, dr)
    mu_oil = _oracle_dimer_chemical_potential(rho_oil, x_oil, A_oil, a_oil, k_bond, l0, n_grid, dr)
    return float((mu_water - mu_oil) / np.log(10.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the target dimer between the water-like solvent and the wet oil ---
        {
            "setup": """import numpy as np
rho_w = 3.0
x_w = np.array([1.0])
A_w = np.array([[25.0]])
a_w = np.array([[38.0, 24.0]])
rho_o = 3.0
x_o = np.array([0.10, 0.675, 0.225])
A_o = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a_o = np.array([[38.0, 24.0], [26.0, 34.0], [33.0, 22.0]])
k_bond = 150.0
l0 = 0.5
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_partition_coefficient(rho_w, x_w.copy(), A_w.copy(), a_w.copy(), rho_o, x_o.copy(), A_o.copy(), a_o.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_partition_coefficient(rho_w, x_w.copy(), A_w.copy(), a_w.copy(), rho_o, x_o.copy(), A_o.copy(), a_o.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Boundary: the two phases exchanged ---
        {
            "setup": """import numpy as np
rho_w = 3.0
x_w = np.array([1.0])
A_w = np.array([[25.0]])
a_w = np.array([[38.0, 24.0]])
rho_o = 3.0
x_o = np.array([0.10, 0.675, 0.225])
A_o = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a_o = np.array([[38.0, 24.0], [26.0, 34.0], [33.0, 22.0]])
k_bond = 150.0
l0 = 0.5
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_partition_coefficient(rho_o, x_o.copy(), A_o.copy(), a_o.copy(), rho_w, x_w.copy(), A_w.copy(), a_w.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_partition_coefficient(rho_o, x_o.copy(), A_o.copy(), a_o.copy(), rho_w, x_w.copy(), A_w.copy(), a_w.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Edge: identical phases ---
        {
            "setup": """import numpy as np
rho_w = 3.0
x_w = np.array([1.0])
A_w = np.array([[25.0]])
a_w = np.array([[38.0, 24.0]])
rho_o = 3.0
x_o = np.array([0.10, 0.675, 0.225])
A_o = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a_o = np.array([[38.0, 24.0], [26.0, 34.0], [33.0, 22.0]])
k_bond = 150.0
l0 = 0.5
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_partition_coefficient(rho_o, x_o.copy(), A_o.copy(), a_o.copy(), rho_o, x_o.copy(), A_o.copy(), a_o.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_partition_coefficient(rho_o, x_o.copy(), A_o.copy(), a_o.copy(), rho_o, x_o.copy(), A_o.copy(), a_o.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Normal: a longer, softer bond ---
        {
            "setup": """import numpy as np
rho_w = 3.0
x_w = np.array([1.0])
A_w = np.array([[25.0]])
a_w = np.array([[38.0, 24.0]])
rho_o = 3.0
x_o = np.array([0.10, 0.675, 0.225])
A_o = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a_o = np.array([[38.0, 24.0], [26.0, 34.0], [33.0, 22.0]])
k_bond = 100.0
l0 = 0.8
n_grid = 1024
dr = 0.02
""",
            "call": "dimer_partition_coefficient(rho_w, x_w.copy(), A_w.copy(), a_w.copy(), rho_o, x_o.copy(), A_o.copy(), a_o.copy(), k_bond, l0, n_grid, dr)",
            "gold_call": "_oracle_dimer_partition_coefficient(rho_w, x_w.copy(), A_w.copy(), a_w.copy(), rho_o, x_o.copy(), A_o.copy(), a_o.copy(), k_bond, l0, n_grid, dr)",
            "tol": 1e-07,
        },
    ]
