"""
Implement stretch_energy, which returns the correlation-driven spin-component-scaled MP2
energy change between two geometries of the same closed-shell molecule.

Each geometry is a separate system for the correlation-driven scheme: its own restricted
Hartree-Fock reference, its own spin-resolved MP2 correlation energy and its own pair of
correlation-driven weights. The energy change is the difference of the two total
energies, final minus initial, so that stretching a bond gives a positive value.

Returns
-------
float, E(final) - E(initial) in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stretch_energy(charges: "np.ndarray", coords_initial: "np.ndarray", coords_final: "np.ndarray", shells: list, n_electrons: int) -> float:
    '''Correlation-driven SCS-MP2 energy change E(final) - E(initial) of one molecule.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coords_initial : np.ndarray
        Nuclear positions of the initial geometry in bohr, shape (n_atoms, 3).
    coords_final : np.ndarray
        Nuclear positions of the final geometry in bohr, shape (n_atoms, 3), same atom order.
    shells : list
        Basis shells (atom, l, exponents, coefficients) in the format of
        compute_ao_integrals; the same shells are used at both geometries.
    n_electrons : int
        Number of electrons of the closed-shell singlet, positive and even.

    Returns
    -------
    delta_e : float
        E(final) - E(initial) in hartree, each energy evaluated with the two-parameter
        correlation-driven weights of the molecule at that geometry.

    Raises
    ------
    ValueError
        If the two geometries do not have the shape (n_atoms, 3) implied by charges, or
        any input is rejected by the steps it relies on.
    '''
    return delta_e

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_stretch_energy(charges: "np.ndarray", coords_initial: "np.ndarray", coords_final: "np.ndarray", shells: list, n_electrons: int) -> float:
    charges = np.asarray(charges, dtype=float).ravel()
    coords_initial = np.asarray(coords_initial, dtype=float)
    coords_final = np.asarray(coords_final, dtype=float)
    if coords_initial.shape != (charges.size, 3) or coords_final.shape != (charges.size, 3):
        raise ValueError("both geometries must have shape (n_atoms, 3)")
    e_initial = _oracle_cd_scs_mp2_energy(charges, coords_initial, shells, n_electrons)
    e_final = _oracle_cd_scs_mp2_energy(charges, coords_final, shells, n_electrons)
    return float(e_final - e_initial)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
from copy import deepcopy as dc
o_631g = [
    (0, [5484.671660, 825.2349460, 188.0469580, 52.96450000, 16.89757040, 5.799635340],
     [0.001831074430, 0.01395017220, 0.06844507810, 0.2327143360, 0.4701928980, 0.3585208530]),
    (0, [15.53961625, 3.599933586, 1.013761750], [-0.1107775495, -0.1480262627, 1.130767015]),
    (1, [15.53961625, 3.599933586, 1.013761750], [0.07087426823, 0.3397528391, 0.7271585773]),
    (0, [0.2700058226], [1.0]),
    (1, [0.2700058226], [1.0]),
]
h_631g = [
    (0, [18.73113696, 2.825394365, 0.6401216923], [0.03349460434, 0.2347269535, 0.8137573261]),
    (0, [0.1612777588], [1.0]),
]
def water_coords(r, angle_deg):
    half = np.deg2rad(angle_deg / 2.0)
    return np.array([[0.0, 0.0, 0.0], [0.0, r * np.sin(half), r * np.cos(half)], [0.0, -r * np.sin(half), r * np.cos(half)]])
water_shells = [(0, l, e, c) for (l, e, c) in o_631g]
for atom in (1, 2):
    water_shells += [(atom, l, e, c) for (l, e, c) in h_631g]
h2_shells = [(0, l, e, c) for (l, e, c) in h_631g] + [(1, l, e, c) for (l, e, c) in h_631g]
"""
    return [
        # --- Typical: symmetric water stretch from 1.2 to 1.6 times the reference bond length ---
        {
            "setup": helper + """
charges = np.array([8.0, 1.0, 1.0])
coords_initial = water_coords(1.2 * 1.84345, 104.5)
coords_final = water_coords(1.6 * 1.84345, 104.5)
""",
            "call": "stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(water_shells), 10)",
            "gold_call": "_oracle_stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(water_shells), 10)",
            "tol": 1e-8,
        },
        # --- Typical: H2 stretched to twice its bond length (weights change strongly) ---
        {
            "setup": helper + """
charges = np.array([1.0, 1.0])
coords_initial = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
coords_final = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.8]])
""",
            "call": "stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(h2_shells), 2)",
            "gold_call": "_oracle_stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(h2_shells), 2)",
            "tol": 1e-8,
        },
        # --- Edge: compression (negative displacement of one bond, final energy higher) ---
        {
            "setup": helper + """
charges = np.array([8.0, 1.0, 1.0])
coords_initial = water_coords(1.84345, 110.565)
coords_final = coords_initial.copy()
coords_final[1] = 0.8 * coords_final[1]
""",
            "call": "stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(water_shells), 10)",
            "gold_call": "_oracle_stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(water_shells), 10)",
            "tol": 1e-8,
        },
        # --- Invalid: geometries with different numbers of atoms ---
        {
            "setup": helper + """
charges = np.array([1.0, 1.0])
coords_initial = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
coords_final = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, 2.8]])
def run_model():
    try:
        stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(h2_shells), 2)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_stretch_energy(dc(charges), dc(coords_initial), dc(coords_final), dc(h2_shells), 2)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
