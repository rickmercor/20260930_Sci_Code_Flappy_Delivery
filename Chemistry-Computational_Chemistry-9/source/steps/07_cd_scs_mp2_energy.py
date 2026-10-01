"""
Implement cd_scs_mp2_energy, which returns the total correlation-driven spin-component-scaled
MP2 energy of a closed-shell molecule at one fixed geometry.

The energy is the restricted Hartree-Fock energy plus the opposite-spin and same-spin MP2
correlation components, each multiplied by its own weight. The weights belong to the
molecule at this geometry: they are obtained from the correlation indices of the
molecule's own unrelaxed MP2 density through the two-parameter correlation-driven rule,
so two geometries of the same molecule generally receive different weights. All
electrons are correlated.

Returns
-------
float, the correlation-driven SCS-MP2 total energy in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cd_scs_mp2_energy(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int) -> float:
    '''Correlation-driven spin-component-scaled MP2 total energy at one geometry.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    shells : list
        Basis shells (atom, l, exponents, coefficients) in the format of
        compute_ao_integrals (normalized contracted Cartesian functions).
    n_electrons : int
        Number of electrons of the closed-shell singlet, positive and even.

    Returns
    -------
    energy : float
        E_HF + c_OS * E_OS + c_SS * E_SS in hartree, where E_HF includes nuclear
        repulsion and (c_OS, c_SS) are the two-parameter correlation-driven weights of
        this molecule at this geometry.

    Raises
    ------
    ValueError
        If n_electrons is not a positive even integer, the closed-shell reference has no
        virtual orbital, or any input is rejected by the steps it relies on.
    '''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cd_scs_mp2_energy(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int) -> float:
    if int(n_electrons) != n_electrons or n_electrons <= 0 or int(n_electrons) % 2:
        raise ValueError("n_electrons must be a positive even integer")
    n_occ = int(n_electrons) // 2
    overlap, core, eri, e_nuc = _oracle_compute_ao_integrals(charges, coords, shells)
    e_hf, mo_energies, mo_coeff = _oracle_run_rhf(overlap, core, eri, e_nuc, n_occ)
    e_os, e_ss = _oracle_mp2_spin_components(eri, mo_coeff, mo_energies, n_occ)
    density = _oracle_unrelaxed_mp2_density(eri, mo_coeff, mo_energies, n_occ)
    c_os, c_ss = _oracle_correlation_driven_weights(_oracle_correlation_indices(density, n_electrons))
    return float(e_hf + c_os * e_os + c_ss * e_ss)

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
def water(r, angle_deg):
    half = np.deg2rad(angle_deg / 2.0)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, r * np.sin(half), r * np.cos(half)], [0.0, -r * np.sin(half), r * np.cos(half)]])
    shells = [(0, l, e, c) for (l, e, c) in o_631g]
    for atom in (1, 2):
        shells += [(atom, l, e, c) for (l, e, c) in h_631g]
    return np.array([8.0, 1.0, 1.0]), coords, shells
"""
    return [
        # --- Typical: water near equilibrium ---
        {
            "setup": helper + """
charges, coords, shells = water(1.84345, 110.565)
""",
            "call": "cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 10)",
            "gold_call": "_oracle_cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 10)",
            "tol": 1e-8,
        },
        # --- Edge: water with one bond stretched to 1.8 times the other (asymmetric, larger nondynamic share) ---
        {
            "setup": helper + """
charges, coords, shells = water(1.84345, 104.0)
coords[2] = coords[2] * 1.8
""",
            "call": "cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 10)",
            "gold_call": "_oracle_cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 10)",
            "tol": 1e-8,
        },
        # --- Boundary: two-electron H2 (no same-spin pair, so only the opposite-spin weight matters) ---
        {
            "setup": helper + """
charges = np.array([1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
shells = [(0, l, e, c) for (l, e, c) in h_631g] + [(1, l, e, c) for (l, e, c) in h_631g]
""",
            "call": "cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 2)",
            "gold_call": "_oracle_cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 2)",
            "tol": 1e-9,
        },
        # --- Invalid: an odd electron count cannot form a closed-shell singlet ---
        {
            "setup": helper + """
charges = np.array([1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
shells = [(0, l, e, c) for (l, e, c) in h_631g] + [(1, l, e, c) for (l, e, c) in h_631g]
def run_model():
    try:
        cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 3)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_cd_scs_mp2_energy(dc(charges), dc(coords), dc(shells), 3)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
