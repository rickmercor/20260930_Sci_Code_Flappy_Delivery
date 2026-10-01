"""
Implement quasiparticle_gap, the end-to-end pipeline, which returns the quasiparticle
HOMO-LUMO gap of a closed-shell molecule, in electronvolts, from the renormalized
quasiparticle-self-consistent second-order Green's-function method in a given contracted
Gaussian basis.

The quasiparticle HOMO-LUMO gap is the difference between the lowest unoccupied and the highest
occupied quasiparticle energy.

Returns
-------
float, quasiparticle HOMO-LUMO gap (e_LUMO - e_HOMO) in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quasiparticle_gap(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int, s: float, c_ss: float, c_os: float) -> float:
    '''Quasiparticle HOMO-LUMO gap (eV) of a closed-shell molecule.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    shells : list
        Basis shells as in compute_ao_integrals.
    n_electrons : int
        Number of electrons, a positive even integer smaller than 2 n_bf.
    s, c_ss, c_os : float
        Flow parameter (hartree^-2) and same-/opposite-spin scaling factors, as in
        srg_self_energy.

    Returns
    -------
    gap : float
        (e_LUMO - e_HOMO) * 27.211386245988, in eV, where e_HOMO and e_LUMO are the
        quasiparticle energies with indices n_electrons/2 - 1 and n_electrons/2 of
        run_srg_qsgf2 for the integrals of compute_ao_integrals.

    Raises
    ------
    ValueError
        If n_electrons is not a positive even integer with n_electrons/2 < n_bf, or the
        inputs are invalid as in compute_ao_integrals or run_srg_qsgf2.
    '''
    return gap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_quasiparticle_gap(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int, s: float, c_ss: float, c_os: float) -> float:
    if int(n_electrons) != n_electrons or n_electrons <= 0 or int(n_electrons) % 2 != 0:
        raise ValueError("n_electrons must be a positive even integer")
    n_occ = int(n_electrons) // 2
    overlap, core_hamiltonian, eri, _ = _oracle_compute_ao_integrals(charges, coords, shells)
    if n_occ >= overlap.shape[0]:
        raise ValueError("the basis has no virtual orbital for this electron count")
    qp = _oracle_run_srg_qsgf2(overlap, core_hamiltonian, eri, n_occ, s, c_ss, c_os)
    hartree_to_ev = 27.211386245988
    return float((qp[n_occ] - qp[n_occ - 1]) * hartree_to_ev)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    basis = """import numpy as np
from copy import deepcopy as dc
N_AUG = [
    (0, [1027.828458, 188.4512226, 52.72186097, 18.11138217, 7.033179691, 2.896651794],
        [0.0091635963, 0.0493614929, 0.1685383049, 0.3705627997, 0.4164915298, 0.1303340841]),
    (0, [0.1846836552, 0.402911141, 0.9277239437, 2.411325783, 7.758467071, 39.19880787],
        [0.2407061763, 0.5951172526, 0.2502417861, -0.0337853715, -0.0469917101, -0.0132527881]),
    (0, [0.09, 0.1846836552, 0.402911141, 0.9277239437],
        [-155.7296559966, 61.8126687524, -69.8698010234, -5.5545900093]),
    (1, [0.1846836552, 0.402911141, 0.9277239437, 2.411325783, 7.758467071, 39.19880787],
        [0.1017082955, 0.4258595477, 0.4180364347, 0.1738967435, 0.0376793698, 0.0037596966]),
    (1, [0.09, 0.1846836552, 0.402911141, 0.9277239437],
        [-0.0840929047, -0.0859046913, 0.0449394729, 0.150796203]),
    (2, [0.09, 0.1846836552, 0.402911141, 0.9277239437],
        [-0.0966051188, -0.131498833, 0.0681453767, -0.1191458075]),
]
H_AUG = [
    (0, [0.100112428, 0.2430767471, 0.6259552659, 1.822142904, 6.513143725, 35.52322122],
        [0.1303340841, 0.4164915298, 0.3705627997, 0.1685383049, 0.0493614929, 0.0091635963]),
    (0, [0.0654284286, 0.100112428, 0.2430767471, 0.6259552659],
        [4.8518572723, 0.4054347007, 0.0547707665, 0.1436562367]),
    (0, [0.0654284286, 0.100112428, 0.2430767471, 0.6259552659],
        [1.746076269, -5.7363906541, -0.3296656157, 0.1098876093]),
    (1, [0.0654284286, 0.100112428, 0.2430767471, 0.6259552659],
        [0.4762021876, -0.1110297174, 5.00489e-05, 0.0083992741]),
]
def on_atom(atom, entries):
    return [(atom, l, list(e), list(c)) for (l, e, c) in entries]
"""
    return [
        # --- Typical: pyramidal NH3 (10 electrons, HOMO index 4) with core s, valence s and
        #     valence p shells on N and a contracted 1s shell on each H, opposite-spin-only
        #     parameters (c_ss = 0) ---
        {
            "setup": basis + """
charges = np.array([7.0, 1.0, 1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [1.77, 0.0, -0.72], [-0.885, 1.533, -0.72], [-0.885, -1.533, -0.72]])
shells = on_atom(0, [N_AUG[0], N_AUG[1], N_AUG[3]]) + sum((on_atom(a, [H_AUG[0]]) for a in (1, 2, 3)), [])
""",
            "call": "quasiparticle_gap(dc(charges), dc(coords), dc(shells), 10, 1.4, 0.0, 1.0)",
            "gold_call": "_oracle_quasiparticle_gap(dc(charges), dc(coords), dc(shells), 10, 1.4, 0.0, 1.0)",
            "tol": 1e-7,
        },
        # --- Edge: equilateral H3+ cation (2 electrons, side 1.65 bohr, degenerate LUMO pair),
        #     unscaled self-energy with s = 0.525 ---
        {
            "setup": basis + """
charges = np.array([1.0, 1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [1.65, 0.0, 0.0], [0.825, 0.825 * np.sqrt(3.0), 0.0]])
shells = sum((on_atom(a, [H_AUG[0], H_AUG[1]]) for a in range(3)), [])
""",
            "call": "quasiparticle_gap(dc(charges), dc(coords), dc(shells), 2, 0.525, 1.0, 1.0)",
            "gold_call": "_oracle_quasiparticle_gap(dc(charges), dc(coords), dc(shells), 2, 0.525, 1.0, 1.0)",
            "tol": 1e-7,
        },
        # --- Boundary: s = 0 gives the Hartree-Fock HOMO-LUMO gap of H2 (contracted 1s and p
        #     shells) ---
        {
            "setup": basis + """
charges = np.array([1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
shells = on_atom(0, [H_AUG[0], H_AUG[3]]) + on_atom(1, [H_AUG[0], H_AUG[3]])
""",
            "call": "quasiparticle_gap(dc(charges), dc(coords), dc(shells), 2, 0.0, 0.0, 1.0)",
            "gold_call": "_oracle_quasiparticle_gap(dc(charges), dc(coords), dc(shells), 2, 0.0, 0.0, 1.0)",
            "tol": 1e-7,
        },
        # --- Typical: linear H4 chain with unequal spacings and both spin components scaled (the
        #     same-spin terms are active with two occupied orbitals) ---
        {
            "setup": basis + """
charges = np.array([1.0, 1.0, 1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, 3.4], [0.0, 0.0, 4.9]])
shells = sum((on_atom(a, [H_AUG[0], H_AUG[1]]) for a in range(4)), [])
""",
            "call": "quasiparticle_gap(dc(charges), dc(coords), dc(shells), 4, 0.7, 0.6, 1.0)",
            "gold_call": "_oracle_quasiparticle_gap(dc(charges), dc(coords), dc(shells), 4, 0.7, 0.6, 1.0)",
            "tol": 1e-7,
        },
        # --- Invalid: odd number of electrons ---
        {
            "setup": basis + """
charges = np.array([1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
shells = on_atom(0, [H_AUG[0], H_AUG[3]]) + on_atom(1, [H_AUG[0], H_AUG[3]])
def run_model():
    try:
        quasiparticle_gap(dc(charges), dc(coords), dc(shells), 3, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_quasiparticle_gap(dc(charges), dc(coords), dc(shells), 3, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
