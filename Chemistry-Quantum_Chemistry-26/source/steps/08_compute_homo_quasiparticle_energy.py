"""
Compute the self-consistent highest-occupied quasiparticle energy of a closed-shell molecule with one of the source method's published parametrisations, starting from the atomic geometry and basis.

This is the end-to-end step of the task. The negative of the highest occupied quasiparticle
energy is the method's estimate of the first vertical ionisation energy.

Returns
-------
float, the converged quasiparticle energy of the highest occupied molecular orbital in hartree, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_homo_quasiparticle_energy(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list, variant: str, charge: int = 0) -> float:
    '''Converged HOMO quasiparticle energy of the source method for a closed-shell molecule.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    nuclear_charges : np.ndarray
        Nuclear charges Z_A, shape (n_atoms,).
    shells : list of dict
        Basis shells in the format of the one-electron-integral step (keys ``"center"``,
        ``"l"`` in {0, 1}, ``"exponents"``, ``"coefficients"``; coefficients of normalised
        primitives, contracted functions renormalised, p functions ordered x, y, z).
    variant : str
        Which of the source method's published parametrisations to use: ``"plain"`` for
        the unscaled parametrisation, ``"scs"`` for the spin-component-scaled one and
        ``"sos"`` for the scaled-opposite-spin one.
    charge : int
        Total molecular charge; the number of electrons is sum(Z_A) - charge.

    Returns
    -------
    homo : float
        The quasiparticle energy (hartree) of the highest occupied orbital, n_occ =
        (sum(Z_A) - charge) / 2, taken directly from the converged quasiparticle
        self-consistent cycle. The reference is the closed-shell restricted Hartree-Fock
        solution (core-Hamiltonian guess, aufbau occupation) that starts the cycle. All
        electrons are correlated, the integrals are exact (no density fitting), and the
        cycle is converged until the total quasiparticle Fock matrix in the current
        orbitals deviates from diagonal form by less than 1e-10 hartree.

    Raises
    ------
    ValueError
        If ``variant`` is not one of "plain", "scs" or "sos", the electron count is odd
        or not positive, or the basis has fewer functions than occupied orbitals.
    '''
    return homo

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_homo_quasiparticle_energy(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list, variant: str, charge: int = 0) -> float:
    # published (c_ss, c_os, s [hartree^-2]) of the source method's three parametrisations
    parameters = {"plain": (1.0, 1.0, 0.525), "scs": (0.6, 1.0, 0.7), "sos": (0.0, 1.0, 1.4)}
    if variant not in parameters:
        raise ValueError("variant must be 'plain', 'scs' or 'sos'")
    c_ss, c_os, s = parameters[variant]
    coords = np.asarray(coords, dtype=float)
    z = np.asarray(nuclear_charges, dtype=float).ravel()
    n_elec = int(round(z.sum())) - int(charge)
    if n_elec <= 0 or n_elec % 2:
        raise ValueError("a closed-shell reference needs a positive, even electron count")
    n_occ = n_elec // 2
    ints = _oracle_compute_one_electron_integrals(coords, z, shells)
    S, h = ints[0], ints[1] + ints[2]
    if n_occ > S.shape[0]:
        raise ValueError("more occupied orbitals than basis functions")
    eri = _oracle_compute_electron_repulsion_integrals(coords, shells)
    e_nuc = sum(z[a] * z[b] / np.linalg.norm(coords[a] - coords[b])
                for a in range(len(z)) for b in range(a))
    _, eps0, C0 = _oracle_solve_restricted_hartree_fock(S, h, eri, n_occ, e_nuc)
    eps, _ = _oracle_iterate_quasiparticle_self_consistency(
        S, h, eri, n_occ, C0, eps0, s, c_ss, c_os, conv_tol=1e-10, max_iter=300)
    return float(eps[n_occ - 1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    basis = """import copy
import numpy as np
H_STO3G = ([3.425250914, 0.6239137298, 0.1688554040], [0.1543289673, 0.5353281423, 0.4446345422])
O_S = ([130.7093214, 23.80886605, 6.443608313], [0.1543289673, 0.5353281423, 0.4446345422])
O_SP = [5.033151319, 1.169596125, 0.3803889600]
N_S = ([99.10616896, 18.05231239, 4.885660238], [0.1543289673, 0.5353281423, 0.4446345422])
N_SP = [3.780455879, 0.8784966449, 0.2857143744]
SP_S = [-0.09996722919, 0.3995128261, 0.7001154689]
SP_P = [0.1559162750, 0.6076837186, 0.3919573931]
H_631G = [([18.73113696, 2.825394365, 0.6401216923], [0.03349460434, 0.2347269535, 0.8137573261]),
          ([0.1612777588], [1.0])]
HE_631G = [([38.421634, 5.77803, 1.241774], [0.04013973935, 0.261246097, 0.7931846246]),
           ([0.297964], [1.0])]
def shell(center, l, exps, coefs):
    return {"center": center, "l": l, "exponents": list(exps), "coefficients": list(coefs)}
def heavy_shells(center, core, sp):
    return [shell(center, 0, *core), shell(center, 0, sp, SP_S), shell(center, 1, sp, SP_P)]
def raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
"""
    return [
        # --- Normal: NH3 / STO-3G pyramid, scaled-opposite-spin parametrisation ---
        {
            "setup": basis + """
coords = np.array([[0.0, 0.0, 0.2210],
                   [1.7720, 0.0, -0.5157],
                   [-0.8860, 1.5346, -0.5157],
                   [-0.8860, -1.5346, -0.5157]])
charges = np.array([7.0, 1.0, 1.0, 1.0])
shells = heavy_shells(0, N_S, N_SP) + [shell(k, 0, *H_STO3G) for k in (1, 2, 3)]
""",
            "call": "compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'sos')",
            "gold_call": "_oracle_compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'sos')",
            "tol": 1e-7,
        },
        # --- Normal: HeH+ / 6-31G cation (charge = +1), unscaled parametrisation ---
        {
            "setup": basis + """
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]])
charges = np.array([2.0, 1.0])
shells = [shell(0, 0, *HE_631G[0]), shell(0, 0, *HE_631G[1]),
          shell(1, 0, *H_631G[0]), shell(1, 0, *H_631G[1])]
""",
            "call": "compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'plain', 1)",
            "gold_call": "_oracle_compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'plain', 1)",
            "tol": 1e-7,
        },
        # --- Normal: distorted H2O / STO-3G (no symmetry), spin-component-scaled parametrisation ---
        {
            "setup": basis + """
coords = np.array([[0.05, -0.12, 0.03],
                   [1.52, 1.20, -0.10],
                   [-1.75, 0.95, 0.20]])
charges = np.array([8.0, 1.0, 1.0])
shells = heavy_shells(0, O_S, O_SP) + [shell(1, 0, *H_STO3G), shell(2, 0, *H_STO3G)]
""",
            "call": "compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'scs')",
            "gold_call": "_oracle_compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'scs')",
            "tol": 1e-7,
        },
        # --- Boundary: H2 / STO-3G, minimal two-orbital system, spin-component-scaled ---
        {
            "setup": basis + """
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
charges = np.array([1.0, 1.0])
shells = [shell(0, 0, *H_STO3G), shell(1, 0, *H_STO3G)]
""",
            "call": "compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'scs', 0)",
            "gold_call": "_oracle_compute_homo_quasiparticle_energy(coords.copy(), charges.copy(), copy.deepcopy(shells), 'scs', 0)",
            "tol": 1e-7,
        },
        # --- Invalid: unknown parametrisation name ---
        {
            "setup": basis + """
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
charges = np.array([1.0, 1.0])
shells = [shell(0, 0, *H_STO3G), shell(1, 0, *H_STO3G)]
""",
            "call": "raises_value_error(compute_homo_quasiparticle_energy, coords.copy(), charges.copy(), copy.deepcopy(shells), 'mp2')",
            "gold_call": "raises_value_error(_oracle_compute_homo_quasiparticle_energy, coords.copy(), charges.copy(), copy.deepcopy(shells), 'mp2')",
        },
        # --- Invalid: odd electron count (H2+ cannot have a closed-shell reference) ---
        {
            "setup": basis + """
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
charges = np.array([1.0, 1.0])
shells = [shell(0, 0, *H_STO3G), shell(1, 0, *H_STO3G)]
""",
            "call": "raises_value_error(compute_homo_quasiparticle_energy, coords.copy(), charges.copy(), copy.deepcopy(shells), 'plain', 1)",
            "gold_call": "raises_value_error(_oracle_compute_homo_quasiparticle_energy, coords.copy(), charges.copy(), copy.deepcopy(shells), 'plain', 1)",
        },
    ]
