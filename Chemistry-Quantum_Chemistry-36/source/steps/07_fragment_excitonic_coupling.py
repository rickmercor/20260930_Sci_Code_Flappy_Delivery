"""
Excitonic coupling of the antiparallel stacked homodimer assembled from the EOM-CCSD transition densities of the isolated molecules.

When two molecules are far enough apart that their pi systems do not overlap, the leading interaction between the state with the excitation on A and the state with the excitation on B is the Coulomb interaction between the transition density of A and the transition density of B. In the zero-differential-overlap picture the transition densities are site populations, the intermolecular operator is sum_{mu in A, nu in B} gamma_AB[mu, nu] (n_mu - Z_mu)(n_nu - Z_nu), and no exchange contribution arises between the molecules.

In a fragment approach the coupling is assembled from quantities of the isolated molecules only. Molecule A supplies the transition density for its excitation going down, rho^{10}, and molecule B the transition density for its excitation going up, rho^{01}, both taken from the EOM-CCSD treatment of the monomer. B is a copy of A, so atom nu of B carries the same transition density as atom nu of A. The fragment coupling is the Coulomb contraction of the two:

  V^c = sum_{mu, nu} gamma_AB[mu, nu] rho^{10}_mu rho^{01}_nu

with gamma_AB the intermolecular Ohno matrix of the antiparallel stack. The sign of V^c depends only on how the transition densities of the two copies are phased against each other, which is arbitrary, so the magnitude |V^c| in eV is reported.

Returns
-------
float: |V^c|, the magnitude of the fragment excitonic coupling in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fragment_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                root: int, separation: float) -> float:
    '''Magnitude of the excitonic coupling built from monomer EOM-CCSD transition densities.

    Parameters
    ----------
    bonds : numpy.ndarray
        Bond lengths of the chain in angstrom, length n_sites - 1.
    betas : numpy.ndarray
        Resonance integral of each bond in eV.
    alphas : numpy.ndarray
        Site energies in eV, length n_sites.
    hubbard_u : numpy.ndarray
        One-centre repulsions in eV, length n_sites.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals of one molecule.
    root : int
        Index, counting from zero, of the monomer singlet excited state.
    separation : float
        Stacking distance between the molecular planes in angstrom.

    Returns
    -------
    coupling : float
        |V^c| in eV.

    Raises
    ------
    ValueError
        If any argument violates the constraints of the monomer matrices, the
        antiparallel stack geometry, the RHF, CCSD or EOM-CCSD steps it feeds, or
        if root does not select an available singlet excited state.
    '''
    return coupling

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fragment_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                        hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                        root: int, separation: float) -> float:
    import numpy as np
    M = _oracle_ppp_monomer_matrices(bonds, betas, alphas, hubbard_u)
    G = _oracle_antiparallel_stack_interactions(bonds, hubbard_u, separation)
    P = _oracle_eom_transition_density_product(M[0], M[1], core_charges, n_occ, root)
    return float(abs(np.sum(G * P)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark C=C-N stack at 7 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "fragment_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 7.0)",
            "gold_call": "_oracle_fragment_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 7.0)",
            "tol": 1e-9,
        },
        # --- Normal: the same stack pressed to 4 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "fragment_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 4.0)",
            "gold_call": "_oracle_fragment_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 4.0)",
            "tol": 1e-9,
        },
        # --- Boundary: a two-electron polar unit ---
        {
            "setup": """import numpy as np
""",
            "call": "fragment_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 0, 5.0)",
            "gold_call": "_oracle_fragment_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 0, 5.0)",
            "tol": 1e-9,
        },
        # --- Edge: butadiene bright state, where the monomer itself is not exact at CCSD ---
        {
            "setup": """import numpy as np
""",
            "call": "fragment_excitonic_coupling(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13), np.ones(4), 2, 1, 6.0)",
            "gold_call": "_oracle_fragment_excitonic_coupling(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13), np.ones(4), 2, 1, 6.0)",
            "tol": 1e-9,
        },
        # --- Edge: an allyl cation stack at 6 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "fragment_excitonic_coupling(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13), np.ones(3), 1, 0, 6.0)",
            "gold_call": "_oracle_fragment_excitonic_coupling(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13), np.ones(3), 1, 0, 6.0)",
            "tol": 1e-9,
        },
        # --- Invalid: a negative stacking separation ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        fragment_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 0, -5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_fragment_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 0, -5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
