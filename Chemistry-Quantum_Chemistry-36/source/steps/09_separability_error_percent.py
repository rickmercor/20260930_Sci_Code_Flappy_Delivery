"""
Relative separability error of the CCSD excitonic coupling: the fragment coupling measured against the supersystem coupling.

In an exact theory the excitonic coupling of two non-overlapping molecules is fully determined by properties of the isolated molecules, and the coupling obtained from the transition densities of the fragments agrees with the one obtained from the splitting of the dimer. For a coupled-cluster model truncated at a fixed excitation rank this separability is not guaranteed: the dimer's excitation manifold is truncated relative to the dimer reference, while the fragment quantities carry the full content of each molecule's own truncated description, and the two routes need not agree even when the molecules are far apart.

The size of the disagreement is expressed as the relative separability error

  Delta = 100 * ( |V^c| - |V^s| ) / |V^s|

in percent, with V^c the fragment coupling from the EOM-CCSD transition densities of the monomer and V^s the supersystem coupling from the first-order EOM-CCSD splitting of the dimer, both for the same monomer state and the same antiparallel stack. A negative value means that the fragment route underestimates the coupling that the dimer calculation produces.

Returns
-------
float: the relative separability error Delta of the CCSD excitonic coupling, in percent
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def separability_error_percent(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                               hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                               root: int, separation: float) -> float:
    '''Relative separability error of the CCSD excitonic coupling in percent.

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
    delta : float
        100 * (|V^c| - |V^s|) / |V^s|.

    Raises
    ------
    ValueError
        If any argument violates the constraints of the steps it feeds, or if the
        supersystem coupling vanishes so that the relative error is undefined.
    '''
    return delta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_separability_error_percent(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                       hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                       root: int, separation: float) -> float:
    vc = _oracle_fragment_excitonic_coupling(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation)
    vs = _oracle_supersystem_excitonic_coupling(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation)
    if vs < 1e-12:
        raise ValueError("the supersystem coupling vanishes")
    return float(100.0 * (vc - vs) / vs)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark C=C-N stack at 7 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 7.0)",
            "gold_call": "_oracle_separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 7.0)",
            "tol": 1e-7,
        },
        # --- Normal: the long-range limit of the same stack at 30 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 30.0)",
            "gold_call": "_oracle_separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 30.0)",
            "tol": 1e-7,
        },
        # --- Boundary: a symmetric two-carbon unit ---
        {
            "setup": """import numpy as np
""",
            "call": "separability_error_percent(np.array([1.34]), np.array([-2.4]), np.array([0.0, 0.0]), np.array([11.13, 11.13]), np.array([1.0, 1.0]), 1, 0, 6.0)",
            "gold_call": "_oracle_separability_error_percent(np.array([1.34]), np.array([-2.4]), np.array([0.0, 0.0]), np.array([11.13, 11.13]), np.array([1.0, 1.0]), 1, 0, 6.0)",
            "tol": 1e-7,
        },
        # --- Edge: an allyl cation stack, where the fragment route overshoots ---
        {
            "setup": """import numpy as np
""",
            "call": "separability_error_percent(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13), np.ones(3), 1, 0, 6.0)",
            "gold_call": "_oracle_separability_error_percent(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13), np.ones(3), 1, 0, 6.0)",
            "tol": 1e-7,
        },
        # --- Edge: a strongly donating C=C-N unit at 12 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -1.6]), np.array([0.0, 0.0, -5.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 12.0)",
            "gold_call": "_oracle_separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -1.6]), np.array([0.0, 0.0, -5.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 12.0)",
            "tol": 1e-7,
        },
        # --- Invalid: zero stacking separation ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_separability_error_percent(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 0.0)
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
