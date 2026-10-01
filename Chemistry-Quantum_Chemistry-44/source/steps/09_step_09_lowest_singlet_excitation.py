"""
Step 09 - Orchestrator: lowest predominantly single-excitation singlet excitation energy in eV.

The quantity asked for is the vertical excitation energy of the lowest singlet state that is predominantly a single excitation, taken from the strict third-order secular matrix built on the self-consistently dressed orbitals. Every earlier stage feeds it: the integrals, the Hartree-Fock reference, the dressing, the second-order amplitudes, the three blocks of the secular matrix and the spin classification. Report it in electronvolts, with 1 hartree = 27.211386245988 eV.

Returns
-------
float, the lowest predominantly single-excitation singlet excitation energy in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def lowest_singlet_excitation(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int, A0: float, B0: float) -> float:
    '''Lowest predominantly single-excitation singlet excitation energy in electronvolts.

    Parameters
    ----------
    coords : array_like
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : array_like
        Nuclear charges, one positive value per atom.
    exponents : list of array_like
        One entry per contracted s shell, carried identically by every atom:
        the primitive Gaussian exponents of that shell in bohr^-2.
    coefficients : list of array_like
        Contraction coefficients, matching exponents entry by entry. They
        multiply normalised primitives, and the contracted function is then
        normalised to unit self-overlap.
    n_electrons : int
        Number of electrons, a positive even integer.
    A0 : float
        Regularisation parameter of the occupied block of W.
    B0 : float
        Regularisation parameter of the virtual block of W.

    Returns
    -------
    excitation_ev : float
        Excitation energy in eV of the lowest singlet whose single-excitation
        weight exceeds 0.5, with 1 hartree = 27.211386245988 eV.

    Raises
    ------
    ValueError
        If coords is not a finite array of shape (n_atoms, 3), two nuclei
        coincide, charges does not hold one positive finite value per atom,
        the basis description is malformed (empty, mismatched lengths,
        non-positive or non-finite exponents, non-finite or all-zero
        coefficients), n_electrons is not a positive even integer or leaves
        no virtual orbital, the smallest eigenvalue of the overlap matrix is
        below 1e-10, the SCF iterations do not converge within 1000 cycles,
        A0 or B0 is not finite, the dressed occupied and virtual levels
        overlap during the iterations, or the dressing does not converge
        within 2000 cycles. Also if either spin sector has no state whose
        single-excitation weight exceeds 0.5.
    '''
    return excitation_ev

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_lowest_singlet_excitation(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                      coefficients: list, n_electrons: int, A0: float, B0: float) -> float:
    import numpy as np
    states = _oracle_spin_resolved_excitations(coords, charges, exponents, coefficients, n_electrons, A0, B0, 1)
    return float(states[0, 0] * 27.211386245988)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target cluster with the third-order regularisation parameters ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.60, 0.00, 0.00], [1.90, 2.40, 0.20], [0.20, 2.50, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = -0.75
B0 = 0.5
""",
            "call": "lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-06,
        },
        # --- Boundary: target cluster in the Møller-Plesset limit ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.60, 0.00, 0.00], [1.90, 2.40, 0.20], [0.20, 2.50, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = 0.0
B0 = 0.0
""",
            "call": "lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-06,
        },
        # --- Edge: two-electron cation with occupied-only regularisation ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.62, 0.00, 0.00], [0.70, 1.52, 0.05]])
charges = np.array([1.0, 1.0, 1.0])
n_electrons = 2
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = 3.8
B0 = 0.0
""",
            "call": "lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-06,
        },
        # --- Normal: six electrons on a puckered ring ---
        {
            "setup": """import numpy as np
coords = np.array([[1.90, 0.00, 0.00], [1.10, 1.65, 0.10], [-0.95, 1.65, 0.20],
                   [-1.75, 0.00, 0.30], [-0.95, -1.65, 0.40], [1.10, -1.65, 0.50]])
charges = np.ones(6)
n_electrons = 6
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
A0 = -0.75
B0 = 0.5
""",
            "call": "lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "gold_call": "_oracle_lowest_singlet_excitation(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0)",
            "tol": 1e-06,
        },
    ]
