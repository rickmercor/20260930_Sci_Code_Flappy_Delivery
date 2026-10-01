"""
Step 08 - Predominantly single-excitation singlet and triplet states of a molecule.

This step joins the two halves of the problem. For a given cluster, basis and pair (A0, B0), take the self-consistently dressed orbitals and express everything in them: the zeroth-order energies are the dressed orbital energies, the Fock matrix is the unmodified Hartree-Fock Fock operator written in the dressed orbitals (diagonal only in the canonical ones), and the antisymmetrised integrals follow from the basis-function integrals. Spin orbitals are ordered with the occupied ones first and alpha and beta alternating, p = 2k + s. The strict third-order secular matrix is then diagonalised in both spin sectors.

Many roots of that matrix describe double excitations. A state counts as predominantly a single excitation when its single-excitation weight exceeds 0.5. The step returns, separately for the singlet and for the triplet sector, the n_states lowest such states.

Returns
-------
numpy.ndarray of shape (2, n_states): excitation energies in hartree of the lowest predominantly single-excitation singlets (row 0) and triplets (row 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def spin_resolved_excitations(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list, coefficients: list, n_electrons: int, A0: float, B0: float, n_states: int) -> np.ndarray:
    '''Lowest predominantly single-excitation singlet and triplet states at the regularised third order.

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
    n_states : int
        Number of states wanted in each spin sector, a positive integer.

    Returns
    -------
    states : numpy.ndarray
        Array of shape (2, n_states) in hartree. Row 0 holds the excitation
        energies of the n_states lowest singlets whose single-excitation
        weight exceeds 0.5, row 1 the same for triplets, each in ascending
        order.

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
        within 2000 cycles. Also if n_states is not a positive integer or
        either sector holds fewer than n_states predominantly
        single-excitation states.
    '''
    return states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _dressed_spin_orbital_hamiltonian(coords, charges, exponents, coefficients, n_electrons, A0, B0):
    """Zeroth-order energies, true Fock matrix and <pq||rs> in the converged dressed spin-orbital basis."""
    import numpy as np
    SH = _oracle_one_electron_integrals(coords, charges, exponents, coefficients)
    ref = _oracle_rhf_canonical_orbitals(coords, charges, exponents, coefficients, n_electrons)
    dressed = _oracle_dressed_orbitals(coords, charges, exponents, coefficients, n_electrons, A0, B0)
    eri = _oracle_electron_repulsion_integrals(coords, exponents, coefficients)
    Cd = dressed[1:]
    R = ref[1:].T @ SH[0] @ Cd
    f_spatial = R.T @ np.diag(ref[0]) @ R
    eri_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, Cd, Cd, Cd, Cd, optimize=True)
    g = _spin_orbital_antisymmetrized(eri_mo)
    return np.repeat(dressed[0], 2), np.kron(f_spatial, np.eye(2)), g, 2 * (int(n_electrons) // 2)


def _oracle_spin_resolved_excitations(coords: npt.ArrayLike, charges: npt.ArrayLike, exponents: list,
                                      coefficients: list, n_electrons: int, A0: float, B0: float,
                                      n_states: int) -> np.ndarray:
    import numpy as np
    ns = float(n_states)
    if ns != int(ns) or int(ns) < 1:
        raise ValueError("n_states must be a positive integer")
    ns = int(ns)
    e, f, g, no = _dressed_spin_orbital_hamiltonian(coords, charges, exponents, coefficients, n_electrons, A0, B0)
    out = np.zeros((2, ns))
    for row, parity in enumerate((1, -1)):
        spectrum = _oracle_adc3_spin_sector_spectrum(e, f, g, no, parity)
        picked = spectrum[0][spectrum[1] > 0.5]
        if picked.size < ns:
            raise ValueError("fewer predominantly single-excitation states than requested")
        out[row] = picked[:ns]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: target cluster, three states per spin sector ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.60, 0.00, 0.00], [1.90, 2.40, 0.20], [0.20, 2.50, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = -0.75
B0 = 0.5
n_states = 3
""",
            "call": "spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "gold_call": "_oracle_spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "tol": 1e-07,
        },
        # --- Boundary: Møller-Plesset limit A0 = B0 = 0 for a two-electron cation ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.62, 0.00, 0.00], [0.70, 1.52, 0.05]])
charges = np.array([1.0, 1.0, 1.0])
n_electrons = 2
exponents = [np.array([18.7311370, 2.8253937, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
A0 = 0.0
B0 = 0.0
n_states = 2
""",
            "call": "spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "gold_call": "_oracle_spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "tol": 1e-07,
        },
        # --- Edge: minimal basis with one single excitation per spin ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [0.00, 0.00, 1.46]])
charges = np.array([2.0, 1.0])
n_electrons = 2
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
A0 = -0.75
B0 = 0.5
n_states = 1
""",
            "call": "spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "gold_call": "_oracle_spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "tol": 1e-07,
        },
        # --- Normal: zigzag chain with occupied-only regularisation ---
        {
            "setup": """import numpy as np
coords = np.array([[0.00, 0.00, 0.00], [1.50, 0.00, 0.00], [2.90, 1.20, 0.00], [4.60, 0.90, 0.00]])
charges = np.array([1.0, 1.0, 1.0, 1.0])
n_electrons = 4
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
A0 = 3.8
B0 = 0.0
n_states = 1
""",
            "call": "spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "gold_call": "_oracle_spin_resolved_excitations(np.array(coords), np.array(charges), [np.array(x) for x in exponents], [np.array(x) for x in coefficients], n_electrons, A0, B0, n_states)",
            "tol": 1e-07,
        },
    ]
