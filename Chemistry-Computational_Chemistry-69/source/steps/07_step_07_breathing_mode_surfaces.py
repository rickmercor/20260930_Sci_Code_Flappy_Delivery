"""
Step 07 - Harmonic metal-ligand breathing surfaces of the two acceptor oxidation states.

An octahedral complex whose six metal-ligand bonds all lengthen on reduction reorganises along its totally symmetric breathing coordinate. In that mode every ligand moves radially by the same amount and the metal, sitting on the inversion centre, does not move at all, so the vibrational frequency is set by the metal-ligand stretching force constant and the ligand mass alone. Writing the displacement per bond from the oxidised equilibrium, the vibrational energy stored in the mode is six times the energy of one harmonic bond.

The two oxidation states are harmonic in the same coordinate but with their own force constants and their own equilibrium bond lengths. A low-spin d6 cobalt(III) ammine and a high-spin d7 cobalt(II) ammine differ by almost a factor of two in stiffness, which is large enough that averaging the two force constants into a single effective value is a real approximation rather than a detail.

The useful summary of the two surfaces is a pair of vertical energies. One is how far the reduced-state surface lies above its own minimum when evaluated at the oxidised equilibrium geometry; the other is how far the oxidised-state surface lies above its own minimum at the reduced equilibrium geometry. For equal force constants the two coincide. Here they differ, and which of them governs a given process depends on which state the nuclei start in.

Returns
-------
numpy.ndarray of shape (4,): two force constants in N/m and two vertical energies in kcal/mol
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def breathing_mode_surfaces(breathing: npt.ArrayLike) -> np.ndarray:
    '''Force constants and vertical energies of a metal-ligand breathing mode.

    Parameters
    ----------
    breathing : array_like
        Length-5 sequence [nu_ox, nu_red, d_ox, d_red, m_ligand]: the totally
        symmetric breathing wavenumbers of the oxidised and reduced complexes in
        cm^-1, the metal-ligand bond lengths of the oxidised and reduced
        complexes in angstrom, and the ligand mass in unified atomic mass units.
        The complex is octahedral with six equivalent metal-ligand bonds.

    Returns
    -------
    surfaces : numpy.ndarray
        Array of shape (4,). Element 0 is the metal-ligand force constant of the
        oxidised complex and element 1 that of the reduced complex, both in N/m.
        Element 2 is the energy of the reduced-state surface at the oxidised
        equilibrium geometry, measured from the reduced-state minimum, and
        element 3 the energy of the oxidised-state surface at the reduced
        equilibrium geometry, measured from the oxidised-state minimum, both in
        kcal/mol.

    Raises
    ------
    ValueError
        If breathing does not have exactly five elements, if any element is not
        finite, or if any wavenumber, bond length or mass is not positive.
    '''
    return surfaces

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_breathing_mode_surfaces(breathing: npt.ArrayLike) -> np.ndarray:
    """Harmonic force constants from the breathing wavenumbers, then vertical energies."""
    _AMU_KG = 1.6605390666e-27
    _C_CM_PER_S = 29979245800.0
    _J_PER_KCAL_B = 6.9476954571e-21
    _N_BONDS = 6
    import numpy as np
    b = np.asarray(breathing, dtype=float).ravel()
    if b.size != 5:
        raise ValueError("breathing must have exactly five elements")
    if not np.all(np.isfinite(b)):
        raise ValueError("breathing must be finite")
    if np.any(b <= 0.0):
        raise ValueError("wavenumbers, bond lengths and mass must be positive")
    nu_ox, nu_red, d_ox, d_red, m_lig = b
    mass = m_lig * _AMU_KG
    f_ox = mass * (2.0 * np.pi * _C_CM_PER_S * nu_ox) ** 2
    f_red = mass * (2.0 * np.pi * _C_CM_PER_S * nu_red) ** 2
    shift = (d_red - d_ox) * 1e-10
    half_n = 0.5 * _N_BONDS
    e_red_at_ox = half_n * f_red * shift ** 2 / _J_PER_KCAL_B
    e_ox_at_red = half_n * f_ox * shift ** 2 / _J_PER_KCAL_B
    return np.array([f_ox, f_red, e_red_at_ox, e_ox_at_red], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: hexaamminecobalt(III/II), low-spin to high-spin ---
        {
            "setup": "import numpy as np\nbr = np.array([494.0, 357.0, 1.936, 2.114, 17.031])\n",
            "call": "breathing_mode_surfaces(br)",
            "gold_call": "_oracle_breathing_mode_surfaces(br)",
        },
        # --- Normal: hexaaquairon(III/II), water ligands ---
        {
            "setup": "import numpy as np\nbr = np.array([490.0, 389.0, 1.990, 2.128, 18.015])\n",
            "call": "breathing_mode_surfaces(br)",
            "gold_call": "_oracle_breathing_mode_surfaces(br)",
        },
        # --- Boundary: equal wavenumbers, the two vertical energies must coincide ---
        {
            "setup": "import numpy as np\nbr = [420.0, 420.0, 2.000, 2.100, 17.031]\n",
            "call": "breathing_mode_surfaces(br)",
            "gold_call": "_oracle_breathing_mode_surfaces(br)",
        },
        # --- Edge: hexaammineruthenium(III/II), a tiny bond-length change ---
        {
            "setup": "import numpy as np\nbr = np.array([500.0, 472.0, 2.104, 2.144, 17.031])\n",
            "call": "breathing_mode_surfaces(br)",
            "gold_call": "_oracle_breathing_mode_surfaces(br)",
        },
    ]
