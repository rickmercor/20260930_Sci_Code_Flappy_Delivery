"""
Step 07: Binding-energy errors of SCF and density-corrected LSDA for H2+.

Binding-energy errors of self-consistent and density-corrected local spin-density treatments of a one-dimensional H2+.

For H2+ with unit charges at -R/2 and +R/2, the binding energy of any treatment is its electronic energy of H2+ at
separation R minus its energy of the H atom, the bare proton contributing zero; the repulsion between nuclei is common to
all treatments and cancels from every error. The exact reference uses the lowest eigenvalues E_g(R) of H2+ and E_H of
the atom from step 03. The error of a treatment is its binding energy minus the exact one.

Three treatments are compared. In the self-consistent treatment both energies are the lowest self-consistent
exchange-only LSDA energies of step 05. The density-corrected treatments evaluate the LSDA energy functional
E[phi] = <phi|T + v|phi> + J[phi^2] + E_x[phi^2, 0] on a supplied orbital instead of minimizing it, with the atom described
by its exact ground-state orbital. With the exact delocalized ground-state orbital of H2+ this is density correction by
the exact density. With the orbital of largest probability on the left half-line within the span of the two lowest H2+
states (step 06), it is density correction by a localized density that dissociates to an electron on one atom and a bare
proton on the other.

The expectation value <phi|T + v|phi> uses the same three-point kinetic operator and rectangle rule as step 03, and the
splitting E_u(R) - E_g(R) between the two lowest H2+ states is also reported. All energies are in hartree.

Returns
-------
numpy.ndarray of shape (4,), SCF, exact-orbital and localized-orbital binding-energy errors and the H2+ splitting, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dissociation_errors(separation: float, softening: float, spacing: float, half_width: float) -> "np.ndarray":
    '''Binding-energy errors of SCF LSDA and of LSDA on the exact and on the localized orbital, plus the H2+ splitting.

    Parameters
    ----------
    separation : float
        Internuclear separation R > 0 of H2+ in bohr, nuclei at -R/2 and +R/2.
    softening : float
        Softening length b > 0 of all interactions, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x an integer.

    Returns
    -------
    result : np.ndarray
        Shape (4,): the error of the self-consistent LSDA binding energy, the error of LSDA evaluated on the exact H2+
        ground-state orbital, the error of LSDA evaluated on the maximally left-localized orbital, each relative to the
        exact binding energy E_g(R) - E_H, and the splitting E_u(R) - E_g(R), all in hartree.

    Raises
    ------
    ValueError
        If the separation is not strictly positive, or for any invalid grid or softening as in soft_coulomb_states.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _one_electron_expectation(phi: "np.ndarray", centers: "np.ndarray", softening: float, spacing: float, half_width: float) -> float:
    """Finite-difference expectation value of T + v for a normalized grid orbital."""
    import numpy as np
    x = -half_width + spacing * np.arange(phi.size)
    v = np.zeros_like(x)
    for c in centers:
        v -= 1.0 / np.sqrt((x - c) ** 2 + softening ** 2)
    padded = np.concatenate(([0.0], phi, [0.0]))
    kinetic = -0.5 * spacing * float(np.sum(phi * (padded[2:] - 2.0 * phi + padded[:-2]))) / spacing ** 2
    return kinetic + spacing * float(np.sum(v * phi ** 2))


def _oracle_dissociation_errors(separation: float, softening: float, spacing: float, half_width: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if not separation > 0.0:
        raise ValueError("separation must be positive")
    atom = np.array([0.0])
    ion = np.array([-0.5 * separation, 0.5 * separation])
    s_atom = _oracle_soft_coulomb_states(atom, softening, spacing, half_width, 1)
    s_ion = _oracle_soft_coulomb_states(ion, softening, spacing, half_width, 2)
    e_h, phi_h = s_atom[0, 0], s_atom[0, 1:]
    e_g, e_u = s_ion[0, 0], s_ion[1, 0]
    psi_g, psi_u = s_ion[0, 1:], s_ion[1, 1:]
    zero = np.zeros_like(phi_h)
    exact = e_g - e_h
    scf = _oracle_lsda_ground_state_energy(ion, softening, spacing, half_width) - _oracle_lsda_ground_state_energy(atom, softening, spacing, half_width)
    atom_dc = e_h + _oracle_hartree_exchange_energy(phi_h ** 2, zero, spacing, softening)
    dc_exact = e_g + _oracle_hartree_exchange_energy(psi_g ** 2, zero, spacing, softening) - atom_dc
    phi_loc = _oracle_maximally_localized_orbital(psi_g, psi_u, spacing, half_width)
    loc_energy = _one_electron_expectation(phi_loc, ion, softening, spacing, half_width)
    dc_loc = loc_energy + _oracle_hartree_exchange_energy(phi_loc ** 2, zero, spacing, softening) - atom_dc
    return np.array([scf - exact, dc_exact - exact, dc_loc - exact, e_u - e_g])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: near-equilibrium H2+, where the localized density is strongly penalized ---
        {
            "setup": "import numpy as np\n",
            "call": "dissociation_errors(2.0, 1.0, 0.1, 20.0)",
            "gold_call": "_oracle_dissociation_errors(2.0, 1.0, 0.1, 20.0)",
            "tol": 1e-8,
        },
        # --- Normal: intermediate separation where the three errors are comparable ---
        {
            "setup": "import numpy as np\n",
            "call": "dissociation_errors(5.0, 1.0, 0.1, 20.0)",
            "gold_call": "_oracle_dissociation_errors(5.0, 1.0, 0.1, 20.0)",
            "tol": 1e-8,
        },
        # --- Edge: stretched H2+ approaching the fractional-charge limit, with a larger box ---
        {
            "setup": "import numpy as np\n",
            "call": "dissociation_errors(8.5, 1.0, 0.1, 24.0)",
            "gold_call": "_oracle_dissociation_errors(8.5, 1.0, 0.1, 24.0)",
            "tol": 1e-8,
        },
        # --- Normal: a shorter softening changes the self-interaction balance ---
        {
            "setup": "import numpy as np\n",
            "call": "dissociation_errors(4.0, 0.75, 0.08, 16.0)",
            "gold_call": "_oracle_dissociation_errors(4.0, 0.75, 0.08, 16.0)",
            "tol": 1e-8,
        },
        # --- Error: a non-positive separation must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.0, 1.0, 0.1, 10.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(dissociation_errors)",
            "gold_call": "_probe(_oracle_dissociation_errors)",
        },
    ]
