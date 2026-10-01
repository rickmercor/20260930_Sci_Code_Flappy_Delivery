"""
Step 08 - Density-induced change of the normal-incidence LP-UP splitting (orchestrator).

This step assembles the whole calculation. The screened interaction (step 01) and the Wannier solution (step 02) fix the exciton, whose wavefunction sets the exchange constant (step 03) and the blocking overlap (step 04). The cavity problem (step 05) and the thermal population (step 06) set where the polaritons sit in momentum and how excitonic they are, and step 07 turns all of these into the shifts of the two branches at normal incidence.

The observable is their difference, Delta E_UP - Delta E_LP, the change of the splitting between the branches at normal incidence produced by the finite polariton density, positive when the branches move apart. A rigid exciton-exciton interaction would shift both branches alike and leave the splitting unchanged, so this quantity isolates the branch asymmetry that the microscopic treatment predicts. The value is returned in meV, and the ground state of step 02 must be bound.

Returns
-------
float, the density-induced change of the normal-incidence LP-UP splitting in meV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def polariton_splitting_change(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> float:
    '''Density-induced change of the normal-incidence splitting, Delta E_UP - Delta E_LP.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of distinct, finite, positive exponents of the
        Gaussian basis used for the 1s exciton, nm^-2.
    m_e : float
        Electron effective mass in units of m0, > 0.
    m_h : float
        Hole effective mass in units of m0, > 0.
    kappa : float
        Mean dielectric constant of the surrounding media, > 0.
    r0 : float
        Screening length of the sheet in nm, >= 0.
    exciton_energy : float
        Exciton energy at Q = 0 in eV, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV; E_C(0) must be positive.
    cavity_index : float
        Effective refractive index of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.
    temperature : float
        Temperature in K, > 0.
    density : float
        Total polariton density in nm^-2, > 0.

    Returns
    -------
    splitting_change : float
        Delta E_UP - Delta E_LP at normal incidence, in meV.

    Raises
    ------
    ValueError
        If the lowest eigenvalue of step 02 is not negative (no bound 1s state),
        or if any input is invalid as in steps 02 to 07.
    '''
    return splitting_change

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_polariton_splitting_change(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> float:
    """Density-induced change of the normal-incidence LP-UP splitting, meV."""
    import numpy as np
    state = _oracle_exciton_ground_state(exponents, m_e, m_h, kappa, r0)
    if not state[0] < 0.0:
        raise ValueError("the screened electron-hole problem has no bound 1s state in this basis")
    shifts = _oracle_normal_incidence_shifts(exponents, state[1:], m_e, m_h, kappa, r0, exciton_energy,
                                             cavity_detuning, cavity_index, coupling, temperature, density)
    return float(shifts[1] - shifts[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the benchmark cavity and gas with a six-function basis ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.01, 0.05, 0.25, 1.25, 6.25, 31.25])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 50.0\n'
                      'density = 0.01\n'),
            "call": 'polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
        # --- Normal: same system at 40 K, deeper in the light-cone-dominated regime ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.01, 0.05, 0.25, 1.25, 6.25, 31.25])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 40.0\n'
                      'density = 0.01\n'),
            "call": 'polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
        # --- Boundary: zero detuning at 20 K, where exchange cancels in the splitting and only phase-space filling remains ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.01, 0.05, 0.25, 1.25, 6.25, 31.25])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = 0.0\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 20.0\n'
                      'density = 0.01\n'),
            "call": 'polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
        # --- Edge: unscreened sheet (r0 = 0) with the cavity 20 meV below the exciton at 100 K ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.01, 0.05, 0.25, 1.25, 6.25, 31.25])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 0.0\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = -0.02\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 100.0\n'
                      'density = 0.01\n'),
            "call": 'polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_polariton_splitting_change(exponents, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
    ]
