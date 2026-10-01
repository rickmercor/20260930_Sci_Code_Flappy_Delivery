"""
Run the whole vibronic chain at a given lattice temperature and report the radiative quantum efficiency of the centre as a percentage.

The radiative quantum efficiency is the fraction of all decay events out of the
excited electronic state that emit a photon,

eta_rad = Gamma_R / (Gamma_R + Gamma_NR),

reported here as a percentage. It is the figure of merit that decides whether a
centre stays bright enough to be read out optically, whatever share of its photons
then falls in the zero-phonon line.

Only the nonradiative channel depends on temperature. The total radiative rate is
fixed by the mean emitted photon energy, and for two equal-curvature wells coupled
linearly that mean stays at E_ZPL - S hbar Omega however the excited-state ladder is
populated, so the phonon-corrected radiative rate keeps its zero-temperature value
at every temperature.

This step assembles the chain: the mass-weighted coordinate offset from the
displaced cluster, the configuration coordinate parameters from the three total
energies, the coupling strength, the electronic dipole rate and its phonon-corrected
total, and the thermally averaged multiphonon rate at the requested temperature.

Returns
-------
float, the radiative quantum efficiency of the centre at the given lattice temperature, as a percentage.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radiative_efficiency(masses: np.ndarray, displacements: np.ndarray,
                         e_ground_relaxed: float, e_excited_relaxed: float,
                         e_ground_at_excited: float, dipole_debye: float,
                         refractive_index: float, field_ratio: float,
                         coupling: float, degeneracy: float,
                         temperature: float) -> float:
    '''Radiative quantum efficiency of a colour centre at a lattice temperature, as a percentage.

    Parameters
    ----------
    masses : numpy.ndarray, shape (N,)
        Atomic masses of the moving atoms in amu.
    displacements : numpy.ndarray, shape (N, 3)
        Displacement of each atom in Angstrom, relaxed excited-state position
        minus relaxed ground-state position.
    e_ground_relaxed : float
        Total energy of the relaxed ground state in eV.
    e_excited_relaxed : float
        Total energy of the relaxed excited state in eV.
    e_ground_at_excited : float
        Total energy of the ground-state electronic configuration at the
        relaxed excited-state geometry, in eV.
    dipole_debye : float
        Transition dipole moment in debye.
    refractive_index : float
        Refractive index of the host at the emission wavelength.
    field_ratio : float
        Ratio of the effective field at the defect to the bulk field.
    coupling : float
        Electron-phonon coupling matrix element in eV amu^-1/2 Angstrom^-1.
    degeneracy : float
        Configurational degeneracy of the point defect.
    temperature : float
        Lattice temperature in K. Must be non-negative.

    Returns
    -------
    eta_rad : float
        The radiative quantum efficiency as a percentage. Raises ValueError
        whenever any stage of the chain receives invalid input, including a
        negative temperature.
    '''
    return eta_rad

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_radiative_efficiency(masses: np.ndarray, displacements: np.ndarray,
                                 e_ground_relaxed: float, e_excited_relaxed: float,
                                 e_ground_at_excited: float, dipole_debye: float,
                                 refractive_index: float, field_ratio: float,
                                 coupling: float, degeneracy: float,
                                 temperature: float) -> float:
    delta_q = _oracle_mass_weighted_displacement(masses, displacements)
    e_zpl, hbar_omega, _e_rel = _oracle_configuration_coordinate_parameters(
        e_ground_relaxed, e_excited_relaxed, e_ground_at_excited, delta_q)
    huang_rhys, _debye_waller = _oracle_electron_phonon_coupling(delta_q, hbar_omega)
    gamma_r0 = _oracle_dipole_transition_rate(dipole_debye, refractive_index,
                                              e_zpl, field_ratio)
    gamma_r, _gamma_zpl, _gamma_psb = _oracle_radiative_rate_partition(
        gamma_r0, huang_rhys, hbar_omega, e_zpl)
    gamma_nr = _oracle_multiphonon_rate(delta_q, huang_rhys, hbar_omega, e_zpl,
                                        coupling, degeneracy, temperature)
    return float(100.0 * gamma_r / (gamma_r + gamma_nr))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    from textwrap import dedent
    centre = dedent("""import numpy as np
    M = np.array([121.760, 28.0855, 28.0855, 28.0855, 12.011, 12.011, 12.011,
                  12.011, 12.011, 12.011, 12.011, 12.011, 12.011])
    D = np.array([[ 0.0258, -0.0179,  0.0409], [-0.0394,  0.0280, -0.0320],
                  [ 0.0318, -0.0421, -0.0289], [ 0.0093,  0.0370,  0.0438],
                  [ 0.0546, -0.0258,  0.0211], [-0.0471,  0.0393, -0.0179],
                  [ 0.0248,  0.0511, -0.0344], [-0.0180, -0.0458,  0.0401],
                  [ 0.0382,  0.0222,  0.0499], [-0.0526, -0.0169, -0.0256],
                  [ 0.0154, -0.0554,  0.0196], [-0.0324,  0.0234,  0.0453],
                  [ 0.0430, -0.0346, -0.0185]])
    EG, EE, EV_ = -8912.475160, -8911.713724, -8912.352817
    """).replace("\n    ", "\n")
    return [
        {   # boundary: the task centre at zero temperature
            "setup": centre,
            "call": "radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.0)",
            "gold_call": "_oracle_radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.0)",
        },
        {   # normal: the task centre at 100 K, where the first excited levels already matter
            "setup": centre,
            "call": "radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 100.0)",
            "gold_call": "_oracle_radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 100.0)",
        },
        {   # normal: the task centre at 200 K, well into thermal quenching
            "setup": centre,
            "call": "radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 200.0)",
            "gold_call": "_oracle_radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 200.0)",
        },
        {   # normal: the same geometry in vacuum with a singly degenerate defect at room temperature
            "setup": centre,
            "call": "radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 1.0, 1.0, 0.128, 1.0, 295.0)",
            "gold_call": "_oracle_radiative_efficiency(M.copy(), D.copy(), EG, EE, EV_, 2.85, 1.0, 1.0, 0.128, 1.0, 295.0)",
        },
        {   # edge: a soft, heavy four-atom centre at 500 K, where about a hundred levels are populated
            "setup": """import numpy as np
M = np.array([121.760, 28.0855, 12.011, 12.011])
D = np.array([[0.061, -0.052, 0.047], [-0.083, 0.071, -0.066], [0.079, -0.058, 0.074], [-0.069, 0.088, -0.057]])
""",
            "call": "radiative_efficiency(M.copy(), D.copy(), -900.0, -899.3, -899.92, 4.0, 2.6, 1.0, 0.1, 2.0, 500.0)",
            "gold_call": "_oracle_radiative_efficiency(M.copy(), D.copy(), -900.0, -899.3, -899.92, 4.0, 2.6, 1.0, 0.1, 2.0, 500.0)",
        },
        {   # edge: a negative temperature must raise ValueError through the chain
            "setup": centre + """def probe(fn):
    try:
        fn(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, -1.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(radiative_efficiency)",
            "gold_call": "probe(_oracle_radiative_efficiency)",
        },
    ]
