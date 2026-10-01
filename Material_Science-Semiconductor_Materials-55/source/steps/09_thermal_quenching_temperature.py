"""
Find the lattice temperature at which the radiative quantum efficiency of the centre has fallen to a given fraction of its zero-temperature value.

A colour centre that looks bright in a cryostat can go dark long before room
temperature, because the multiphonon channel is thermally activated: warming the
lattice populates excited-state vibrational levels from which far fewer ground-state
quanta are needed to absorb the electronic energy. The quantity that decides whether
a centre survives warm operation is therefore the quenching temperature, the lattice
temperature at which the radiative quantum efficiency has dropped to a chosen
fraction of its zero-temperature value. Half is the usual choice in luminescence
work.

The target is fixed by the zero-temperature efficiency of the same centre, and the
quenching temperature is the lowest temperature at which the efficiency has fallen
to that target. Nothing here restricts the answer to room temperature or below: a
stiff, weakly coupled centre can stay bright to several hundred kelvin, where a
large part of the excited-state ladder is populated.

The crossing has no closed form, because temperature enters through the Boltzmann
weights of every excited-state level at once, so it has to be solved for numerically.
Locate it to within 1e-9 K, so that the reported temperature is reproducible.

Returns
-------
float, the lowest lattice temperature in K at which the radiative quantum efficiency falls to the given fraction of its zero-temperature value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thermal_quenching_temperature(masses: np.ndarray, displacements: np.ndarray,
                                  e_ground_relaxed: float, e_excited_relaxed: float,
                                  e_ground_at_excited: float, dipole_debye: float,
                                  refractive_index: float, field_ratio: float,
                                  coupling: float, degeneracy: float,
                                  fraction: float) -> float:
    '''Lowest temperature at which the radiative efficiency falls to a fraction of its zero-temperature value.

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
    fraction : float
        Target efficiency as a fraction of the zero-temperature efficiency.
        Must lie strictly between 0 and 1.

    Returns
    -------
    t_quench : float
        The quenching temperature in K. Raises ValueError on a fraction outside
        (0, 1), when any stage of the chain receives invalid input, or when the
        efficiency does not fall to the target fraction below 3000 K.
    '''
    return t_quench

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thermal_quenching_temperature(masses: np.ndarray, displacements: np.ndarray,
                                          e_ground_relaxed: float, e_excited_relaxed: float,
                                          e_ground_at_excited: float, dipole_debye: float,
                                          refractive_index: float, field_ratio: float,
                                          coupling: float, degeneracy: float,
                                          fraction: float) -> float:
    from scipy.optimize import brentq
    f = float(fraction)
    if not np.isfinite(f) or f <= 0.0 or f >= 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")

    def _eta(t):
        return _oracle_radiative_efficiency(masses, displacements, e_ground_relaxed,
                                            e_excited_relaxed, e_ground_at_excited,
                                            dipole_debye, refractive_index, field_ratio,
                                            coupling, degeneracy, t)

    target = f * _eta(0.0)
    # Walk up in small steps so that the first crossing is the one bracketed, then refine.
    lo, step = 0.0, 5.0
    while lo < 3000.0:
        hi = lo + step
        if _eta(hi) <= target:
            return float(brentq(lambda t: _eta(t) - target, lo, hi, xtol=1e-11, rtol=1e-15))
        lo = hi
    raise ValueError("the efficiency does not fall to the target fraction below 3000 K")

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
        {   # normal: the task centre at the conventional half-efficiency point
            "setup": centre,
            "call": "thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.5)",
            "gold_call": "_oracle_thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.5)",
        },
        {   # boundary: a fraction close to one, the first few percent of quenching
            "setup": centre,
            "call": "thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.97)",
            "gold_call": "_oracle_thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.97)",
        },
        {   # normal: a deep quench to one percent of the cold efficiency
            "setup": centre,
            "call": "thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.01)",
            "gold_call": "_oracle_thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 0.01)",
        },
        {   # edge: an eight-order quench crosses only after the thermally populated ladder becomes broad
            "setup": centre,
            "call": "thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 1.0e-8)",
            "gold_call": "_oracle_thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 1.0e-8)",
        },
        {   # normal: the same geometry with a weaker dipole in vacuum, which quenches sooner
            "setup": centre,
            "call": "thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 1.2, 1.0, 1.0, 0.128, 4.0, 0.5)",
            "gold_call": "_oracle_thermal_quenching_temperature(M.copy(), D.copy(), EG, EE, EV_, 1.2, 1.0, 1.0, 0.128, 4.0, 0.5)",
        },
        {   # edge: a soft, heavy four-atom centre that stays bright far above room temperature
            "setup": """import numpy as np
M = np.array([121.760, 28.0855, 12.011, 12.011])
D = np.array([[0.061, -0.052, 0.047], [-0.083, 0.071, -0.066], [0.079, -0.058, 0.074], [-0.069, 0.088, -0.057]])
""",
            "call": "thermal_quenching_temperature(M.copy(), D.copy(), -900.0, -899.3, -899.92, 4.0, 2.6, 1.0, 0.1, 2.0, 0.5)",
            "gold_call": "_oracle_thermal_quenching_temperature(M.copy(), D.copy(), -900.0, -899.3, -899.92, 4.0, 2.6, 1.0, 0.1, 2.0, 0.5)",
        },
        {   # edge: a fraction of one or more must raise ValueError
            "setup": centre + """def probe(fn):
    try:
        fn(M.copy(), D.copy(), EG, EE, EV_, 2.85, 2.5732, 1.0, 0.128, 4.0, 1.0)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(thermal_quenching_temperature)",
            "gold_call": "probe(_oracle_thermal_quenching_temperature)",
        },
    ]
