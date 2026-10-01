"""
Add the classical harmonic free energy of the stiff modes to the potential energy of a configuration to obtain the coarse-grained free energy at a given temperature.

Eliminating the fast degrees of freedom from the partition function does not leave behind a potential; it leaves behind a free energy, which depends on temperature and carries an entropy. Writing the all-atom configuration integral as an integral over the reduced manifold, the quantity that appears in the Boltzmann factor is the potential energy at the point on the manifold plus the free energy of the harmonic fluctuations about it in the eliminated directions. That sum is what plays the role of the coarse-grained potential, and it is what must be compared between the frozen and the relaxed treatments.

In the classical, high-temperature regime, where the level spacing of a stiff mode is small compared with the thermal energy, the harmonic contribution of each mode reduces to the thermal energy times the logarithm of the ratio of the level spacing to the thermal energy. Summed over the retained modes, that gives a correction which is negative for soft modes and positive for stiff ones. In the quantum regime the same mode instead contributes its zero-point energy plus a thermal occupation term, the latter being the thermal energy times the logarithm of one minus the Boltzmann factor of the level spacing.

The two regimes behave very differently for a molecule as stiff as water, and the contrast is what this step exists to expose. Both forms require the mode frequency in absolute units, which means converting an eigenvalue expressed in the working units of this calculation, energy in kcal/mol, length in angstrom and mass in unified atomic mass units, into radians per second before any comparison with the thermal energy can be made. That conversion is the one place in the whole pipeline where the physical constants genuinely matter, and it is easy to get wrong by an enormous factor.

Once it is done the two regimes separate sharply. In the classical form the reduced Planck constant enters only as a multiplicative constant inside a logarithm, so it cancels identically from any difference of free energies taken at the same temperature over the same number of modes; the classical branch is therefore insensitive to the conversion. In the quantum form it does not cancel, because the zero-point term is linear rather than logarithmic in the frequency. At 250 K the O-H stretches have beta*hbar*omega about 22 and negligible occupations, but the bends have beta*hbar*omega about 9.1 and contribute about -4.52e-5 kcal/mol to the frozen-minus-relaxed difference, so the full occupation term is retained.

Returns
-------
float: the coarse-grained free energy of the configuration in kcal/mol in the selected vibrational regime, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_cg_free_energy(potential_energy: float, frequencies: np.ndarray,
                           temperature: float = 250.0,
                           quantum: bool = False) -> float:
    """Return the coarse-grained free energy of a configuration.

    Parameters
    ----------
    potential_energy : float
        All-atom potential energy of the configuration in kcal/mol.
    frequencies : np.ndarray
        Array of shape (n_monomers, n_stiff) holding the stiff angular
        frequencies in (kcal/(mol angstrom**2 u))**(1/2). All entries must be
        strictly positive.
    temperature : float
        Absolute temperature in kelvin (temperature > 0).
    quantum : bool
        When False, charge each mode the classical harmonic free energy; when
        True, charge the quantum harmonic free energy instead.

    Returns
    -------
    free_energy : float
        Coarse-grained free energy in kcal/mol, as a native Python float,
        equal to the potential energy plus the harmonic free energy of every
        retained stiff mode in the selected regime.

    Raises
    ------
    ValueError
        If ``potential_energy`` is non-finite; if ``frequencies`` has the
        wrong shape or any non-finite or non-positive entry; if ``temperature``
        is non-finite or non-positive; or if ``quantum`` is not boolean.
    """
    return free_energy  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_cg_free_energy(potential_energy: float, frequencies: np.ndarray,
                                   temperature: float = 250.0,
                                   quantum: bool = False) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(potential_energy, (int, float, np.floating, np.integer))
            and not isinstance(potential_energy, bool) and np.isfinite(potential_energy)):
        raise ValueError("potential_energy must be a finite number")
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 2 or frequencies.size < 1:
        raise ValueError("frequencies must have shape (n_monomers, n_stiff)")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be finite and strictly positive")
    if not (isinstance(temperature, (int, float, np.floating, np.integer))
            and not isinstance(temperature, bool) and np.isfinite(temperature)
            and float(temperature) > 0.0):
        raise ValueError("temperature must be a finite number > 0")
    if not isinstance(quantum, (bool, np.bool_)):
        raise ValueError("quantum must be a boolean")

    temperature = float(temperature)

    # Physical constants (CODATA 2018) and the unit conversions implied by
    # energies in kcal/mol, lengths in angstrom and masses in u.
    avogadro = 6.02214076e23
    kcal_per_mole_in_joule = 4184.0 / avogadro
    atomic_mass_unit = 1.66053906660e-27
    angstrom = 1.0e-10
    hbar = 1.054571817e-34
    boltzmann = 1.380649e-23

    # A frequency of one in working units, expressed in radians per second.
    frequency_unit = np.sqrt(kcal_per_mole_in_joule / (angstrom ** 2 * atomic_mass_unit))
    thermal_energy_joule = boltzmann * temperature
    thermal_energy_kcal = thermal_energy_joule / kcal_per_mole_in_joule

    angular = frequencies.ravel() * frequency_unit
    reduced = hbar * angular / thermal_energy_joule

    if bool(quantum):
        # Zero-point energy plus the free energy of the thermal occupation.
        zero_point = float(np.sum(0.5 * hbar * angular)) / kcal_per_mole_in_joule
        occupation = thermal_energy_kcal * float(np.sum(np.log1p(-np.exp(-reduced))))
        harmonic = zero_point + occupation
    else:
        harmonic = thermal_energy_kcal * float(np.sum(np.log(reduced)))

    return float(potential_energy) + harmonic

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: tetramer-like frequencies at the working temperature ---
        {
            "setup": """import numpy as np
potential_energy = -25.303092644
frequencies = np.array([[35.996882, 35.194210, 14.624828],
                        [35.997096, 35.195142, 14.625408],
                        [35.998083, 35.197364, 14.626857],
                        [35.997313, 35.191385, 14.629107]])
""",
            "call": "compute_cg_free_energy(potential_energy, frequencies, 250.0)",
            "gold_call": "_oracle_compute_cg_free_energy(potential_energy, frequencies, 250.0)",
        },
        # --- Valid: the same configuration in the quantum regime, retaining
        #     the small occupation term alongside the dominant zero-point term ---
        {
            "setup": """import numpy as np
potential_energy = -25.303092644
frequencies = np.array([[35.996882, 35.194210, 14.624828],
                        [35.997096, 35.195142, 14.625408],
                        [35.998083, 35.197364, 14.626857],
                        [35.997313, 35.191385, 14.629107]])
""",
            "call": "compute_cg_free_energy(potential_energy, frequencies, 250.0, True)",
            "gold_call": "_oracle_compute_cg_free_energy(potential_energy, frequencies, 250.0, True)",
        },
        # --- Valid: a single monomer at a higher temperature ---
        {
            "setup": """import numpy as np
potential_energy = 0.0
frequencies = np.array([[36.0, 35.2, 14.6]])
""",
            "call": "compute_cg_free_energy(potential_energy, frequencies, 400.0)",
            "gold_call": "_oracle_compute_cg_free_energy(potential_energy, frequencies, 400.0)",
        },
        # --- Boundary: the classical limit, where a very soft mode at high
        #     temperature makes the quantum and classical forms converge ---
        {
            "setup": """import numpy as np
potential_energy = 0.0
frequencies = np.array([[1.0e-4, 2.0e-4, 3.0e-4]])
""",
            "call": "compute_cg_free_energy(potential_energy, frequencies, 800.0, True)",
            "gold_call": "_oracle_compute_cg_free_energy(potential_energy, frequencies, 800.0, True)",
        },
        # --- Boundary: a very soft mode, where the logarithm turns negative
        #     and the harmonic term lowers the free energy ---
        {
            "setup": """import numpy as np
potential_energy = -3.5
frequencies = np.array([[1.0e-3, 2.0, 30.0]])
""",
            "call": "compute_cg_free_energy(potential_energy, frequencies, 250.0)",
            "gold_call": "_oracle_compute_cg_free_energy(potential_energy, frequencies, 250.0)",
        },
        # --- Edge: very low temperature, where the classical form is far from
        #     the quantum one and the correction dominates the energy ---
        {
            "setup": """import numpy as np
potential_energy = -25.0
frequencies = np.array([[36.0, 35.2, 14.6], [36.1, 35.3, 14.7]])
""",
            "call": "compute_cg_free_energy(potential_energy, frequencies, 5.0)",
            "gold_call": "_oracle_compute_cg_free_energy(potential_energy, frequencies, 5.0)",
        },
        # --- Invalid: a non-positive frequency ---
        {
            "setup": """import numpy as np
frequencies = np.array([[36.0, 0.0, 14.6]])
def run_model():
    try:
        compute_cg_free_energy(-25.0, frequencies, 250.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cg_free_energy(-25.0, frequencies, 250.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive temperature ---
        {
            "setup": """import numpy as np
frequencies = np.array([[36.0, 35.2, 14.6]])
def run_model():
    try:
        compute_cg_free_energy(-25.0, frequencies, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_cg_free_energy(-25.0, frequencies, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
