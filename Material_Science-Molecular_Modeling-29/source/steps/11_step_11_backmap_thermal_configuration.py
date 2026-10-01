"""
Regenerate a canonical all-atom configuration from a relaxed coarse-grained configuration by adding harmonic thermal displacements along the stiff modes.

A configuration on the reduced manifold has, by construction, no intramolecular fluctuation: every monomer sits at mechanical equilibrium along its stiff coordinates. Any all-atom observable that depends on those coordinates is therefore biased if it is evaluated directly on the manifold. Bond-length and bond-angle distributions collapse to spikes, and any property sensitive to the width of those distributions is wrong by a finite amount that does not diminish with more sampling.

The remedy exploits the fact that the eliminated degrees of freedom were eliminated under a harmonic approximation whose parameters are already in hand. Within that approximation the thermal distribution along each stiff mode is a zero-mean Gaussian whose variance is fixed by the mode frequency and the temperature, so an all-atom configuration consistent with the canonical ensemble is obtained by drawing one independent Gaussian per stiff mode, forming the corresponding displacement in mass-scaled coordinates, converting it back to Cartesian displacements through the inverse square roots of the atomic masses, and adding it to the manifold configuration monomer by monomer.

In the classical regime the variance of a mode is the thermal energy divided by the square of its angular frequency, which is equipartition: each stiff mode carries half the thermal energy on average. The quantum counterpart would replace this by the ground-state spread multiplied by the hyperbolic cotangent of half the reduced level spacing, reducing to the classical result at high temperature and to the zero-point spread at low temperature. The classical form is the one consistent with the classical free energy used elsewhere in this calculation, and pairing a classical free energy with a quantum spread, or the reverse, produces an ensemble that is not the one whose partition function was evaluated.

The reconstruction is essentially free: the mode frequencies and directions were computed anyway to evaluate the free energy, so recovering all-atom statistics costs one Gaussian draw per mode and no additional energy evaluation. This is what allows all-atom bond-length and angle distributions to be accumulated during a coarse-grained simulation. The displacement is applied along the stiff modes only; the rigid-body directions are untouched, because the coarse-grained configuration already describes them explicitly and displacing along them would double-count.

Returns
-------
np.ndarray of shape (n_atoms, 3), float: one canonically distributed all-atom configuration reconstructed from the relaxed configuration, in angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def backmap_thermal_configuration(coords: np.ndarray, frequencies: np.ndarray,
                                  modes: np.ndarray, atom_masses: np.ndarray,
                                  temperature: float = 250.0,
                                  seed: int = 0) -> np.ndarray:
    """Add harmonic thermal displacements to a relaxed configuration.

    Random numbers are drawn from ``np.random.default_rng(seed)``, taking one
    call of size ``n_stiff`` per monomer in increasing monomer order.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the relaxed
        configuration, ordered O, H, H within each monomer.
    frequencies : np.ndarray
        Array of shape (n_monomers, n_stiff) holding the stiff angular
        frequencies in (kcal/(mol angstrom**2 u))**(1/2). All entries must be
        strictly positive.
    modes : np.ndarray
        Array of shape (n_monomers, 9, n_stiff) whose columns are the
        orthonormal mass-scaled mode vectors matching ``frequencies``.
    atom_masses : np.ndarray
        Array of shape (3 * n_monomers,) in unified atomic mass units holding
        the mass of every atom, in the same order as ``coords``. All entries
        must be strictly positive.
    temperature : float
        Absolute temperature in kelvin (temperature > 0).
    seed : int
        Seed of the random generator; the result must be reproducible.

    Returns
    -------
    sampled_coords : np.ndarray
        Array with the same shape as ``coords`` in angstrom, holding one
        all-atom configuration drawn from the classical harmonic ensemble
        about the relaxed configuration.

    Raises
    ------
    ValueError
        If the coordinate, frequency, mode, or mass arrays have incompatible
        shapes or non-finite entries; if a frequency or mass is non-positive;
        if ``temperature`` is non-finite or non-positive; or if ``seed`` is
        not a non-negative integer.
    """
    return sampled_coords  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_backmap_thermal_configuration(coords: np.ndarray, frequencies: np.ndarray,
                                          modes: np.ndarray, atom_masses: np.ndarray,
                                          temperature: float = 250.0,
                                          seed: int = 0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    modes = np.asarray(modes, dtype=float)
    atom_masses = np.asarray(atom_masses, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    n_monomers = coords.shape[0] // 3
    if frequencies.ndim != 2 or frequencies.shape[0] != n_monomers or frequencies.shape[1] < 1:
        raise ValueError("frequencies must have shape (n_monomers, n_stiff)")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be finite and strictly positive")
    n_stiff = frequencies.shape[1]
    if modes.shape != (n_monomers, 9, n_stiff):
        raise ValueError("modes must have shape (n_monomers, 9, n_stiff)")
    if not np.all(np.isfinite(modes)):
        raise ValueError("modes must contain only finite entries")
    if atom_masses.ndim != 1 or atom_masses.size != coords.shape[0]:
        raise ValueError("atom_masses must have one entry per atom")
    if not np.all(np.isfinite(atom_masses)) or np.any(atom_masses <= 0.0):
        raise ValueError("atom_masses must be finite and strictly positive")
    if not (isinstance(temperature, (int, float, np.floating, np.integer))
            and not isinstance(temperature, bool) and np.isfinite(temperature)
            and float(temperature) > 0.0):
        raise ValueError("temperature must be a finite number > 0")
    if not (isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and int(seed) >= 0):
        raise ValueError("seed must be a non-negative integer")

    temperature = float(temperature)
    boltzmann_kcal = 0.0019872042586408316  # kcal/(mol K), from CODATA 2018
    thermal_energy = boltzmann_kcal * temperature

    generator = np.random.default_rng(int(seed))
    sampled_coords = coords.copy()

    for mono in range(n_monomers):
        rows = np.arange(3 * mono, 3 * mono + 3)
        # Classical equipartition variance of each stiff mode.
        deviations = np.sqrt(thermal_energy) / frequencies[mono]
        amplitudes = generator.normal(size=n_stiff) * deviations
        scaled_displacement = modes[mono] @ amplitudes
        inverse_root = 1.0 / np.sqrt(np.repeat(atom_masses[rows], 3))
        sampled_coords[rows] += (inverse_root * scaled_displacement).reshape(3, 3)

    return sampled_coords

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    preamble = """import numpy as np

def _modes_from_spectrum(n_mono, spectrum, seed):
    rng = np.random.default_rng(seed)
    freqs = np.zeros((n_mono, len(spectrum)))
    modes = np.zeros((n_mono, 9, len(spectrum)))
    for i in range(n_mono):
        basis = np.linalg.qr(rng.normal(size=(9, 9)))[0]
        columns = basis[:, :len(spectrum)]
        dominant = np.argmax(np.abs(columns), axis=0)
        signs = np.sign(columns[dominant, np.arange(len(spectrum))])
        signs[signs == 0.0] = 1.0
        modes[i] = columns * signs
        freqs[i] = np.array(spectrum, dtype=float)
    return freqs, modes
"""
    return [
        # --- Valid: four monomers with tetramer-like frequencies (normal case) ---
        {
            "setup": preamble + """
coords = np.array([
    [1.965757, 0.000000, 0.000000], [1.276254, 0.689503, 0.097837],
    [2.425271, 0.000868, 0.837181], [0.000000, 1.965757, 0.000000],
    [-0.680176, 1.285581, -0.077118], [0.004437, 2.439238, -0.852304],
    [-1.965757, 0.000000, 0.000000], [-1.270755, -0.695002, 0.118515],
    [-2.428412, 0.000015, 0.841160], [0.000000, -1.965757, 0.000000],
    [0.684659, -1.281098, -0.058165], [0.015394, -2.427969, -0.869683]])
masses = np.tile(np.array([15.9994, 1.00794, 1.00794]), 4)
freqs, modes = _modes_from_spectrum(4, [35.58, 32.56, 15.01], 17)
""",
            "call": "backmap_thermal_configuration(coords, freqs, modes, masses, 250.0, 0)",
            "gold_call": "_oracle_backmap_thermal_configuration(coords, freqs, modes, masses, 250.0, 0)",
        },
        # --- Valid: a different seed must give a different but reproducible draw ---
        {
            "setup": preamble + """
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
masses = np.tile(np.array([15.9994, 1.00794, 1.00794]), 2)
freqs, modes = _modes_from_spectrum(2, [36.0, 35.2, 14.6], 9)
""",
            "call": "backmap_thermal_configuration(coords, freqs, modes, masses, 250.0, 12345)",
            "gold_call": "_oracle_backmap_thermal_configuration(coords, freqs, modes, masses, 250.0, 12345)",
        },
        # --- Boundary: a very low temperature, where the displacement all but
        #     vanishes and the manifold configuration is recovered ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
freqs, modes = _modes_from_spectrum(1, [36.0, 35.2, 14.6], 4)
""",
            "call": "backmap_thermal_configuration(coords, freqs, modes, masses, 1.0e-6, 3)",
            "gold_call": "_oracle_backmap_thermal_configuration(coords, freqs, modes, masses, 1.0e-6, 3)",
        },
        # --- Edge: a single very soft mode at high temperature, where the
        #     amplitude is large and the mass weighting is most visible ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
freqs, modes = _modes_from_spectrum(1, [0.75], 6)
""",
            "call": "backmap_thermal_configuration(coords, freqs, modes, masses, 600.0, 1)",
            "gold_call": "_oracle_backmap_thermal_configuration(coords, freqs, modes, masses, 600.0, 1)",
        },
        # --- Invalid: mode array inconsistent with the frequency array ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
freqs, modes = _modes_from_spectrum(1, [36.0, 35.2, 14.6], 4)
bad_modes = modes[:, :, :2]
def run_model():
    try:
        backmap_thermal_configuration(coords, freqs, bad_modes, masses, 250.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_backmap_thermal_configuration(coords, freqs, bad_modes, masses, 250.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative seed ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
freqs, modes = _modes_from_spectrum(1, [36.0, 35.2, 14.6], 4)
def run_model():
    try:
        backmap_thermal_configuration(coords, freqs, modes, masses, 250.0, -5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_backmap_thermal_configuration(coords, freqs, modes, masses, 250.0, -5)
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
