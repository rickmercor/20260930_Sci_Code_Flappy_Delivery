"""
Return the vibrational free-energy contribution to the formation free energy of each defect at a given temperature, from the net number of atoms the defect adds to or removes from the crystal.

Defect formation energies from first-principles supercells are 0 K energies. At growth and annealing temperatures the vibrational entropy change on forming a defect can shift equilibrium concentrations by orders of magnitude, yet mode-resolved phonon calculations for every charge state are seldom available. The defect calculation therefore uses an averaged treatment built from a single representative vibrational quantum of the host lattice.

Returns
-------
np.ndarray of float, the vibrational contribution (eV) to each defect's formation free energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vibrational_free_energy_shift(temperature: float, hw0: float, n_added: "np.ndarray") -> "np.ndarray":
    '''Vibrational free-energy term added to each defect's formation energy.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    hw0 : float
        Representative vibrational quantum of the host lattice (eV), strictly positive.
    n_added : np.ndarray
        One-dimensional integer array: net number of host or impurity atoms each defect adds
        to the crystal (positive for interstitial-type defects, negative for vacancies, zero
        for substitutionals and antisites).

    Returns
    -------
    shift : np.ndarray
        Float array with the same length as n_added: the vibrational contribution (eV) to each
        defect's formation free energy under the source framework's average mode-counting
        treatment. The same value applies to every charge state of a defect; a negative value
        lowers the formation free energy.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If temperature or hw0 is not positive.
    '''
    return shift

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_vibrational_free_energy_shift(temperature: float, hw0: float, n_added: "np.ndarray") -> "np.ndarray":
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    if not float(hw0) > 0.0:
        raise ValueError("hw0 must be positive")
    x = float(hw0) / (kb_ev * t)
    # Entropy (in kB) of one quantum harmonic oscillator, written in an overflow-safe form.
    s_mode = x / np.expm1(x) - np.log(-np.expm1(-x))
    n = np.asarray(n_added, dtype=float).ravel()
    return -3.0 * n * kb_ev * t * s_mode

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    raise_setup = """import numpy as np
def run(fn, t, hw, n):
    try:
        fn(t, hw, n)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Valid: interstitial, vacancy, substitutional and a two-atom cluster at 581 K ---
        {
            "setup": """import numpy as np
n_added = np.array([1, -1, 0, 2])
""",
            "call": "vibrational_free_energy_shift(581.0, 0.0119, n_added.copy())",
            "gold_call": "_oracle_vibrational_free_energy_shift(581.0, 0.0119, n_added.copy())",
            "tol": 1e-8,
        },
        # --- Valid: high temperature, stiffer lattice ---
        {
            "setup": """import numpy as np
n_added = np.array([1, 0, -1])
""",
            "call": "vibrational_free_energy_shift(1123.0, 0.0215, n_added.copy())",
            "gold_call": "_oracle_vibrational_free_energy_shift(1123.0, 0.0215, n_added.copy())",
            "tol": 1e-8,
        },
        # --- Edge: deep quantum regime (vibrational quantum well above kT, entropy nearly frozen out) ---
        {
            "setup": """import numpy as np
n_added = np.array([1, -2])
""",
            "call": "vibrational_free_energy_shift(20.0, 0.0119, n_added.copy())",
            "gold_call": "_oracle_vibrational_free_energy_shift(20.0, 0.0119, n_added.copy())",
            "tol": 1e-8,
        },
        # --- Boundary: room temperature ---
        {
            "setup": """import numpy as np
n_added = np.array([1])
""",
            "call": "vibrational_free_energy_shift(296.0, 0.0119, n_added.copy())",
            "gold_call": "_oracle_vibrational_free_energy_shift(296.0, 0.0119, n_added.copy())",
            "tol": 1e-8,
        },
        # --- Invalid: non-positive vibrational quantum ---
        {
            "setup": raise_setup + """
n_added = np.array([1])
""",
            "call": "run(vibrational_free_energy_shift, 600.0, 0.0, n_added.copy())",
            "gold_call": "run(_oracle_vibrational_free_energy_shift, 600.0, 0.0, n_added.copy())",
            "tol": 1e-8,
        },
    ]
