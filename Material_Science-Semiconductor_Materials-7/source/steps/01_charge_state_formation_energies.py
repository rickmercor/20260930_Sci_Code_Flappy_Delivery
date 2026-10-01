"""
Compute the formation energy of every listed charge state of one point defect at a Fermi level of zero, from the neutral-state formation energy and the thermodynamic transition levels.

Defect databases usually report a neutral formation energy together with the charge-transition levels at which successive charge states become equally stable. Rebuilding the full set of charge-state formation energies from these data is the first step of any defect-concentration calculation, because every later equilibrium or freeze-in evaluation needs the formation energy of each charge state at an arbitrary Fermi level.

Returns
-------
np.ndarray of float, the formation energy (eV) of each listed charge state at a Fermi level of zero, in the order of charges
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def charge_state_formation_energies(e_neutral: float, charges: "np.ndarray", levels: "np.ndarray") -> "np.ndarray":
    '''Formation energies of the listed charge states at a Fermi level of zero.

    Parameters
    ----------
    e_neutral : float
        Formation energy of the neutral charge state (eV) at a Fermi level of zero.
    charges : np.ndarray
        One-dimensional integer array of charge states, strictly decreasing in steps of one,
        that contains the neutral state 0.
    levels : np.ndarray
        One-dimensional array of transition levels (eV, on the same energy scale as the Fermi
        level) with length len(charges) - 1. Entry i is the level between charges[i] and
        charges[i + 1], i.e. the Fermi level at which those two states have equal formation
        energy.

    Returns
    -------
    energies : np.ndarray
        Float array with the same length and order as charges: the formation energy (eV) of
        each charge state at a Fermi level of zero.

    Raises
    ------
    ValueError
        If charges does not contain 0, is not strictly decreasing in unit steps, or levels
        does not have length len(charges) - 1.
    '''
    return energies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_charge_state_formation_energies(e_neutral: float, charges: "np.ndarray", levels: "np.ndarray") -> "np.ndarray":
    charges = np.asarray(charges, dtype=int).ravel()
    levels = np.asarray(levels, dtype=float).ravel()
    if charges.size == 0 or not np.any(charges == 0):
        raise ValueError("charges must contain the neutral state 0")
    if charges.size > 1 and not np.all(np.diff(charges) == -1):
        raise ValueError("charges must decrease in steps of one")
    if levels.size != charges.size - 1:
        raise ValueError("levels must have length len(charges) - 1")

    energies = np.empty(charges.size, dtype=float)
    i0 = int(np.flatnonzero(charges == 0)[0])
    energies[i0] = float(e_neutral)
    # Walk outward from the neutral state; states i and i+1 cross at levels[i].
    for i in range(i0 - 1, -1, -1):
        energies[i] = energies[i + 1] + (charges[i + 1] - charges[i]) * levels[i]
    for i in range(i0 + 1, charges.size):
        energies[i] = energies[i - 1] + (charges[i - 1] - charges[i]) * levels[i - 1]
    return energies

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    raise_setup = """import numpy as np
def run(fn, e0, charges, levels):
    try:
        fn(e0, charges, levels)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Valid: double donor (2+, +, 0) with levels near the conduction band ---
        {
            "setup": """import numpy as np
charges = np.array([2, 1, 0])
levels = np.array([1.37, 1.49])
""",
            "call": "charge_state_formation_energies(3.15, charges.copy(), levels.copy())",
            "gold_call": "_oracle_charge_state_formation_energies(3.15, charges.copy(), levels.copy())",
        },
        # --- Valid: double acceptor (0, -, 2-) ---
        {
            "setup": """import numpy as np
charges = np.array([0, -1, -2])
levels = np.array([0.31, 0.74])
""",
            "call": "charge_state_formation_energies(2.61, charges.copy(), levels.copy())",
            "gold_call": "_oracle_charge_state_formation_energies(2.61, charges.copy(), levels.copy())",
        },
        # --- Valid: amphoteric defect (+, 0, -) with the neutral state in the middle ---
        {
            "setup": """import numpy as np
charges = np.array([1, 0, -1])
levels = np.array([0.42, 0.95])
""",
            "call": "charge_state_formation_energies(1.87, charges.copy(), levels.copy())",
            "gold_call": "_oracle_charge_state_formation_energies(1.87, charges.copy(), levels.copy())",
        },
        # --- Edge: neutral-only defect (no transition levels) ---
        {
            "setup": """import numpy as np
charges = np.array([0])
levels = np.array([])
""",
            "call": "charge_state_formation_energies(0.93, charges.copy(), levels.copy())",
            "gold_call": "_oracle_charge_state_formation_energies(0.93, charges.copy(), levels.copy())",
        },
        # --- Boundary: level below the energy zero (resonant with the valence band) ---
        {
            "setup": """import numpy as np
charges = np.array([1, 0])
levels = np.array([-0.08])
""",
            "call": "charge_state_formation_energies(1.30, charges.copy(), levels.copy())",
            "gold_call": "_oracle_charge_state_formation_energies(1.30, charges.copy(), levels.copy())",
        },
        # --- Invalid: charge list without the neutral state ---
        {
            "setup": raise_setup + """
charges = np.array([2, 1])
levels = np.array([1.1])
""",
            "call": "run(charge_state_formation_energies, 1.0, charges.copy(), levels.copy())",
            "gold_call": "run(_oracle_charge_state_formation_energies, 1.0, charges.copy(), levels.copy())",
        },
        # --- Invalid: wrong number of levels ---
        {
            "setup": raise_setup + """
charges = np.array([1, 0, -1])
levels = np.array([0.5])
""",
            "call": "run(charge_state_formation_energies, 1.0, charges.copy(), levels.copy())",
            "gold_call": "run(_oracle_charge_state_formation_energies, 1.0, charges.copy(), levels.copy())",
        },
    ]
