"""
Applies the method's closed-form bond-level phase-field evolution to the damage-history variable, with the critical crack-driving-force threshold derived from Gc, the horizon, and the normalization constant produced by step 2. Works elementwise on arrays of bond histories. Validates the evolution law and the threshold it pivots on; the returned bond phase-field is consumed by every internal-force evaluation of step 7, by the initial state of the dynamics of step 8, and by the orchestrator of step 9. Deliberately excluded: how the history variable is accumulated in time.

No phase-field partial differential equation is solved anywhere in this method: the source gives a closed-form algebraic bond-level expression for the bond phase-field in terms of the bond damage-history variable, pivoting on a critical crack-driving-force threshold that is fixed by Gc, the horizon, and the normalization constant of step 2. Irreversibility rests on the history variable, which the caller accumulates over the loading history. The law and the threshold must be taken from the source.

Returns
-------
float or np.ndarray: bond phase-field value(s) in [0, 1] (dimensionless)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bond_phase_field(calY: "float | np.ndarray", Gc: float, delta: float,
                     c0: float) -> "float | np.ndarray":
    """Bond phase-field value(s) from the damage-history variable.

    Applies the method's closed-form bond-level evolution law to the bond
    damage-history variable, with the critical crack-driving-force
    threshold derived from Gc, the horizon delta and the Griffith-
    consistency normalization constant c0 of the kernel (the output of
    step 2) as the source prescribes. The law and the threshold must be
    taken from the source; irreversibility is carried by the history
    variable that the caller supplies. Works elementwise on arrays so that
    it can be applied to every bond pair at once; it is consumed by every
    internal-force evaluation of step 7, by the initial state of the
    dynamics of step 8, and by the orchestrator of step 9.

    Parameters
    ----------
    calY : float or np.ndarray
        Bond damage-history variable(s) in J/m^3 (>= 0).
    Gc : float
        Critical energy release rate in J/m^2 (> 0).
    delta : float
        Horizon radius in m (> 0).
    c0 : float
        Griffith-consistency normalization constant of the damage model
        for the kernel in use (dimensionless, in (0, 1)).

    Returns
    -------
    float or np.ndarray
        Bond phase-field value(s) in [0, 1]: a float for scalar input, an
        array of the same shape for array input.

    Raises
    ------
    ValueError
        If any calY < 0, Gc <= 0, delta <= 0, or c0 is not in (0, 1).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bond_phase_field(calY, Gc, delta, c0):
    calY = np.asarray(calY, dtype=float)
    if np.any(calY < 0.0):
        raise ValueError("history values must be non-negative")
    if not (Gc > 0.0 and delta > 0.0):
        raise ValueError("require Gc > 0 and delta > 0")
    if not (0.0 < c0 < 1.0):
        raise ValueError("normalization constant must lie in (0, 1)")
    Yc = Gc/(2.0*c0*delta)
    s = np.minimum(1.0, calY/(calY + Yc))
    if s.ndim == 0:
        return float(s)
    return s

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"name": "normal_subcritical_history",
         "setup": "",
         "call": "bond_phase_field(1.0e5, 120.0, 2.015e-3, pfpd_normalization_constant('cubic'))",
         "gold_call": "_oracle_bond_phase_field(1.0e5, 120.0, 2.015e-3, _oracle_pfpd_normalization_constant('cubic'))"},
        {"name": "boundary_supercritical_history",
         "setup": "",
         "call": "bond_phase_field(2.5e5, 120.0, 2.015e-3, pfpd_normalization_constant('cubic'))",
         "gold_call": "_oracle_bond_phase_field(2.5e5, 120.0, 2.015e-3, _oracle_pfpd_normalization_constant('cubic'))"},
        {"name": "edge_pre_notch_history_saturates_at_one",
         "setup": "",
         "call": "bond_phase_field(1.0e30, 120.0, 2.015e-3, pfpd_normalization_constant('cubic'))",
         "gold_call": "_oracle_bond_phase_field(1.0e30, 120.0, 2.015e-3, _oracle_pfpd_normalization_constant('cubic'))"},
        {"name": "edge_vectorized_history_array",
         "setup": "import numpy as np\ncalY = np.array([0.0, 5.0e4, 1.0e5, 2.5e5, 1.0e30])",
         "call": "float(np.sum(bond_phase_field(calY, 120.0, 2.015e-3, pfpd_normalization_constant('cubic'))))",
         "gold_call": "float(np.sum(_oracle_bond_phase_field(calY, 120.0, 2.015e-3, _oracle_pfpd_normalization_constant('cubic'))))"},
    ]
