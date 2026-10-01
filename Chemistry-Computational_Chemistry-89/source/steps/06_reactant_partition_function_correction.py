"""
Compute the first-order, order-hbar relative correction to the reactant partition function of the reaction surface.

The asymptotic expansion in hbar at fixed thermal time applies to the whole rate expression, so the reactant partition function in its denominator carries a correction of the same order as the one about the tunnelling orbit. In the reactant channel the barrier has died away and the vibration is the only anharmonic degree of freedom, with the anharmonicity constant of that channel. The correction is evaluated on a chain of the same length as the orbit, so that the discretization errors of numerator and denominator cancel.

Returns
-------
float: the first-order relative correction to the reactant partition function (dimensionless)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reactant_partition_correction(beta: float, n_beads: int, params: dict) -> float:
    '''First-order relative correction to the reactant partition function.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1).
    n_beads : int
        Number of beads N >= 3 of the chain used for the reactant channel,
        matching the chain used for the tunnelling orbit.
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    correction : float
        The first-order relative correction to the reactant partition
        function, dimensionless, defined with the same sign convention as
        the correction about the tunnelling orbit, that is as the term to be
        added to 1.

    Raises
    ------
    ValueError
        If n_beads is not an integer >= 3, or beta, the vibrational
        frequency or the reactant-channel anharmonicity constant is not
        positive.
    '''
    return correction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reactant_partition_correction(beta: float, n_beads: int, params: dict) -> float:
    if isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer)) or n_beads < 3:
        raise ValueError("n_beads must be an integer >= 3")
    m, we, chi = float(params["m"]), float(params["omega_e"]), float(params["chi_inf"])
    if not (float(beta) > 0.0 and we > 0.0 and chi > 0.0):
        raise ValueError("beta, omega_e and chi_inf must be positive")
    alpha = np.sqrt(2.0 * m * we * chi)
    k2 = m * we ** 2
    g = _oracle_ring_polymer_propagator(np.full(n_beads, k2), beta, m)
    return float(_oracle_anharmonic_fluctuation_correction(
        g, np.full(n_beads, -3.0 * k2 * alpha), np.full(n_beads, 7.0 * k2 * alpha ** 2), beta))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
params = {"V0": 0.01135, "V_inf": -0.00485, "a": 0.735, "m": 1836.15267, "omega_e": 0.00415,
          "chi_inf": 0.0138, "chi_0": 0.0735, "sigma_e": 0.565}
"""
    return [
        # Normal: benchmark reactant vibration, 64 beads.
        {
            "setup": base,
            "call": "reactant_partition_correction(2975.0, 64, dict(params))",
            "gold_call": "_oracle_reactant_partition_correction(2975.0, 64, dict(params))",
        },
        # Normal: higher temperature and stronger reactant anharmonicity.
        {
            "setup": base + """
params.update({"omega_e": 0.0037, "chi_inf": 0.027})
""",
            "call": "reactant_partition_correction(1250.0, 48, dict(params))",
            "gold_call": "_oracle_reactant_partition_correction(1250.0, 48, dict(params))",
        },
        # Boundary: coarsest chain (N = 3).
        {
            "setup": base,
            "call": "reactant_partition_correction(2975.0, 3, dict(params))",
            "gold_call": "_oracle_reactant_partition_correction(2975.0, 3, dict(params))",
        },
        # Edge: low temperature, where the vibration is effectively in its ground state.
        {
            "setup": base,
            "call": "reactant_partition_correction(19000.0, 400, dict(params))",
            "gold_call": "_oracle_reactant_partition_correction(19000.0, 400, dict(params))",
        },
        # Invalid: a vanishing anharmonicity constant leaves the reactant Morse undefined.
        {
            "setup": base + """
params.update({"chi_inf": 0.0})
def run_model():
    try:
        reactant_partition_correction(2975.0, 32, dict(params))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_reactant_partition_correction(2975.0, 32, dict(params))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
