"""
Assemble the first-order corrected, exponentially resummed instanton rate constant of the reaction surface at a given inverse temperature.

This is the quantity requested in the problem statement: the thermal rate constant expanded asymptotically in hbar at fixed thermal time, with its complete first-order relative correction resummed exponentially rather than added as a partial sum. The earlier steps provide the tunnelling orbit, the leading-order rate constant and the order-hbar corrections; this step combines them and converts the result to the reported units, one atomic unit of velocity being 2.18769126364e8 cm/s. The same number of beads is used everywhere, so that the discretization errors of the parts cancel as far as possible.

Returns
-------
float: the first-order corrected, exponentially resummed rate constant in cm molecule^-1 s^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def corrected_instanton_rate(beta: float, n_beads: int, params: dict) -> float:
    '''First-order corrected, exponentially resummed instanton rate constant.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1), below the
        crossover temperature of the barrier.
    n_beads : int
        Number of beads N, an even integer, used for the tunnelling orbit and
        for every chain-based quantity derived from it. Larger N approaches
        the continuum limit.
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    k : float
        The first-order corrected, exponentially resummed rate constant in
        cm molecule^-1 s^-1, that is a flux per unit length of reactant
        density along x with the reactant vibration thermally populated.

    Raises
    ------
    ValueError
        If any component rejects its input: an invalid bead number, a
        temperature at or above the crossover temperature, or invalid
        surface parameters.
    '''
    return k

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_corrected_instanton_rate(beta: float, n_beads: int, params: dict) -> float:
    au_velocity_cm_s = 2.18769126364e8
    m = float(params["m"])
    beads = _oracle_ring_polymer_instanton(beta, n_beads, params)
    k0 = _oracle_leading_order_instanton_rate(beads, beta, params)
    curv = _oracle_potential_derivative_tensor(beads, 2, params)[:, 1, 1]
    cubic = _oracle_potential_derivative_tensor(beads, 3, params)[:, 1, 1, 1]
    quartic = _oracle_potential_derivative_tensor(beads, 4, params)[:, 1, 1, 1, 1]
    g = _oracle_ring_polymer_propagator(curv, beta, m)
    gamma_perp = _oracle_anharmonic_fluctuation_correction(g, cubic, quartic, beta)
    gamma_react = _oracle_reactant_partition_correction(beta, n_beads, params)
    gamma_path = _oracle_eckart_tunneling_correction(
        beta, params["V0"], params["V_inf"], params["a"], m)
    return float(au_velocity_cm_s * k0 * np.exp(gamma_path + gamma_perp - gamma_react))

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
        # Normal: benchmark surface and temperature with a 32-bead chain.
        {
            "setup": base,
            "call": "corrected_instanton_rate(2975.0, 32, dict(params))",
            "gold_call": "_oracle_corrected_instanton_rate(2975.0, 32, dict(params))",
        },
        # Normal: symmetric barrier with a narrow anharmonicity bump, lower temperature.
        {
            "setup": base + """
params.update({"V_inf": 0.0, "sigma_e": 1.1})
""",
            "call": "corrected_instanton_rate(2600.0, 48, dict(params))",
            "gold_call": "_oracle_corrected_instanton_rate(2600.0, 48, dict(params))",
        },
        # Boundary: uniform anharmonicity (chi_0 = chi_inf), so the transverse contributions cancel.
        {
            "setup": base + """
params.update({"chi_0": 0.0138})
""",
            "call": "corrected_instanton_rate(2975.0, 24, dict(params))",
            "gold_call": "_oracle_corrected_instanton_rate(2975.0, 24, dict(params))",
        },
        # Edge: close to the crossover temperature, where the orbit is short.
        {
            "setup": base,
            "call": "corrected_instanton_rate(1750.0, 16, dict(params))",
            "gold_call": "_oracle_corrected_instanton_rate(1750.0, 16, dict(params))",
        },
        # Invalid: above the crossover temperature no tunnelling orbit exists.
        {
            "setup": base + """
def run_model():
    try:
        corrected_instanton_rate(1000.0, 16, dict(params))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_corrected_instanton_rate(1000.0, 16, dict(params))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
