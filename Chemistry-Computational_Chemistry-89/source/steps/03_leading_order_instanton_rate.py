"""
Compute the leading-order semiclassical instanton rate constant of the reaction surface from a converged discretized tunnelling orbit.

This is the standard leading term of the semiclassical expansion, in which the fluctuations about the orbit are treated harmonically. Two conventions fix the value at finite bead number and are used by the tests: energies are measured from the reactant asymptote, and the reactant partition function is built with the same number of beads as the orbit, so that discretization errors cancel between the two. The reactant state is free motion along x together with the vibration of the reactant channel.

Returns
-------
float: the leading-order instanton rate constant in atomic units (bohr per atomic unit of time)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def leading_order_instanton_rate(beads: "np.ndarray", beta: float, params: dict) -> float:
    '''Leading-order instanton rate constant for a given tunnelling orbit.

    Parameters
    ----------
    beads : np.ndarray
        Array of shape (N, 2), N >= 4, with the (x, y) bead coordinates in
        bohr of a converged orbit at this beta. The chain is closed: bead
        N - 1 is a neighbour of bead 0.
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1).
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    k0 : float
        The leading-order rate constant in atomic units, that is bohr per
        atomic unit of time, being a flux per unit length of reactant density
        along x with the reactant vibration thermally populated.

    Raises
    ------
    ValueError
        If beads is not an (N, 2) array with N >= 4.
    '''
    return k0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _cyclic_laplacian(n):
    """Second-difference matrix of a closed ring of n beads."""
    eye = np.eye(n)
    return 2.0 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)


def _oracle_leading_order_instanton_rate(beads: "np.ndarray", beta: float, params: dict) -> float:
    q = np.asarray(beads, dtype=float)
    if q.ndim != 2 or q.shape[1] != 2 or q.shape[0] < 4:
        raise ValueError("beads must have shape (N, 2) with N >= 4")
    n = q.shape[0]
    m, we = float(params["m"]), float(params["omega_e"])
    beta_n = float(beta) / n
    dq = np.roll(q, -1, axis=0) - q
    stretch = float(np.sum(dq ** 2))
    u_n = float(np.sum(_oracle_potential_derivative_tensor(q, 0, params))) + 0.5 * m * stretch / beta_n ** 2
    b_n = m * stretch

    hess = np.kron(_cyclic_laplacian(n), np.eye(2)) * (m / beta_n ** 2)
    blocks = _oracle_potential_derivative_tensor(q, 2, params)
    for i in range(n):
        hess[2 * i:2 * i + 2, 2 * i:2 * i + 2] += blocks[i]
    lam = np.linalg.eigvalsh(hess / m)
    lam = np.delete(lam, int(np.argmin(np.abs(lam))))
    eta = np.sqrt(np.abs(lam))

    log_kz = (-np.log(beta_n) + 0.5 * np.log(b_n / (2.0 * np.pi * beta_n))
              - np.sum(np.log(beta_n * eta)) - beta_n * u_n)
    w_k = np.sqrt(we ** 2 + (2.0 * np.sin(np.pi * np.arange(n) / n) / beta_n) ** 2)
    log_z = 0.5 * np.log(m / (2.0 * np.pi * beta)) - np.sum(np.log(beta_n * w_k))
    return float(np.exp(log_kz - log_z))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
params = {"V0": 0.01135, "V_inf": -0.00485, "a": 0.735, "m": 1836.15267, "omega_e": 0.00415,
          "chi_inf": 0.0138, "chi_0": 0.0735, "sigma_e": 0.565}
def ring(half):
    x = np.array(half)
    return np.column_stack([np.r_[x, x[::-1]], np.zeros(2 * x.size)])
"""
    return [
        # Normal: converged 32-bead orbit on the asymmetric surface.
        {
            "setup": base + """
beads = ring([0.67494670035, 0.6175896465417, 0.4992084933957, 0.3148117690676, 0.0682774829937,
              -0.2070049234843, -0.459757482802, -0.667425492545, -0.8321740918445,
              -0.9619730392169, -1.0638628169998, -1.1429856404206, -1.2029111247497,
              -1.2460630561916, -1.274034938918, -1.2877927910398])
""",
            "call": "np.log(leading_order_instanton_rate(beads.copy(), 2975.0, dict(params)))",
            "gold_call": "np.log(_oracle_leading_order_instanton_rate(beads.copy(), 2975.0, dict(params)))",
        },
        # Normal: symmetric barrier (V_inf = 0) at a different temperature, 32 beads.
        {
            "setup": base + """
params.update({"V_inf": 0.0})
beads = ring([0.9568817409981, 0.932315158664, 0.8819984327892, 0.8034464242426, 0.6927134518273,
              0.5445502274627, 0.354190543771, 0.1240801417113, -0.1240801417113,
              -0.354190543771, -0.5445502274627, -0.6927134518273, -0.8034464242426,
              -0.8819984327892, -0.932315158664, -0.9568817409981])
""",
            "call": "np.log(leading_order_instanton_rate(beads.copy(), 2600.0, dict(params)))",
            "gold_call": "np.log(_oracle_leading_order_instanton_rate(beads.copy(), 2600.0, dict(params)))",
        },
        # Boundary: coarse 24-bead chain at the benchmark temperature.
        {
            "setup": base + """
beads = ring([0.6708534667401, 0.5683877065723, 0.3528407596142, 0.022812055978, -0.3428113661049,
              -0.6396274547636, -0.8583821432883, -1.0175416597557, -1.1321900682244,
              -1.2118873491788, -1.2623728013222, -1.2868840924925])
""",
            "call": "np.log(leading_order_instanton_rate(beads.copy(), 2975.0, dict(params)))",
            "gold_call": "np.log(_oracle_leading_order_instanton_rate(beads.copy(), 2975.0, dict(params)))",
        },
        # Edge: near the crossover temperature, short orbit with a weak unstable mode.
        {
            "setup": base + """
beads = ring([0.4177888168435, 0.3258671787686, 0.1469158973101, -0.0908432589165,
              -0.325403369836, -0.5085299663963, -0.6281543870705, -0.6865036367066])
""",
            "call": "np.log(leading_order_instanton_rate(beads.copy(), 1750.0, dict(params)))",
            "gold_call": "np.log(_oracle_leading_order_instanton_rate(beads.copy(), 1750.0, dict(params)))",
        },
        # Normal: reduced units (m = 1), a different barrier shape.
        {
            "setup": base + """
params = {"V0": 13.5 / np.pi, "V_inf": -18.0 / np.pi, "a": 8.0 / np.sqrt(3.0 * np.pi), "m": 1.0,
          "omega_e": 1.0, "chi_inf": 0.01, "chi_0": 0.01, "sigma_e": 1.0}
beads = ring([0.6927263724096, 0.1926556468194, -0.6735307040992, -1.6078274923024,
              -2.4058809763663, -3.036155467274, -3.5234371053762, -3.8956401685531,
              -4.1737057007376, -4.3719058314438, -4.4994938412488, -4.5619962388852])
""",
            "call": "np.log(leading_order_instanton_rate(beads.copy(), 4.0 * np.pi, dict(params)))",
            "gold_call": "np.log(_oracle_leading_order_instanton_rate(beads.copy(), 4.0 * np.pi, dict(params)))",
        },
        # Invalid: a flat list of x coordinates is not an (N, 2) bead array.
        {
            "setup": base + """
beads = np.array([0.5, 0.1, -0.3, -0.6])
def run_model():
    try:
        leading_order_instanton_rate(beads.copy(), 2975.0, dict(params))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_leading_order_instanton_rate(beads.copy(), 2975.0, dict(params))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
