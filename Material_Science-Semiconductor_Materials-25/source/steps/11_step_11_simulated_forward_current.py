"""
Final orchestrator: run the doping study of the silicon junction and return the simulated forward current density at the lowest doping level.

This step answers the scientific question end to end. For each doping level N0 = 1e16, 3e16 and 1e17 cm^-3 it solves the junction at 0 V to obtain the depletion width and the peak electric field, solves it again at the forward bias to obtain the terminal current density, and evaluates the depletion-approximation and short-diode predictions from the same material parameters. The signed percentage differences expose where the simple theory fails: the peak field sits a few percent low because majority carriers spill a Debye length into the depletion region, so only about V_bi - 2V_T is supported by depleted charge; the half-density depletion width sits further low because that crossing also lies inside the smooth edge; and the current falls short of the short-diode law, most at the lowest doping where the finite depletion region occupies the largest fraction of the device. The scored result is the simulated forward current density at N0 = 1e16 cm^-3, the case where the drift-diffusion solution departs most from the ideal-diode picture, and the full table supports the reasoning behind it.

Returns
-------
float, the simulated forward current density at N0 = 1e16 cm^-3 in A/cm^2, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulated_forward_current(M: int = 64, Va: float = 0.5) -> float:
    """Simulated forward current density of the junction at N0 = 1e16 cm^-3.

    Builds, for each N0 in (1e16, 3e16, 1e17) cm^-3, the row
    [W_sim (micrometres), E_sim (V/cm), J_sim (A/cm^2), W_DA (micrometres),
    E_DA (V/cm), J_SD (A/cm^2), 100*(W_sim/W_DA - 1), 100*(E_sim/E_DA - 1),
    100*(J_sim/J_SD - 1)], where W_sim and E_sim come from the 0 V solution,
    J_sim from the solution at Va, and the rest from the textbook predictions.

    Parameters
    ----------
    M : int
        Mesh parameter of the distorted triangulation (64 for the reported result).
    Va : float
        Forward bias in volts applied to the top contact.

    Returns
    -------
    current : float
        J_sim for N0 = 1e16 cm^-3 in A/cm^2, positive for conventional current
        entering the device through the top contact.
    """
    return current  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _doping_table(M: int, Va: float) -> "np.ndarray":
    """Simulated and textbook quantities for the three doping levels, one row each."""
    V = (M + 1) ** 2
    rows = []
    for N0 in (1e16, 3e16, 1e17):
        s0 = _oracle_solve_drift_diffusion(N0, 0.0, M)
        W, E = _oracle_junction_electrostatics(s0[0, -V:], s0[1, -V:], s0[2, -V:], N0, M)
        s1 = _oracle_solve_drift_diffusion(N0, Va, M)
        J = _oracle_terminal_current(s1[0], s1[1], s1[2], M)
        _, Wda, Eda, Jsd = _oracle_textbook_predictions(N0, Va)
        rows.append([W, E, J, Wda, Eda, Jsd,
                     100.0 * (W / Wda - 1.0), 100.0 * (E / Eda - 1.0), 100.0 * (J / Jsd - 1.0)])
    return np.array(rows)


def _oracle_simulated_forward_current(M: int = 64, Va: float = 0.5) -> float:
    return float(_doping_table(M, Va)[0, 2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: full pipeline on the 8x8 mesh at the study bias of 0.5 V.
        {"setup": "import numpy as np\n",
         "call": "simulated_forward_current(8, 0.5)",
         "gold_call": "_oracle_simulated_forward_current(8, 0.5)",
         "tol": 1e-6},
        # Boundary: very coarse mesh (M = 4), only two cell rows on each side of the junction.
        {"setup": "import numpy as np\n",
         "call": "simulated_forward_current(4, 0.5)",
         "gold_call": "_oracle_simulated_forward_current(4, 0.5)",
         "tol": 1e-6},
        # Edge: lower bias of 0.4 V, where the current is about 50 times smaller.
        {"setup": "import numpy as np\n",
         "call": "simulated_forward_current(8, 0.4)",
         "gold_call": "_oracle_simulated_forward_current(8, 0.4)",
         "tol": 1e-6},
    ]
