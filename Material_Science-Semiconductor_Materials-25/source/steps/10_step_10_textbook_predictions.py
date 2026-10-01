"""
Compute the depletion-approximation and short-diode-law predictions for the symmetric silicon junction from its material parameters.

For a symmetric abrupt junction the built-in potential is V_bi = V_T ln(N0^2 / n_ie^2). The depletion approximation treats the depleted region as sharp-edged and free of mobile carriers, which gives W = sqrt(4 eps V_bi / (q N0)) and a triangular field profile peaking at the junction with E = q N0 W/(2 eps) = sqrt(q N0 V_bi / eps). For the forward current, the absence of recombination makes the diffusion length infinite, and the Ohmic contacts force the excess minority density to zero, so each neutral region is a short diode. The minority density injected at the depletion edge is (n_ie^2/N0)(exp(Va/V_T) - 1) above equilibrium and falls linearly to zero across the neutral width W' = 0.5 um - W(Va)/2, where W(Va) uses V_bi - Va. Adding the electron and hole diffusion currents gives J = q (n_ie^2/N0)(exp(Va/V_T) - 1)(D_n + D_p)/W'. These formulas carry the material physics: the depletion width falls as N0^(-1/2), the peak field rises as N0^(1/2), and the current falls as 1/N0 because the injected minority population is proportional to n_ie^2/N0. At 0.5 V the injected density is at most about 3e12 cm^-3, far below N0, so the low-injection assumption holds for all three doping levels.

Returns
-------
np.ndarray of shape (4,), float: V_bi (V), W_DA (micrometres), E_DA (V/cm), J_SD (A/cm^2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def textbook_predictions(N0: float, Va: float) -> "np.ndarray":
    """Depletion-approximation and short-diode predictions for the symmetric junction.

    Parameters
    ----------
    N0 : float
        Doping magnitude on each side, cm^-3.
    Va : float
        Forward bias in volts for the current (0 <= Va < V_bi).

    Returns
    -------
    result : "np.ndarray"
        Shape (4,): [V_bi (V), W_DA (micrometres, at 0 V), E_DA (V/cm, at 0 V),
        J_SD (A/cm^2, at Va)], with V_bi = V_T ln(N0^2/n_ie^2),
        W_DA = sqrt(4 eps V_bi/(q N0)), E_DA = q N0 W_DA/(2 eps),
        J_SD = q (n_ie^2/N0) (exp(Va/V_T) - 1) (D_n + D_p) / W', and
        W' = 0.5e-4 cm - sqrt(4 eps (V_bi - Va)/(q N0))/2. Constants:
        q = 1.602192e-19 C, eps = 1.035941e-12 C/(V cm), V_T = 0.025852 V,
        n_ie = 1.087386e10 cm^-3, D_n = V_T*1417.0, D_p = V_T*470.5 cm^2/s.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_textbook_predictions(N0: float, Va: float) -> "np.ndarray":
    q, eps, VT, nie = 1.602192e-19, 1.035941e-12, 0.025852, 1.087386e10
    Dn, Dp = VT * 1417.0, VT * 470.5
    Vbi = VT * np.log(N0 * N0 / nie ** 2)
    W = np.sqrt(4.0 * eps * Vbi / (q * N0))
    E = q * N0 * W / (2.0 * eps)
    Wn = 0.5e-4 - 0.5 * np.sqrt(4.0 * eps * (Vbi - Va) / (q * N0))
    J = q * nie ** 2 / N0 * np.expm1(Va / VT) * (Dn + Dp) / Wn
    return np.array([Vbi, W * 1e4, E, J])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: lowest doping of the study at the study bias.
        {"setup": "import numpy as np\n",
         "call": "textbook_predictions(1e16, 0.5)",
         "gold_call": "_oracle_textbook_predictions(1e16, 0.5)",
         "tol": 1e-9},
        # Normal: intermediate doping.
        {"setup": "import numpy as np\n",
         "call": "textbook_predictions(3e16, 0.5)",
         "gold_call": "_oracle_textbook_predictions(3e16, 0.5)",
         "tol": 1e-9},
        # Edge: highest doping, narrowest depletion region.
        {"setup": "import numpy as np\n",
         "call": "textbook_predictions(1e17, 0.5)",
         "gold_call": "_oracle_textbook_predictions(1e17, 0.5)",
         "tol": 1e-9},
        # Boundary: zero bias, where the diode current vanishes exactly.
        {"setup": "import numpy as np\n",
         "call": "textbook_predictions(1e17, 0.0)",
         "gold_call": "_oracle_textbook_predictions(1e17, 0.0)",
         "tol": 1e-9},
    ]
