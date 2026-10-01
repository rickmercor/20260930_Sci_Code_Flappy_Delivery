"""
Convert a fixed-target photon energy into the Mandelstam variable s and mass ratio tau, and evaluate the exact leading-order (Klein-Nishina) total Compton cross section.

The fixed-target relation is s = m^2 + 2 m E_gamma, with tau = m^2/s. Return all three quantities for the supplied positive E_gamma, m and alpha. The cross section must remain accurate in double precision across the documented domain, including small E_gamma/m and the high-energy limit. Cross sections are in GeV^-2.

Returns
-------
tuple (float, float, float) — (s in GeV^2, tau = m^2/s, Klein-Nishina total cross section in GeV^-2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compton_kinematics_lo(E_gamma: float, m: float, alpha: float) -> tuple:
    '''Kinematics and Klein-Nishina total cross section for fixed-target Compton scattering.

    Parameters
    ----------
    E_gamma : float
        Laboratory photon energy in GeV, > 0.
    m : float
        Electron mass in GeV, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    result : tuple of (float, float, float)
        (s, tau, sigma_LO): s = m^2 + 2 m E_gamma in GeV^2, tau = m^2/s, and
        the Klein-Nishina total cross section in GeV^-2, as native Python
        floats.

    Raises
    ------
    ValueError
        If E_gamma, m or alpha is not strictly positive.
    '''
    return s, tau, sigma_LO  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_compton_kinematics_lo(E_gamma: float, m: float, alpha: float) -> tuple:
    E_gamma = float(E_gamma); m = float(m); alpha = float(alpha)
    if not (E_gamma > 0.0 and m > 0.0 and alpha > 0.0):
        raise ValueError("E_gamma, m and alpha must be strictly positive")
    s = m * m + 2.0 * m * E_gamma
    tau = m * m / s
    x = E_gamma / m
    u = 2.0 * x
    l = math.log1p(u)
    # The combination g(u) = u (1 + u/2) / (1 + u) - log1p(u) is O(u^3) and
    # cancels catastrophically for u << 1 (it costs ~3 digits per decade, e.g.
    # 32 % error at E_gamma = 10 keV(-ish) when evaluated directly).  For small
    # u it is summed from its convergent series
    #   g(u) = u^2 sum_{k>=1} (-1)^(k+1) u^k k / (2 (k + 2)),
    # which is exact to double precision; the closed form is used otherwise.
    if u < 0.5:
        term = u ** 3 / 6.0          # k = 1
        g = term
        k = 2
        while abs(term) > 1e-19 * abs(g) and k < 200:
            term = (-1.0) ** (k + 1) * u ** (k + 2) * k / (2.0 * (k + 2))
            g += term
            k += 1
    else:
        g = u * (1.0 + 0.5 * u) / (1.0 + u) - l
    sigma = 2.0 * math.pi * (alpha / m) ** 2 * ((1.0 + x) / x ** 3 * g
                                                + l / u - (1.0 + 3.0 * x) / (1.0 + u) ** 2)
    return float(s), float(tau), float(sigma)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the problem instance, E_gamma = 8 GeV ---
        {
            "setup": """import numpy as np
import copy
E = 8.0; m = 0.51099895e-3; a = 1.0 / 137.035999084
""",
            "call": "np.array(compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "gold_call": "np.array(_oracle_compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "tol": 1e-9,
        },
        # --- Boundary: low energy, E_gamma = 1 keV, where the cross section is
        #     within a few permille of the Thomson value 8 pi alpha^2 / (3 m^2).
        #     The bracket suffers cancellation for x << 1 (a direct transcription
        #     of the formula loses ~8 digits), so this case is graded at 1e-7. ---
        {
            "setup": """import numpy as np
import copy
E = 1.0e-6; m = 0.51099895e-3; a = 1.0 / 137.035999084
""",
            "call": "np.array(compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "gold_call": "np.array(_oracle_compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "tol": 1e-7,
        },
        # --- Boundary (regression, numerical stability): E_gamma = 1e-11 GeV,
        #     where the Klein-Nishina bracket cancels catastrophically; a direct
        #     evaluation of the printed formula is 32 % high here, while the
        #     cross section must equal 1708.477374 GeV^-2 (Thomson limit
        #     1708.477441 GeV^-2 as E_gamma -> 0) ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
""",
            "call": "np.array(compton_kinematics_lo(*copy.deepcopy((1.0e-11, m, a))))",
            "gold_call": "np.array(_oracle_compton_kinematics_lo(*copy.deepcopy((1.0e-11, m, a))))",
            "tol": 1e-9,
        },
        # --- Normal: the paper's benchmark energy sqrt(s) = 1 GeV, i.e.
        #     E_gamma = (1 - m^2)/(2 m) ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
E = (1.0 - m * m) / (2.0 * m)
""",
            "call": "np.array(compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "gold_call": "np.array(_oracle_compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "tol": 1e-9,
        },
        # --- Edge: a heavy 'electron' (muon mass) and a scaled coupling, to
        #     check the m and alpha dependence separately ---
        {
            "setup": """import numpy as np
import copy
E = 0.5; m = 0.1056583755; a = 0.01
""",
            "call": "np.array(compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "gold_call": "np.array(_oracle_compton_kinematics_lo(*copy.deepcopy((E, m, a))))",
            "tol": 1e-9,
        },
        # --- Invalid: non-positive photon energy -> ValueError ---
        {
            "setup": """import numpy as np
import copy
def run_model():
    try:
        compton_kinematics_lo(*copy.deepcopy((0.0, 0.51099895e-3, 1.0 / 137.035999084)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compton_kinematics_lo(*copy.deepcopy((0.0, 0.51099895e-3, 1.0 / 137.035999084)))
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
