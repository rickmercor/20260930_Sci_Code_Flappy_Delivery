"""
Convert a positron total energy in a superallowed beta decay into its velocity, its momentum and the energy left to the neutrino, and validate that the energy lies inside the allowed spectrum.

In a superallowed 0+ -> 0+ positron decay the charged lepton carries a total energy E_e between its rest mass m_e and the endpoint E_0 = Q_EC - m_e, where Q_EC is the electron-capture Q value of the transition; the neutrino carries the remainder Ebar = E_0 - E_e. The tree-level spectrum is proportional to p_e E_e Ebar^2 with p_e = sqrt(E_e^2 - m_e^2), and every radiative correction is expressed through the velocity beta = p_e / E_e = sqrt(1 - m_e^2 / E_e^2) in (0, 1) together with Ebar and E_e. All energies are in MeV.

Returns
-------
tuple (float, float, float) — the velocity beta, the momentum p_e in MeV and the neutrino energy Ebar = E_0 - E_e in MeV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lepton_kinematics(E_e: float, m_e: float, E_0: float) -> tuple:
    '''Velocity, momentum and neutrino energy for a positron of energy E_e.

    Parameters
    ----------
    E_e : float
        Positron total energy in MeV, with m_e < E_e < E_0.
    m_e : float
        Positron mass in MeV, > 0.
    E_0 : float
        Endpoint energy E_0 = Q_EC - m_e in MeV, > m_e.

    Returns
    -------
    result : tuple of (float, float, float)
        (beta, p_e, Ebar): the velocity beta = p_e / E_e, the momentum
        p_e = sqrt(E_e^2 - m_e^2) in MeV and the neutrino energy Ebar = E_0 -
        E_e in MeV, all as native Python floats.

    Raises
    ------
    ValueError
        If m_e <= 0, E_0 <= m_e, or E_e is not strictly inside (m_e, E_0).
    '''
    return beta, p_e, Ebar  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_lepton_kinematics(E_e: float, m_e: float, E_0: float) -> tuple:
    E_e = float(E_e); m_e = float(m_e); E_0 = float(E_0)
    if not (m_e > 0.0 and E_0 > m_e and m_e < E_e < E_0):
        raise ValueError("require m_e > 0, E_0 > m_e and m_e < E_e < E_0")
    p_e = math.sqrt((E_e - m_e) * (E_e + m_e))
    beta = p_e / E_e
    return float(beta), float(p_e), float(E_0 - E_e)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: mid-spectrum energy of the 26Al -> 26Mg transition ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895; E0 = 4.2327 - m
""",
            "call": "np.array(lepton_kinematics(*copy.deepcopy((2.0, m, E0))))",
            "gold_call": "np.array(_oracle_lepton_kinematics(*copy.deepcopy((2.0, m, E0))))",
            "tol": 1e-9,
        },
        # --- Boundary: just above threshold, where beta is small and the
        #     difference E_e^2 - m_e^2 must be formed without cancellation ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895; E0 = 4.2327 - m
""",
            "call": "np.array(lepton_kinematics(*copy.deepcopy((m * (1.0 + 1.0e-9), m, E0))))",
            "gold_call": "np.array(_oracle_lepton_kinematics(*copy.deepcopy((m * (1.0 + 1.0e-9), m, E0))))",
            "tol": 1e-7,
        },
        # --- Boundary: just below the endpoint (Ebar -> 0) ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895; E0 = 4.2327 - m
""",
            "call": "np.array(lepton_kinematics(*copy.deepcopy((E0 - 1.0e-6, m, E0))))",
            "gold_call": "np.array(_oracle_lepton_kinematics(*copy.deepcopy((E0 - 1.0e-6, m, E0))))",
            "tol": 1e-9,
        },
        # --- Edge: a heavier lepton and a low endpoint (muon-like numbers) ---
        {
            "setup": """import numpy as np
import copy
""",
            "call": "np.array(lepton_kinematics(*copy.deepcopy((150.0, 105.6583755, 300.0))))",
            "gold_call": "np.array(_oracle_lepton_kinematics(*copy.deepcopy((150.0, 105.6583755, 300.0))))",
            "tol": 1e-9,
        },
        # --- Invalid: energy above the endpoint -> ValueError ---
        {
            "setup": """import numpy as np
import copy
def run_model():
    try:
        lepton_kinematics(*copy.deepcopy((5.0, 0.51099895, 3.7217)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lepton_kinematics(*copy.deepcopy((5.0, 0.51099895, 3.7217)))
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
