"""
Compute the Compton total cross section at NNLO matched to the all-order leading-logarithmic resummation for a single fixed-target photon energy, and return it in microbarn.

This step combines the fixed-order and resummed pieces at one energy: the kinematics (Step 1), the high-energy expansions of the leading, next-to-leading and next-to-next-to-leading orders (Steps 2-4), the leading-logarithmic ladder terms (Step 5) and their closed-form resummation (Step 6). The matched prediction adds to the fixed-order NNLO cross section the resummed leading logarithms with their first three ladder orders removed (additive matching, so that no logarithm is counted twice): sigma_{NNLO+LL} = sigma_LO + sigma_NLO + sigma_NNLO + [sigma_LL - sum_{n=1}^{3} sigma_LL^{(n)}]. The result in natural units is converted with 1 GeV^-2 = 389.3793721 microbarn. The high-energy expansions are valid for tau = m^2/s <= 0.01, i.e. photon energies above 25.3 MeV on a fixed electron target.

Returns
-------
float — the NNLO+LL Compton total cross section in microbarn.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compton_nnlo_ll(E_gamma: float, m: float, alpha: float) -> float:
    '''NNLO+LL Compton total cross section for a fixed-target photon energy.

    Parameters
    ----------
    E_gamma : float
        Laboratory photon energy in GeV, large enough that tau = m^2/s <= 0.01.
    m : float
        Electron mass in GeV, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_ub : float
        sigma_{NNLO+LL} in microbarn, as a native Python float.

    Raises
    ------
    ValueError
        Propagated from the individual steps on invalid input, in particular
        if tau > 0.01 (outside the domain of the high-energy expansions).
    '''
    return sigma_ub  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compton_nnlo_ll(E_gamma: float, m: float, alpha: float) -> float:
    s, tau, _sigma_kn = _oracle_compton_kinematics_lo(E_gamma, m, alpha)
    sigma = _oracle_lo_high_energy_series(tau, s, alpha) \
        + _oracle_nlo_high_energy_series(tau, s, alpha) \
        + _oracle_nnlo_high_energy_series(tau, s, alpha)
    _sigma_ll, remainder = _oracle_ll_resummed(tau, s, alpha, 3)
    sigma += remainder
    return float(sigma * 389.3793721)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the exact problem instance, E_gamma = 8 GeV ---
        {
            "setup": """import numpy as np
import copy
E = 8.0; m = 0.51099895e-3; a = 1.0 / 137.035999084
""",
            "call": "compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "gold_call": "_oracle_compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "tol": 1e-9,
        },
        # --- Normal: the paper's benchmark sqrt(s) = 1 GeV ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
E = (1.0 - m * m) / (2.0 * m)
""",
            "call": "compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "gold_call": "_oracle_compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "tol": 1e-9,
        },
        # --- Boundary: the lowest energy inside the domain, tau = 0.01 exactly,
        #     E_gamma = m (1/tau - 1)/2 ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
E = m * (1.0 / 0.01 - 1.0) / 2.0
""",
            "call": "compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "gold_call": "_oracle_compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "tol": 1e-9,
        },
        # --- Edge: sqrt(s) = 1 TeV, where the NNLO term is 2.5 % of NLO and the
        #     resummation beyond it is visible ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
E = (1.0e6 - m * m) / (2.0 * m)
""",
            "call": "compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "gold_call": "_oracle_compton_nnlo_ll(*copy.deepcopy((E, m, a)))",
            "tol": 1e-9,
        },
        # --- Invalid: a photon energy below the high-energy domain (tau > 0.01)
        #     -> ValueError ---
        {
            "setup": """import numpy as np
import copy
def run_model():
    try:
        compton_nnlo_ll(*copy.deepcopy((0.01, 0.51099895e-3, 1.0 / 137.035999084)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compton_nnlo_ll(*copy.deepcopy((0.01, 0.51099895e-3, 1.0 / 137.035999084)))
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
