"""
[ORCHESTRATOR] Compute the NNLO+LL Compton total cross section averaged over a supplied fixed-target photon-energy window with the specified 1/E_gamma spectral weight, using Gauss-Legendre quadrature in ln E_gamma.

Use the matched cross section returned by the preceding step. The energy endpoints, electron mass, coupling and number of quadrature nodes are supplied as parameters. The final result is the normalized spectral average in microbarn. Compose the earlier steps through the matched per-energy prediction.

Returns
-------
float — the 1/E-weighted mean NNLO+LL Compton total cross section over the window, in microbarn.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectrum_averaged_cross_section(E_min: float, E_max: float, m: float, alpha: float, n_nodes: int) -> float:
    '''1/E-weighted mean of the NNLO+LL Compton cross section over [E_min, E_max].

    Parameters
    ----------
    E_min : float
        Lower edge of the photon-energy window in GeV, > 0 and large enough
        that tau = m^2/s <= 0.01 (about 25.3 MeV for the electron mass).
    E_max : float
        Upper edge of the window in GeV, > E_min.
    m : float
        Electron mass in GeV, > 0.
    alpha : float
        Fine-structure constant, > 0.
    n_nodes : int
        Number of Gauss-Legendre nodes on [ln E_min, ln E_max], >= 1
        (n_nodes = 1 is the single mid-point exp((ln E_min + ln E_max)/2)).

    Returns
    -------
    sigma_bar : float
        The spectrum-averaged sigma_{NNLO+LL} in microbarn, as a native
        Python float.

    Raises
    ------
    ValueError
        If E_min <= 0, E_max <= E_min or n_nodes < 1; also propagated from the
        earlier steps if any node lies outside the high-energy domain
        (tau > 0.01) or m or alpha is not strictly positive.
    '''
    return sigma_bar  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_spectrum_averaged_cross_section(E_min: float, E_max: float, m: float, alpha: float, n_nodes: int) -> float:
    if not (E_min > 0.0 and E_max > E_min and int(n_nodes) >= 1):
        raise ValueError("require 0 < E_min < E_max and n_nodes >= 1")
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    u_min, u_max = math.log(E_min), math.log(E_max)
    u = 0.5 * (u_max - u_min) * x + 0.5 * (u_max + u_min)
    total = 0.0
    for u_i, w_i in zip(u, w):
        total += float(w_i) * _oracle_compton_nnlo_ll(math.exp(float(u_i)), m, alpha)
    return float(0.5 * total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the problem instance, GlueX window 6.5-11.1 GeV, 64 nodes ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
""",
            "call": "spectrum_averaged_cross_section(*copy.deepcopy((6.5, 11.1, m, a, 64)))",
            "gold_call": "_oracle_spectrum_averaged_cross_section(*copy.deepcopy((6.5, 11.1, m, a, 64)))",
            "tol": 1e-9,
        },
        # --- Normal: the PrimEx window 4.400-5.475 GeV with 16 nodes ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
""",
            "call": "spectrum_averaged_cross_section(*copy.deepcopy((4.400, 5.475, m, a, 16)))",
            "gold_call": "_oracle_spectrum_averaged_cross_section(*copy.deepcopy((4.400, 5.475, m, a, 16)))",
            "tol": 1e-9,
        },
        # --- Boundary: a single node is the geometric mid-point of the window ---
        {
            "setup": """import numpy as np
import copy
m = 0.51099895e-3; a = 1.0 / 137.035999084
""",
            "call": "spectrum_averaged_cross_section(*copy.deepcopy((2.0, 8.0, m, a, 1)))",
            "gold_call": "_oracle_spectrum_averaged_cross_section(*copy.deepcopy((2.0, 8.0, m, a, 1)))",
            "tol": 1e-9,
        },
        # --- Edge: a wide window reaching the domain limit at E_min (tau = 0.01)
        #     with a heavier lepton and a different coupling ---
        {
            "setup": """import numpy as np
import copy
m = 0.1056583755; a = 0.01
E_lo = m * (1.0 / 0.01 - 1.0) / 2.0
""",
            "call": "spectrum_averaged_cross_section(*copy.deepcopy((E_lo, 1000.0, m, a, 40)))",
            "gold_call": "_oracle_spectrum_averaged_cross_section(*copy.deepcopy((E_lo, 1000.0, m, a, 40)))",
            "tol": 1e-9,
        },
        # --- Invalid: E_max <= E_min -> ValueError ---
        {
            "setup": """import numpy as np
import copy
def run_model():
    try:
        spectrum_averaged_cross_section(*copy.deepcopy((11.1, 6.5, 0.51099895e-3, 1.0 / 137.035999084, 8)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_spectrum_averaged_cross_section(*copy.deepcopy((11.1, 6.5, 0.51099895e-3, 1.0 / 137.035999084, 8)))
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
