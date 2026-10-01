"""
Classify which finite-frequency exceptional-point (EP) regime a given (eps_z, gamma0) point of the gyroelectric-chiral medium falls into.

The medium supports a finite-frequency EP ring only within specific, disjoint windows of gamma0^2 that depend on which of three qualitatively different regimes eps_z belongs to. These windows are not the same inequality repeated three times; each regime has its own distinct threshold condition, and one of them requires the equatorial critical value computed in the previous step. Consult the paper's own case-by-case derivation rather than assuming a single uniform rule across all eps_z.

Returns
-------
int: 2 if a finite-frequency EP ring exists for this (eps_z, gamma0) pair, 0 otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def classify_ep_sector(eps_z: float, gamma0: float) -> int:
    """Finite-frequency EP-ring existence classifier.

    Args:
        eps_z (float): axial relative permittivity of the uniaxial medium.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.

    Raises:
        ValueError: if gamma0 < 0.

    Expected return:
        int: 2 if a finite-frequency exceptional-point ring exists for this
        parameter pair, 0 if it does not.
    """
    return 0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: classify_ep_sector
def _oracle_classify_ep_sector(eps_z: float, gamma0: float) -> int:
    if gamma0 < 0.0:
        raise ValueError("gamma0 must be non-negative")
    g0sq = gamma0 ** 2
    if eps_z > 4.0:
        return 2 if (4.0 < g0sq < eps_z) else 0
    elif -12.0 < eps_z < -4.0:
        crit = _oracle_equatorial_critical_gamma0_sq(eps_z)
        return 2 if g0sq >= crit else 0
    elif eps_z <= -12.0:
        return 2 if g0sq > 4.0 else 0
    else:
        return 0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = classify_ep_sector(9.0, 2.5)",
            "call": "v",
            "gold_call": "_oracle_classify_ep_sector(9.0, 2.5)",
        },
        {
            "setup": "v = classify_ep_sector(4.5, 1.9)",
            "call": "v",
            "gold_call": "_oracle_classify_ep_sector(4.5, 1.9)",
        },
        {
            "setup": "v = classify_ep_sector(-6.0, 3.0)",
            "call": "v",
            "gold_call": "_oracle_classify_ep_sector(-6.0, 3.0)",
        },
        {
            "setup": "v = classify_ep_sector(-6.0, 2.0)",
            "call": "v",
            "gold_call": "_oracle_classify_ep_sector(-6.0, 2.0)",
        },
        {
            "setup": "v = classify_ep_sector(-15.0, 3.0)",
            "call": "v",
            "gold_call": "_oracle_classify_ep_sector(-15.0, 3.0)",
        },
        {
            "setup": "v = classify_ep_sector(-15.0, 1.5)",
            "call": "v",
            "gold_call": "_oracle_classify_ep_sector(-15.0, 1.5)",
        },
        {
            "setup": "v = classify_ep_sector(0.0, 1.0)",
            "call": "v",
            "gold_call": "_oracle_classify_ep_sector(0.0, 1.0)",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        classify_ep_sector(9.0, -1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_classify_ep_sector(9.0, -1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
