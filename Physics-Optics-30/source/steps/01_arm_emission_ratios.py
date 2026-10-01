"""
Convert one arm's measured source statistics (second-order coherence and mean photon number per pulse at the gate input) into the two emission-probability ratios that weight every multiphoton event in both the gate and the HOM bookkeeping.

Each arm's photon-number distribution is truncated at two photons and is pinned exactly, not to first order, by g2(0) and the mean photon number per pulse. The pair-to-single ratio weights events in which this arm contributes an extra photon; the vacuum-to-single ratio weights events in which this arm is empty while the other arm emits a pair. Consult the paper's appendix on emission-event weights for the exact inversion of the two defining moments.

Returns
-------
np.ndarray of shape (2,): [w, r], the pair-to-single ratio P(2)/P(1) and the vacuum-to-single ratio P(0)/P(1) of this arm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def arm_emission_ratios(g2: float, nbar: float) -> "np.ndarray":
    """Pair-to-single and vacuum-to-single emission ratios of one arm.

    Args:
        g2 (float): second-order coherence g2(0) of this arm, g2 >= 0.
        nbar (float): mean photon number per pulse delivered at the gate
            input, nbar > 0.

    Raises:
        ValueError: if g2 < 0, if nbar <= 0, or if (g2, nbar) do not define a
            valid two-photon-truncated distribution (non-positive single-photon
            probability or negative vacuum probability).

    Expected return:
        np.ndarray of shape (2,): [w, r]. w >= 0 and vanishes at g2 = 0;
        r >= 0 and decreases as the arm gets brighter.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: arm_emission_ratios
def _oracle_arm_emission_ratios(g2: float, nbar: float) -> "np.ndarray":
    if g2 < 0.0:
        raise ValueError("g2 must be non-negative")
    if nbar <= 0.0:
        raise ValueError("nbar must be positive")
    p1 = nbar * (1.0 - g2 * nbar)
    p2 = 0.5 * g2 * nbar ** 2
    p0 = 1.0 - p1 - p2
    if p1 <= 0.0 or p0 < 0.0:
        raise ValueError("(g2, nbar) do not define a valid two-photon-truncated distribution")
    return np.array([p2 / p1, p0 / p1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np',
            'call': 'np.round(arm_emission_ratios(0.018, 0.42), 8)',
            'gold_call': 'np.round(_oracle_arm_emission_ratios(0.018, 0.42), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(arm_emission_ratios(0.041, 0.77), 8)',
            'gold_call': 'np.round(_oracle_arm_emission_ratios(0.041, 0.77), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(arm_emission_ratios(0.0, 0.5), 8)',
            'gold_call': 'np.round(_oracle_arm_emission_ratios(0.0, 0.5), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(arm_emission_ratios(0.05, 1.0), 8)',
            'gold_call': 'np.round(_oracle_arm_emission_ratios(0.05, 1.0), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'def run_model():\n    try:\n        arm_emission_ratios(0.018, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_arm_emission_ratios(0.018, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'def run_model():\n    try:\n        arm_emission_ratios(-0.01, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_arm_emission_ratios(-0.01, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'def run_model():\n    try:\n        arm_emission_ratios(0.1, 1.2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_arm_emission_ratios(0.1, 1.2)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'def run_model():\n    try:\n        arm_emission_ratios(1.5, 0.9)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_arm_emission_ratios(1.5, 0.9)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
