"""
Forward model of the measured two-source HOM visibility: combine each arm's emission ratios with the per-event HOM coincidence rates, and compare the total coincidence rate with the fully delayed rate.

The visibility is defined against the fully delayed coincidence rate, i.e. the rate when the temporal overlap between the two arms vanishes. The photons of a pair stay overlapped with each other at that delay, while every inter-arm overlap, including the effective one of the three-photon events, goes to zero. Event weights come from the arms' pair-to-single and vacuum-to-single ratios; a pair-plus-vacuum event needs one arm to emit a pair while the other is empty.

Returns
-------
float: the two-source HOM visibility V predicted for the given intrinsic parameters and source statistics.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hom_visibility(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                   g2_s: float, nbar_s: float) -> float:
    """Two-source HOM visibility including multiphoton events.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap (same for both arms), 0 <= m_sn <= 1.
        g2_m (float): second-order coherence of the control (memory) arm.
        nbar_m (float): mean photon number per pulse of the control arm.
        g2_s (float): second-order coherence of the target (source) arm.
        nbar_s (float): mean photon number per pulse of the target arm.

    Raises:
        ValueError: propagated from invalid arm statistics or invalid
            eta / m_sn.

    Expected return:
        float: V, equal to eta for ideal single-photon sources and lower than
        eta once multiphoton events are present (for 0 < eta <= 1).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: hom_visibility
def _oracle_hom_visibility(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                           g2_s: float, nbar_s: float) -> float:
    w_m, r_m = _oracle_arm_emission_ratios(g2_m, nbar_m)
    w_s, r_s = _oracle_arm_emission_ratios(g2_s, nbar_s)
    event_weights = np.array([1.0, w_m * r_s + w_s * r_m, w_m + w_s])
    rate = float(event_weights @ _oracle_hom_event_rates(eta, m_sn))
    rate_delayed = float(event_weights @ _oracle_hom_event_rates(0.0, m_sn))
    return 1.0 - rate / rate_delayed

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': '',
            'call': 'round(hom_visibility(0.9618090900338264, 0.25, 0.018, 0.42, 0.041, 0.77), 8)',
            'gold_call': 'round(_oracle_hom_visibility(0.9618090900338264, 0.25, 0.018, 0.42, 0.041, 0.77), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(hom_visibility(1.0, 1.0, 0.0, 0.5, 0.0, 0.5), 8)',
            'gold_call': 'round(_oracle_hom_visibility(1.0, 1.0, 0.0, 0.5, 0.0, 0.5), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(hom_visibility(0.85, 0.0, 0.05, 0.6, 0.02, 0.9), 8)',
            'gold_call': 'round(_oracle_hom_visibility(0.85, 0.0, 0.05, 0.6, 0.02, 0.9), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(hom_visibility(0.7, 0.6, 0.003, 0.85, 0.03, 0.25), 8)',
            'gold_call': 'round(_oracle_hom_visibility(0.7, 0.6, 0.003, 0.85, 0.03, 0.25), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'def run_model():\n    try:\n        hom_visibility(0.9, 0.25, 0.1, 1.2, 0.01, 0.3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_hom_visibility(0.9, 0.25, 0.1, 1.2, 0.01, 0.3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
