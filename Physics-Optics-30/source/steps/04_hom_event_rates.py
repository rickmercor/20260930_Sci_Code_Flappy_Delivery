"""
Compute the two-fold coincidence rate at a single balanced HOM beam splitter, per unit event weight, for each class of emission event: one photon per arm, a pair in one arm with the other arm empty, and a pair in one arm with a single photon in the other.

In the visibility measurement the two sources meet on one balanced beam splitter with both outputs monitored, so there are no unmonitored ports and no polarization degree of freedom. The extra photon of a three-photon event interferes with the same effective indistinguishability that governs the gate. Consult the paper's HOM coincidence-rate table rather than assuming the three-photon rate is a rescaled single-photon rate.

Returns
-------
np.ndarray of shape (3,): [Gamma_11, Gamma_pair, Gamma_3], the per-event coincidence rates of the one-plus-one event, of either pair-plus-vacuum event, and of either pair-plus-single event.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hom_event_rates(eta: float, m_sn: float) -> "np.ndarray":
    """Per-event coincidence rates at the balanced HOM beam splitter.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap, 0 <= m_sn <= 1.

    Raises:
        ValueError: if eta or m_sn lies outside [0, 1].

    Expected return:
        np.ndarray of shape (3,): [Gamma_11, Gamma_pair, Gamma_3]. Gamma_11
        vanishes for perfectly indistinguishable photons; Gamma_pair does not
        depend on eta; Gamma_3 decreases linearly with eta.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: hom_event_rates
def _oracle_hom_event_rates(eta: float, m_sn: float) -> "np.ndarray":
    e_eff = _oracle_effective_indistinguishability(eta, m_sn)
    return np.array([0.5 * (1.0 - eta), 0.5, 1.5 - e_eff])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np',
            'call': 'np.round(hom_event_rates(0.96, 0.25), 8)',
            'gold_call': 'np.round(_oracle_hom_event_rates(0.96, 0.25), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(hom_event_rates(1.0, 1.0), 8)',
            'gold_call': 'np.round(_oracle_hom_event_rates(1.0, 1.0), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(hom_event_rates(0.7, 0.0), 8)',
            'gold_call': 'np.round(_oracle_hom_event_rates(0.7, 0.0), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'import numpy as np',
            'call': 'np.round(hom_event_rates(0.0, 0.6), 8)',
            'gold_call': 'np.round(_oracle_hom_event_rates(0.0, 0.6), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'def run_model():\n    try:\n        hom_event_rates(-0.1, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_hom_event_rates(-0.1, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
