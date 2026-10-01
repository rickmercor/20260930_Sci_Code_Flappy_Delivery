"""
Compute the single effective indistinguishability that governs every three-photon emission event (a pair in one arm plus a single photon in the other) when the noise photon only partly overlaps its own arm's signal mode.

In the paper's generalized noise model, a three-photon coincidence matrix at arbitrary signal-noise overlap is an incoherent mixture of an identical-noise configuration and a distinguishable-noise configuration. Because both configurations are linear in the indistinguishability, the mixture collapses onto the identical-noise form evaluated at one rescaled indistinguishability. The mixing weights are not a plain linear interpolation in the overlap: consult the paper's own weighting of the doubly occupied temporal mode and of the distinguishable-noise configuration.

Returns
-------
float: the effective indistinguishability eta_eff that replaces eta in all three-photon events.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def effective_indistinguishability(eta: float, m_sn: float) -> float:
    """Effective indistinguishability of three-photon events.

    Args:
        eta (float): intrinsic indistinguishability of the two arms' signal
            photons, 0 <= eta <= 1.
        m_sn (float): normalized overlap between a noise photon and its own
            arm's signal mode, 0 <= m_sn <= 1 (1 = identical noise,
            0 = fully distinguishable noise).

    Raises:
        ValueError: if eta or m_sn lies outside [0, 1].

    Expected return:
        float: eta_eff, proportional to eta, equal to eta for identical noise
        and strictly smaller than eta (for eta > 0) for any m_sn < 1.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: effective_indistinguishability
def _oracle_effective_indistinguishability(eta: float, m_sn: float) -> float:
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    if not (0.0 <= m_sn <= 1.0):
        raise ValueError("m_sn must lie in [0, 1]")
    return float((1.0 + 3.0 * m_sn) / (2.0 * (1.0 + m_sn)) * eta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': '',
            'call': 'round(effective_indistinguishability(0.96, 0.25), 8)',
            'gold_call': 'round(_oracle_effective_indistinguishability(0.96, 0.25), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(effective_indistinguishability(0.8, 1.0), 8)',
            'gold_call': 'round(_oracle_effective_indistinguishability(0.8, 1.0), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(effective_indistinguishability(0.8, 0.0), 8)',
            'gold_call': 'round(_oracle_effective_indistinguishability(0.8, 0.0), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(effective_indistinguishability(0.5, 0.6), 8)',
            'gold_call': 'round(_oracle_effective_indistinguishability(0.5, 0.6), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'def run_model():\n    try:\n        effective_indistinguishability(1.2, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_effective_indistinguishability(1.2, 0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'def run_model():\n    try:\n        effective_indistinguishability(0.9, -0.1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_effective_indistinguishability(0.9, -0.1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
