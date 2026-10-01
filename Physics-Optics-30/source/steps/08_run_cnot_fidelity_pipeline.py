"""
Chain the earlier steps end to end: decode eta from the measured visibility, confirm the decoded eta reproduces that visibility through the forward model, and evaluate the gate's Bell-state fidelity. The reference implementation calls the earlier public functions by name rather than reproducing their contents.

This step evaluates the task instance (V = 0.912, M_sn = 0.25, g2_m = 0.018, nbar_m = 0.42, g2_s = 0.041, nbar_s = 0.77). Every earlier convention meets here: the exact two-photon-truncated emission weights, the effective three-photon indistinguishability, the paper's gate and HOM event tables, and the exact visibility inversion.

Returns
-------
float: the Bell-state fidelity F of the gate for the given measured visibility and source statistics.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_cnot_fidelity_pipeline(v: float, m_sn: float, g2_m: float, nbar_m: float,
                               g2_s: float, nbar_s: float) -> float:
    """Measured (V, g2, nbar) -> decoded eta -> CNOT Bell-state fidelity.

    Args:
        v (float): measured two-source HOM visibility, 0 < v <= 1.
        m_sn (float): signal-noise overlap (same for both arms), 0 <= m_sn <= 1.
        g2_m (float): second-order coherence of the control arm.
        nbar_m (float): mean photon number per pulse of the control arm.
        g2_s (float): second-order coherence of the target arm.
        nbar_s (float): mean photon number per pulse of the target arm.

    Raises:
        ValueError: if any input is invalid, if the decoded eta exceeds 1, or
            if the decoded eta fails to reproduce v through the forward model.

    Expected return:
        float: the Bell-state fidelity F.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: run_cnot_fidelity_pipeline
def _oracle_run_cnot_fidelity_pipeline(v: float, m_sn: float, g2_m: float, nbar_m: float,
                                       g2_s: float, nbar_s: float) -> float:
    eta = _oracle_decode_intrinsic_eta(v, m_sn, g2_m, nbar_m, g2_s, nbar_s)
    v_check = _oracle_hom_visibility(eta, m_sn, g2_m, nbar_m, g2_s, nbar_s)
    if abs(v_check - v) > 1e-9:
        raise ValueError("visibility round trip failed")
    return _oracle_cnot_gate_fidelity(eta, m_sn, g2_m, nbar_m, g2_s, nbar_s)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': '',
            'call': 'round(run_cnot_fidelity_pipeline(0.912, 0.25, 0.018, 0.42, 0.041, 0.77), 8)',
            'gold_call': 'round(_oracle_run_cnot_fidelity_pipeline(0.912, 0.25, 0.018, 0.42, 0.041, 0.77), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(run_cnot_fidelity_pipeline(0.95, 1.0, 0.01, 0.3, 0.01, 0.3), 8)',
            'gold_call': 'round(_oracle_run_cnot_fidelity_pipeline(0.95, 1.0, 0.01, 0.3, 0.01, 0.3), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(run_cnot_fidelity_pipeline(0.80, 0.0, 0.05, 0.6, 0.02, 0.9), 8)',
            'gold_call': 'round(_oracle_run_cnot_fidelity_pipeline(0.80, 0.0, 0.05, 0.6, 0.02, 0.9), 8)',
            'tol': 1e-06,
        },
        {
            'setup': '',
            'call': 'round(run_cnot_fidelity_pipeline(0.97, 0.6, 0.003, 0.85, 0.03, 0.25), 8)',
            'gold_call': 'round(_oracle_run_cnot_fidelity_pipeline(0.97, 0.6, 0.003, 0.85, 0.03, 0.25), 8)',
            'tol': 1e-06,
        },
        {
            'setup': 'def run_model():\n    try:\n        run_cnot_fidelity_pipeline(0.999, 0.0, 0.2, 0.9, 0.2, 0.9)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_run_cnot_fidelity_pipeline(0.999, 0.0, 0.2, 0.9, 0.2, 0.9)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'def run_model():\n    try:\n        run_cnot_fidelity_pipeline(0.912, 0.25, 0.018, 0.0, 0.041, 0.77)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_run_cnot_fidelity_pipeline(0.912, 0.25, 0.018, 0.0, 0.041, 0.77)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
